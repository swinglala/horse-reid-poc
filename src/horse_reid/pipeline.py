"""Phase 1 pipeline: detect + track -> head -> quality -> selection -> outputs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, resolve_project_path
from .detection import YoloHorseDetector
from .device import select_device
from .face import build_head_detector
from .quality import FrameQualityScorer
from .quality.scorer import rescore
from .quality.metrics import yaw_bin
from .selection import default_min_gap, select_frames
from .tracking import TrackStore
from .types import Detection, FrameScore, bbox_area, crop, expand_bbox
from .video import VideoReader
from .visualization import draw_detections, make_contact_sheet

logger = logging.getLogger(__name__)

MAX_HORSES_PER_FRAME = 3  # cap head/quality work per frame


class _Timer:
    """Accumulates wall time per named stage."""

    def __init__(self) -> None:
        self.t: dict[str, float] = defaultdict(float)

    def add(self, name: str, dt: float) -> None:
        self.t[name] += dt

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.t.items()}


def _write_jpg(path: Path, img: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _score_label_lines(s: FrameScore) -> list[str]:
    return [
        f"frame: {s.frame}",
        f"time: {s.time_s:.2f}s",
        f"score: {s.score:.2f}",
        f"yaw: {s.yaw:+.0f} ({_bin_short(s.yaw)})",
        f"vis: {s.visibility:.2f}",
        f"blur: {s.blur:.2f}",
        f"occl: {s.occlusion:.2f}",
        f"conf: {s.det_conf:.2f}",
    ]


def _bin_short(yaw: float) -> str:
    return {"frontal": "front", "three_quarter": "3/4", "profile": "prof"}[yaw_bin(yaw)]


def auto_min_gap(span: int, n_scored: int, num_frames: int, head_stride: int) -> int:
    """Default min gap (in frame indices) between selected frames.

    :func:`default_min_gap` is applied to the number of *scored* frames and
    converted back to frame indices with the mean index step between scored
    frames; the result is at least ``head_stride``.
    """
    if n_scored <= 0:
        return max(head_stride, default_min_gap(span, num_frames))
    step = max(1.0, span / n_scored)
    return max(int(head_stride), int(round(default_min_gap(n_scored, num_frames) * step)))


def run_phase1(cfg: PipelineConfig) -> dict[str, Any]:
    """Run Phase 1 end to end and write all outputs under ``cfg.output``.

    Returns a small summary dict (also written to ``summary.txt``).
    """
    t_start = time.time()
    timer = _Timer()
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    # Remove stale images from a previous run into the same output dir.
    for sub in ("face_crops", "selected_frames", "selected_frames_raw", "all_face_crops"):
        for old in (out / sub).glob("frame_*.jpg"):
            old.unlink()
    device = select_device(cfg.device)
    logger.info("Device: %s", device)

    t0 = time.time()
    detector = YoloHorseDetector(cfg.model, device=device, tracker=cfg.tracker, conf=cfg.conf, imgsz=cfg.imgsz)
    head_detector = build_head_detector(cfg.head_detector, device=device,
                                        seg_weights=str(cfg.seg_model), seg_imgsz=cfg.seg_imgsz,
                                        cache_dir=str(resolve_project_path(cfg.hf_cache_dir)),
                                        fallback_name=cfg.head_fallback)
    scorer = FrameQualityScorer(cfg.weights, cfg.quality)
    timer.add("model_load", time.time() - t0)

    store = TrackStore()
    scores: list[FrameScore] = []
    n_processed = 0
    n_head_frames = 0  # frames on which the head stage ran (frame_idx % head_stride == 0)
    frames_with_horse = 0
    last_best: Optional[FrameScore] = None  # for drawing heads on non-scored video frames
    last_frame_idx = -1
    writer: Optional[cv2.VideoWriter] = None

    with VideoReader(cfg.input, stride=cfg.frame_stride, rotation=cfg.rotation,
                     max_frames=cfg.max_frames) as reader:
        video_info = {
            "path": str(cfg.input), "fps": reader.fps, "frame_count": reader.frame_count,
            "raw_size": [reader.raw_width, reader.raw_height],
            "size": [reader.width, reader.height],
            "rotation_meta": reader.metadata_rotation, "rotation_applied": reader.rotation,
            "duration_s": round(reader.duration_s, 3),
        }
        if cfg.visualize:
            vpath = out / "annotated.mp4"
            writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"),
                                     reader.fps / cfg.frame_stride, (reader.width, reader.height))
            if not writer.isOpened():
                logger.warning("Could not open video writer for %s", vpath)
                writer = None
        detector.reset()

        t_read = time.time()
        for idx, ts, frame in tqdm(reader, total=len(reader), desc="phase1", unit="f"):
            timer.add("decode", time.time() - t_read)
            n_processed += 1
            last_frame_idx = idx

            t0 = time.time()
            dets = detector.track(frame, idx, ts)
            timer.add("detect_track", time.time() - t0)
            store.add(dets)

            horses = [d for d in dets if d.cls_name == "horse"]
            persons = [d for d in dets if d.cls_name == "person"]
            if horses:
                frames_with_horse += 1
            tracked = sorted((d for d in horses if d.track_id is not None),
                             key=lambda d: -bbox_area(d.bbox))[:MAX_HORSES_PER_FRAME]

            frame_scores: list[tuple[FrameScore, Detection]] = []
            run_head = idx % cfg.head_stride == 0
            if run_head:
                n_head_frames += 1
            for horse in (tracked if run_head else []):
                t0 = time.time()
                head = head_detector.detect(frame, horse.bbox, horse.confidence)
                timer.add("head", time.time() - t0)
                if head is None:
                    continue
                t0 = time.time()
                others = [p.bbox for p in persons] + [h.bbox for h in horses if h is not horse]
                fs = scorer.score(frame, horse, head, others)
                timer.add("quality", time.time() - t0)
                scores.append(fs)
                frame_scores.append((fs, horse))
                if cfg.save_all:
                    t0 = time.time()
                    cb = expand_bbox(head.bbox, cfg.head_crop_margin, frame.shape[1], frame.shape[0])
                    _write_jpg(out / "all_face_crops" / f"frame_{idx:05d}_t{horse.track_id}.jpg", crop(frame, cb))
                    timer.add("write", time.time() - t0)

            if writer is not None:
                t0 = time.time()
                if run_head:
                    last_best = max(frame_scores, key=lambda x: x[0].score)[0] if frame_scores else None
                # Between scored frames, keep showing the last head result (same track only).
                best = last_best
                if best is not None and not run_head and best.track_id not in {h.track_id for h in horses}:
                    best = None
                lines = [f"frame {idx}  t={ts:.2f}s"]
                if best is not None:
                    held = "" if best.frame == idx else f"  (head @ {best.frame})"
                    lines.append(f"score {best.score:.2f} yaw {best.yaw:+.0f} ({_bin_short(best.yaw)})"
                                 f" vis {best.visibility:.2f}{held}")
                    lines.append(f"blur {best.blur:.2f} exp {best.exposure:.2f} occl {best.occlusion:.2f}")
                img = draw_detections(frame, horses, persons, best.head_bbox if best else None, lines,
                                      primary_track_id=best.track_id if best else None, copy=False,
                                      keypoints=best.extra.get("keypoints") if best else None)
                writer.write(img)
                timer.add("visualize", time.time() - t0)
            t_read = time.time()

        if writer is not None:
            writer.release()

        # ---------------- selection ---------------- #
        t0 = time.time()
        span = (last_frame_idx + 1) if last_frame_idx >= 0 else 0
        sel = _select(store, scores, cfg, span, n_head_frames)
        timer.add("select", time.time() - t0)

        # ---------------- outputs ---------------- #
        t0 = time.time()
        _write_selection_outputs(out, reader, store, sel.selected, cfg.head_crop_margin)
        timer.add("write", time.time() - t0)

    wall = time.time() - t_start
    timer.add("total_wall", wall)

    # ---------------- reports ---------------- #
    fps_proc = n_processed / wall if wall > 0 else 0.0
    summary: dict[str, Any] = {
        "frames_processed": n_processed,
        "frames_with_horse": frames_with_horse,
        "head_stride": cfg.head_stride,
        "frames_head_stage": n_head_frames,
        "span_frames": span,
        **_selection_summary(store, sel, cfg),
        "video_duration_s": video_info["duration_s"],
        "device": device,
        "wall_time_s": round(wall, 2),
        "processed_fps": round(fps_proc, 2),
    }
    with open(out / "frame_scores.json", "w") as f:
        json.dump({
            "video": video_info,
            "config": cfg.to_dict(),
            "device": device,
            "primary_track_id": sel.primary,
            "primary_track_group": sorted(sel.group),
            "min_gap_frames": sel.min_gap,
            "head_stride": cfg.head_stride,
            "selected": [s.frame for s in sel.selected],
            "timing_s": timer.as_dict(),
            "summary": summary,
            "scores": [s.to_dict() for s in scores],
        }, f, indent=1)
    _write_summary(out / "summary.txt", summary, video_info, timer.as_dict())
    logger.info("Phase 1 done in %.1fs -> %s", wall, out)
    return summary


# --------------------------------------------------------------------------- #
# Selection + outputs shared by run_phase1 and run_reselect
# --------------------------------------------------------------------------- #
@dataclass
class _Selection:
    primary: Optional[int]
    group: list[int]
    pool: list[FrameScore]
    min_gap: int
    selected: list[FrameScore]


def _candidate_pool(scores: list[FrameScore], group: set[int]) -> list[FrameScore]:
    """Scores of the primary track group, keeping the best one per frame."""
    best: dict[int, FrameScore] = {}
    for s in scores:
        if s.track_id in group and (s.frame not in best or s.score > best[s.frame].score):
            best[s.frame] = s
    return [best[f] for f in sorted(best)]


def _select(store: TrackStore, scores: list[FrameScore], cfg: PipelineConfig,
            span: int, n_head_frames: int) -> _Selection:
    primary = store.primary_horse_track()
    group = store.primary_track_group()
    pool = _candidate_pool(scores, set(group))
    min_gap = (cfg.min_frame_gap if cfg.min_frame_gap is not None
               else auto_min_gap(span, n_head_frames, cfg.num_frames, cfg.head_stride))
    selected = select_frames(pool, cfg.num_frames, min_gap_frames=min_gap, total_frames=span,
                             min_score=cfg.min_score)
    logger.info("Primary track %s (group %s): %d candidate frames -> %d of %d requested selected "
                "(min_gap=%d, min_score=%.2f)", primary, group, len(pool), len(selected),
                cfg.num_frames, min_gap, cfg.min_score)
    return _Selection(primary, group, pool, min_gap, selected)


def _write_selection_outputs(out: Path, reader: VideoReader, store: TrackStore,
                             selected: list[FrameScore], margin: float) -> None:
    """Write ``detections.json``, ``face_crops/``, ``selected_frames(_raw)/`` and
    ``contact_sheet.jpg`` for ``selected`` (old ``frame_*.jpg`` are removed first)."""
    for sub in ("face_crops", "selected_frames", "selected_frames_raw"):
        for old in (out / sub).glob("frame_*.jpg"):
            old.unlink()
    store.export(out / "detections.json")
    sheet_items = []
    for s in selected:
        frame = reader.read_frame(s.frame)
        if frame is None:
            logger.warning("Could not re-read frame %d", s.frame)
            continue
        cb = expand_bbox(s.head_bbox, margin, frame.shape[1], frame.shape[0])
        head_crop = crop(frame, cb).copy()
        _write_jpg(out / "face_crops" / f"frame_{s.frame:05d}.jpg", head_crop)
        _write_jpg(out / "selected_frames_raw" / f"frame_{s.frame:05d}.jpg", frame)
        dets = store.frame_detections(s.frame)
        ann = draw_detections(
            frame, [d for d in dets if d.cls_name == "horse"], [d for d in dets if d.cls_name == "person"],
            s.head_bbox, [f"frame {s.frame}  t={s.time_s:.2f}s  score {s.score:.2f}",
                          f"yaw {s.yaw:+.0f} ({_bin_short(s.yaw)})  vis {s.visibility:.2f}",
                          f"head: {s.head_method}"],
            primary_track_id=s.track_id, keypoints=s.extra.get("keypoints"),
        )
        _write_jpg(out / "selected_frames" / f"frame_{s.frame:05d}.jpg", ann)
        sheet_items.append((head_crop, _score_label_lines(s)))
    make_contact_sheet(sheet_items, save_path=out / "contact_sheet.jpg")


def _selection_summary(store: TrackStore, sel: _Selection, cfg: PipelineConfig) -> dict[str, Any]:
    """Summary fields that depend on the track grouping and the selection."""
    pool, selected = sel.pool, sel.selected
    return {
        "frames_with_head": len({s.frame for s in pool}),
        "selected": len(selected),
        "num_frames_requested": cfg.num_frames,
        "min_score": cfg.min_score,
        "candidates_above_min_score": sum(1 for s in pool if s.score >= cfg.min_score),
        "selected_frames": [s.frame for s in selected],
        "primary_track_id": sel.primary,
        "primary_track_length": store.track_length(sel.primary) if sel.primary is not None else 0,
        "primary_track_group": sorted(sel.group),
        "horse_tracks": store.horse_track_ids(),
        "min_gap_frames": sel.min_gap,
        "selected_time_range_s": ([round(selected[0].time_s, 2), round(selected[-1].time_s, 2)]
                                  if selected else None),
        "mean_score_all": round(float(np.mean([s.score for s in pool])), 4) if pool else None,
        "mean_score_selected": round(float(np.mean([s.score for s in selected])), 4) if selected else None,
        "mean_visibility_all": round(float(np.mean([s.visibility for s in pool])), 4) if pool else None,
        "mean_visibility_selected": (round(float(np.mean([s.visibility for s in selected])), 4)
                                     if selected else None),
        "yaw_sources": dict(Counter(s.extra.get("yaw_source", "?") for s in pool)),
        "yaw_bins_candidates": dict(Counter(yaw_bin(s.yaw) for s in pool)),
        "yaw_bins_selected": dict(Counter(yaw_bin(s.yaw) for s in selected)),
        "head_methods": dict(Counter(s.head_method for s in pool)),
        "head_methods_frames": {m: len(fr) for m, fr in _frames_per_method(pool).items()},
    }


def run_reselect(cfg: PipelineConfig) -> dict[str, Any]:
    """Re-run track grouping + frame selection on an existing Phase 1 output.

    Reads ``<output>/detections.json`` and ``<output>/frame_scores.json`` (no
    detection / head / scoring pass), applies the current track logic and
    ``cfg.num_frames`` / ``cfg.min_frame_gap`` / ``cfg.min_score``, re-reads the
    selected frames from ``cfg.input`` and rewrites the same outputs as
    :func:`run_phase1`. ``scores`` and the original timing are preserved.
    """
    t_start = time.time()
    out = Path(cfg.output)
    fs_path, det_path = out / "frame_scores.json", out / "detections.json"
    with open(fs_path) as f:
        fs_doc = json.load(f)
    with open(det_path) as f:
        det_doc = json.load(f)

    store = TrackStore.from_detection_dicts(det_doc.get("detections", []))
    scores = [FrameScore.from_dict(d) for d in fs_doc.get("scores", [])]
    orig: dict[str, Any] = dict(fs_doc.get("summary") or {})
    orig_cfg: dict[str, Any] = fs_doc.get("config") or {}
    video_info = dict(fs_doc.get("video") or {})
    video_info["path"] = str(cfg.input)

    # Keep the Phase 1 sampling parameters; only selection parameters come from cfg.
    head_stride = int(fs_doc.get("head_stride") or orig.get("head_stride") or orig_cfg.get("head_stride") or 1)
    cfg = replace(cfg, head_stride=head_stride)
    n_head_frames = int(orig.get("frames_head_stage") or len({s.frame for s in scores}))
    span = orig.get("span_frames")
    if not span:
        n_proc = orig.get("frames_processed")
        stride = int(orig_cfg.get("frame_stride") or 1)
        span = ((n_proc - 1) * stride + 1 if n_proc else
                max([d.frame for d in store.detections] + [s.frame for s in scores] + [-1]) + 1)
    span = int(span)

    for fs in scores:  # apply current weights / gates without re-running detection
        rescore(fs, cfg.quality, cfg.weights)
    sel = _select(store, scores, cfg, span, n_head_frames)
    rotation = cfg.rotation if cfg.rotation is not None else video_info.get("rotation_applied")
    with VideoReader(cfg.input, rotation=rotation) as reader:
        _write_selection_outputs(out, reader, store, sel.selected, cfg.head_crop_margin)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    original_run = orig.get("original_run") or {
        "wall_time_s": orig.get("wall_time_s"), "processed_fps": orig.get("processed_fps"),
        "device": orig.get("device"), "primary_track_id": fs_doc.get("primary_track_id"),
        "primary_track_group": fs_doc.get("primary_track_group"), "selected": orig.get("selected"),
    }
    summary: dict[str, Any] = {**orig, **_selection_summary(store, sel, cfg)}
    summary.update({
        "span_frames": span,
        "mode": "reselect",
        "reselected_utc": now,
        "original_run": original_run,
        "reselect_wall_time_s": round(time.time() - t_start, 2),
    })
    summary.setdefault("video_duration_s", video_info.get("duration_s"))
    fs_doc.update({
        "primary_track_id": sel.primary,
        "primary_track_group": sorted(sel.group),
        "min_gap_frames": sel.min_gap,
        "selected": [s.frame for s in sel.selected],
        "summary": summary,
        "reselected_utc": now,
        "reselect": {"input": str(cfg.input), "num_frames": cfg.num_frames,
                     "min_frame_gap": cfg.min_frame_gap, "min_score": cfg.min_score,
                     "rescored": True},
        "scores": [s.to_dict() for s in scores],
    })
    with open(fs_path, "w") as f:
        json.dump(fs_doc, f, indent=1)
    _write_summary(out / "summary.txt", summary, video_info, fs_doc.get("timing_s") or {})
    logger.info("Reselect done in %.1fs -> %s (Phase 2/3 outputs, if any, are now stale)",
                summary["reselect_wall_time_s"], out)
    return summary


def _frames_per_method(scores: list[FrameScore]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for s in scores:
        out[s.head_method].add(s.frame)
    return dict(out)


def _write_summary(path: Path, summary: dict[str, Any], video: dict[str, Any], timing: dict[str, float]) -> None:
    cov = summary["selected_time_range_s"]
    dur = summary["video_duration_s"] or 0
    cov_txt = (f"{cov[0]:.2f}s - {cov[1]:.2f}s ({(cov[1] - cov[0]) / dur * 100:.0f}% of {dur:.1f}s video)"
               if cov and dur else "n/a")
    reselect = summary.get("mode") == "reselect"
    req = summary.get("num_frames_requested")
    sel_txt = str(summary["selected"])
    if req is not None:
        sel_txt += (f" of {req} requested (min_score {summary.get('min_score')}: "
                    f"{summary.get('candidates_above_min_score')} of {summary['frames_with_head']} candidates pass)")
    lines = [
        "horse_reid Phase 1 summary",
        "==========================",
    ]
    if reselect:
        lines.append(f"mode:                 reselect at {summary.get('reselected_utc')} "
                     "(selection re-run on stored detections/scores; detection not re-run)")
        lines.append("scores:               rescored with current weights")
    lines += [
        f"input:                {video['path']}",
        f"video:                {video['size'][0]}x{video['size'][1]} (rotation {video['rotation_applied']}), "
        f"{video['fps']:.2f} fps, {video['frame_count']} frames",
        f"device:               {summary['device']}",
        f"frames processed:     {summary['frames_processed']}",
        f"frames with horse:    {summary['frames_with_horse']}",
        f"head stride:          {summary['head_stride']} ({summary['frames_head_stage']} frames ran the head stage)",
        f"frames with head:     {summary['frames_with_head']} (primary track group)",
        f"selected frames:      {sel_txt}",
        f"primary track id:     {summary['primary_track_id']} (length {summary['primary_track_length']}, "
        f"group {summary['primary_track_group']}, all horse tracks {summary['horse_tracks']})",
        f"min gap (frames):     {summary['min_gap_frames']}",
        f"selected time range:  {cov_txt}",
        f"mean score (cands):   {summary['mean_score_all']}",
        f"mean score (selected):{summary['mean_score_selected']}",
        f"yaw bins candidates:  {summary['yaw_bins_candidates']}",
        f"yaw bins selected:    {summary['yaw_bins_selected']}",
        f"head methods:         {summary['head_methods_frames']} (frames per method, primary track group)",
        f"yaw sources:          {summary['yaw_sources']}",
        f"mean visibility:      cands {summary['mean_visibility_all']}, selected {summary['mean_visibility_selected']}",
    ]
    if reselect:
        orig = summary.get("original_run") or {}
        lines += [
            f"original run:         wall time {orig.get('wall_time_s')}s ({orig.get('processed_fps')} frames/s), "
            f"primary track {orig.get('primary_track_id')}, {orig.get('selected')} selected",
            f"reselect wall time:   {summary.get('reselect_wall_time_s')}s",
            f"stage timing (s):     original run {timing}",
        ]
    else:
        lines += [
            f"wall time:            {summary['wall_time_s']}s ({summary['processed_fps']} frames/s)",
            f"stage timing (s):     {timing}",
        ]
    path.write_text("\n".join(lines) + "\n")


def run_phase2(cfg: Any) -> dict[str, Any]:
    """Phase 2 (white-marking segmentation); see :mod:`horse_reid.marking_pipeline`.

    ``cfg`` is a :class:`horse_reid.marking_pipeline.MarkingConfig`.
    """
    from .marking_pipeline import run_phase2 as _run_phase2

    return _run_phase2(cfg)
