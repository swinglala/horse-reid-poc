"""Phase 2 pipeline: white-marking segmentation of the Phase 1 selected face frames.

Reads ``<output>/frame_scores.json`` (Phase 1), re-reads the selected frames
from the video and writes everything under ``<output>/marking/``.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .config import resolve_project_path
from .device import select_device
from .marking import FaceRegionExtractor, build_marking_segmenter
from .marking.visualize import four_panel, mask_overlay
from .quality.metrics import yaw_bin
from .video import VideoReader
from .visualization import make_contact_sheet

logger = logging.getLogger(__name__)


@dataclass
class MarkingConfig:
    """Settings for :func:`run_phase2`."""

    output: Path = Path("data/output")
    input: Optional[Path] = None             # None -> video path recorded in frame_scores.json
    segmenter: str = "sam_refined"
    device: str = "cpu"
    sam_weights: Path = Path("models/mobile_sam.pt")
    seg_model: Path = Path("models/yolo11s-seg.pt")
    seg_imgsz: int = 416
    marking_weights: Optional[Path] = None   # yolo_seg only
    max_frames: Optional[int] = None
    margin: float = 0.15
    upscale: float = 2.0
    rotation: Optional[int] = None           # None -> rotation used by Phase 1
    segmenter_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output = Path(self.output)
        self.input = Path(self.input) if self.input is not None else None
        self.sam_weights = Path(self.sam_weights)
        self.seg_model = Path(self.seg_model)

    def to_dict(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


def _kps_to_crop(kps: Optional[dict], crop_bbox: Any, upscale: float) -> Optional[dict]:
    """Frame-coordinate part keypoints ``{name: [[x, y, score], ...]}`` -> crop/mask pixels (x upscale)."""
    if not kps:
        return None
    out: dict[str, list[list[float]]] = {}
    for k, v in kps.items():
        try:
            out[k] = [[(float(pt[0]) - crop_bbox[0]) * upscale, (float(pt[1]) - crop_bbox[1]) * upscale,
                       *[float(q) for q in pt[2:]]] for pt in v]
        except (TypeError, IndexError, ValueError):
            continue
    return out


def _selected_entries(fs: dict[str, Any]) -> list[dict[str, Any]]:
    """Score entry of the primary track group for every selected frame (in order)."""
    group = set(fs.get("primary_track_group") or [])
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in fs.get("scores", []):
        by_frame[int(s["frame"])].append(s)
    out = []
    for f in fs.get("selected", []):
        cands = by_frame.get(int(f), [])
        pref = [s for s in cands if s.get("track_id") in group] or cands
        if not pref:
            logger.warning("Selected frame %s has no score entry; skipped", f)
            continue
        out.append(max(pref, key=lambda s: s.get("score", 0.0)))
    return out


def _clear_outputs(md: Path) -> None:
    for sub, pat in (("masks", "frame_*.png"), ("prob", "frame_*.png"), ("overlays", "frame_*.jpg")):
        for old in (md / sub).glob(pat):
            old.unlink()


def run_phase2(cfg: MarkingConfig) -> dict[str, Any]:
    """Run marking segmentation on the Phase 1 selection; returns a summary dict."""
    t_start = time.time()
    timing: dict[str, float] = defaultdict(float)
    out = Path(cfg.output)
    fs_path = out / "frame_scores.json"
    if not fs_path.exists():
        raise FileNotFoundError(f"{fs_path} not found - run Phase 1 first")
    with open(fs_path) as f:
        fs = json.load(f)
    entries = _selected_entries(fs)
    if cfg.max_frames is not None:
        entries = entries[: cfg.max_frames]
    video = cfg.input or Path(fs.get("video", {}).get("path") or fs.get("config", {}).get("input"))
    video = resolve_project_path(video) if not Path(video).exists() else Path(video)
    rotation = cfg.rotation if cfg.rotation is not None else fs.get("config", {}).get("rotation")
    seg_imgsz = int(fs.get("config", {}).get("seg_imgsz", cfg.seg_imgsz))
    device = select_device(cfg.device)

    md = out / "marking"
    for sub in ("masks", "prob", "overlays"):
        (md / sub).mkdir(parents=True, exist_ok=True)
    _clear_outputs(md)

    t0 = time.time()
    seg_kw: dict[str, Any] = dict(cfg.segmenter_kwargs)
    if cfg.segmenter == "sam_refined":
        seg_kw.setdefault("sam_weights", str(resolve_project_path(cfg.sam_weights)))
    if cfg.segmenter == "yolo_seg" and cfg.marking_weights is not None:
        seg_kw.setdefault("weights_path", str(cfg.marking_weights))
    segmenter = build_marking_segmenter(cfg.segmenter, device=device, **seg_kw)
    extractor = FaceRegionExtractor(seg_model_path=cfg.seg_model, device=device, imgsz=seg_imgsz,
                                    upscale=cfg.upscale)
    timing["model_load"] += time.time() - t0

    results: list[dict[str, Any]] = []
    sheet_items = []
    with VideoReader(video, rotation=rotation) as reader:
        for e in tqdm(entries, desc="phase2", unit="f"):
            fidx = int(e["frame"])
            t0 = time.time()
            frame = reader.read_frame(fidx)
            timing["decode"] += time.time() - t0
            if frame is None:
                continue
            t0 = time.time()
            region = extractor.extract(frame, e.get("horse_bbox"), tuple(e["head_bbox"]), margin=cfg.margin)
            timing["face_region"] += time.time() - t0
            kps = _kps_to_crop((e.get("extra") or {}).get("keypoints"), region.crop_bbox, region.upscale)
            t0 = time.time()
            res = segmenter.predict(region.crop, region.face_mask, kps)
            timing["segment"] += time.time() - t0

            t0 = time.time()
            name = f"frame_{fidx:05d}"
            cv2.imwrite(str(md / "masks" / f"{name}.png"), (res.mask > 0).astype(np.uint8) * 255)
            cv2.imwrite(str(md / "prob" / f"{name}.png"), np.clip(res.prob * 255, 0, 255).astype(np.uint8))
            cv2.imwrite(str(md / "overlays" / f"{name}.jpg"), four_panel(region.crop, res),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            yaw = float(e.get("yaw", 0.0))
            rec = {
                "frame": fidx,
                "time_s": e.get("time_s"),
                "yaw": round(yaw, 2),
                "yaw_bin": yaw_bin(yaw),
                "crop_bbox": [int(v) for v in region.crop_bbox],
                "upscale": region.upscale,
                "face_mask_source": region.face_mask_source,
                "head_method": e.get("head_method"),
                **res.to_dict(),
            }
            results.append(rec)
            st = res.stats
            tile = mask_overlay(region.crop, res.mask, (0, 220, 0), 0.5)
            sheet_items.append((tile, [
                f"frame: {fidx}  t={float(e.get('time_s') or 0):.1f}s",
                f"yaw: {yaw:+.0f} ({yaw_bin(yaw)})",
                f"mark: {100 * st.get('marking_area_frac', 0):.1f}% / {st.get('n_components', 0)} comp",
                f"method: {res.method}",
            ]))
            timing["write"] += time.time() - t0

    t0 = time.time()
    make_contact_sheet(sheet_items, save_path=md / "marking_sheet.jpg")
    timing["write"] += time.time() - t0
    wall = time.time() - t_start
    timing["total_wall"] = wall

    with_mark = [r for r in results if r["stats"].get("n_components", 0) > 0]
    bins_all = Counter(r["yaw_bin"] for r in results)
    bins_mark = Counter(r["yaw_bin"] for r in with_mark)
    excl = Counter(x["reason"] for r in results for x in r["excluded"])
    summary: dict[str, Any] = {
        "segmenter": cfg.segmenter,
        "methods": dict(Counter(r["method"] for r in results)),
        "frames_selected": len(entries),
        "frames_processed": len(results),
        "frames_with_marking": len(with_mark),
        "mean_marking_area_frac": round(float(np.mean([r["stats"]["marking_area_frac"] for r in results])), 6)
        if results else None,
        "mean_marking_area_frac_marked": round(float(np.mean([r["stats"]["marking_area_frac"] for r in with_mark])), 6)
        if with_mark else None,
        "yaw_bins": {b: {"frames": bins_all[b], "with_marking": bins_mark.get(b, 0)} for b in bins_all},
        "excluded_reasons": dict(excl),
        "face_mask_sources": dict(Counter(r["face_mask_source"] for r in results)),
        "device": device,
        "wall_time_s": round(wall, 2),
        "timing_s": {k: round(v, 3) for k, v in timing.items()},
    }
    with open(md / "marking_results.json", "w") as f:
        json.dump({"config": cfg.to_dict(), "video": str(video), "summary": summary, "frames": results},
                  f, indent=1, ensure_ascii=False)
    _write_summary(md / "summary.txt", summary)
    logger.info("Phase 2 done in %.1fs -> %s", wall, md)
    return summary


def _write_summary(path: Path, s: dict[str, Any]) -> None:
    mean = s["mean_marking_area_frac"]
    mean_m = s["mean_marking_area_frac_marked"]
    lines = [
        "horse_reid Phase 2 (white-marking) summary",
        "==========================================",
        f"segmenter:            {s['segmenter']} (methods {s['methods']})",
        f"frames processed:     {s['frames_processed']} / {s['frames_selected']} selected",
        f"frames with marking:  {s['frames_with_marking']}",
        f"mean marking area:    {100 * mean:.2f}% of face (all frames)" if mean is not None else
        "mean marking area:    n/a",
        f"mean area (marked):   {100 * mean_m:.2f}% of face (frames with marking)" if mean_m is not None else
        "mean area (marked):   n/a",
        "yaw bins (marked/all): " + ", ".join(f"{b}: {v['with_marking']}/{v['frames']}"
                                             for b, v in sorted(s["yaw_bins"].items())),
        f"excluded regions:     {s['excluded_reasons']}",
        f"face mask sources:    {s['face_mask_sources']}",
        f"device:               {s['device']}",
        f"wall time:            {s['wall_time_s']}s",
        f"stage timing (s):     {s['timing_s']}",
    ]
    path.write_text("\n".join(lines) + "\n")
