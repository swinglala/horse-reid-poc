"""Load + join Phase 1 (``frame_scores.json``) and Phase 2 (``marking/``) outputs.

Input contract::

    <output>/frame_scores.json            {"selected": [...], "scores": [FrameScore dicts]}
    <output>/marking/marking_results.json list | {"frames"|"results"|"entries": list} | {frame: entry}
                                          entries: frame, crop_bbox [x1,y1,x2,y2], upscale, stats, method
    <output>/marking/masks/frame_XXXXX.png  0/255, size = crop size x upscale
    <output>/marking/prob/frame_XXXXX.png   0..255 (optional)

Keypoints (frame coords) are converted into mask-pixel coordinates with
``((x - crop_x1) * upscale, (y - crop_y1) * upscale)``. Every problem with a
single frame is logged and that frame is skipped.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JoinedFrame:
    frame: int
    time_s: float
    yaw: float
    score: float
    view: Any
    head_bbox: Optional[list[float]]          # frame coords
    crop_bbox: list[float]                    # frame coords
    upscale: float
    mask: np.ndarray                          # uint8 HxW 0/255
    prob: Optional[np.ndarray]                # uint8 HxW 0..255 or None
    keypoints_px: dict[str, list[list[float]]] = field(default_factory=dict)
    head_bbox_px: Optional[list[float]] = None
    selected: bool = False
    method: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    mask_path: Optional[Path] = None


def to_mask_px(x: float, y: float, crop_bbox, upscale: float) -> tuple[float, float]:
    return ((float(x) - float(crop_bbox[0])) * upscale, (float(y) - float(crop_bbox[1])) * upscale)


def keypoints_to_mask_px(keypoints: Optional[dict], crop_bbox, upscale: float) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for part, pts in (keypoints or {}).items():
        rows = []
        for p in pts or []:
            if p is None or len(p) < 2:
                continue
            x, y = to_mask_px(p[0], p[1], crop_bbox, upscale)
            rows.append([x, y, float(p[2]) if len(p) > 2 else 1.0])
        out[str(part)] = rows
    return out


def _marking_entries(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    if isinstance(data, dict):
        for key in ("frames", "results", "entries", "per_frame", "items"):
            if isinstance(data.get(key), list):
                return [e for e in data[key] if isinstance(e, dict)]
        out = []
        for k, v in data.items():
            if isinstance(v, dict):
                e = dict(v)
                if "frame" not in e:
                    try:
                        e["frame"] = int(k)
                    except (TypeError, ValueError):
                        continue
                out.append(e)
        return out
    return []


def _resolve(marking_dir: Path, entry: dict, keys: tuple[str, ...], sub: str, frame: int) -> Path:
    for k in keys:
        v = entry.get(k)
        if isinstance(v, str) and v:
            p = Path(v)
            for cand in (p, marking_dir / p, marking_dir.parent / p):
                if cand.exists():
                    return cand
    return marking_dir / sub / f"frame_{frame:05d}.png"


def load_frame_scores(output_dir: str | Path) -> tuple[list[int], dict[int, dict]]:
    """``(selected_frames, {frame: score_dict})`` from ``frame_scores.json``."""
    p = Path(output_dir) / "frame_scores.json"
    with open(p) as f:
        data = json.load(f)
    selected = [int(v) for v in (data.get("selected") or [])]
    scores: dict[int, dict] = {}
    for s in data.get("scores") or []:
        try:
            scores[int(s["frame"])] = s
        except (KeyError, TypeError, ValueError):
            logger.warning("frame_scores: entry without a frame index skipped")
    return selected, scores


def load_joined(output_dir: str | Path, selected_only: bool = False) -> list[JoinedFrame]:
    """Join marking results with frame scores by frame index (sorted by frame)."""
    out_dir = Path(output_dir)
    selected, scores = load_frame_scores(out_dir)
    sel_set = set(selected)
    mdir = out_dir / "marking"
    with open(mdir / "marking_results.json") as f:
        entries = _marking_entries(json.load(f))
    joined: list[JoinedFrame] = []
    for e in entries:
        try:
            fr = int(e["frame"])
        except (KeyError, TypeError, ValueError):
            logger.warning("marking entry without frame index skipped: %s", list(e)[:6])
            continue
        if selected_only and fr not in sel_set:
            continue
        if any(j.frame == fr for j in joined):
            logger.warning("frame %d: duplicate marking entry ignored", fr)
            continue
        s = scores.get(fr)
        if s is None:
            logger.warning("frame %d: no FrameScore -> skipped", fr)
            continue
        cb = e.get("crop_bbox")
        if not cb or len(cb) < 4:
            logger.warning("frame %d: marking entry has no crop_bbox -> skipped", fr)
            continue
        try:
            up = float(e.get("upscale", 1.0) or 1.0)
        except (TypeError, ValueError):
            up = 1.0
        mpath = _resolve(mdir, e, ("mask_path", "mask"), "masks", fr)
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) if mpath.exists() else None
        if mask is None:
            logger.warning("frame %d: mask %s missing/unreadable -> skipped", fr, mpath)
            continue
        ppath = _resolve(mdir, e, ("prob_path", "prob"), "prob", fr)
        prob = cv2.imread(str(ppath), cv2.IMREAD_GRAYSCALE) if ppath.exists() else None
        if prob is None:
            logger.info("frame %d: no prob map, using the binary mask", fr)
        elif prob.shape != mask.shape:
            prob = cv2.resize(prob, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        kps = (s.get("extra") or {}).get("keypoints")
        hb = s.get("head_bbox")
        hb_px = None
        if hb and len(hb) >= 4:
            x1, y1 = to_mask_px(hb[0], hb[1], cb, up)
            x2, y2 = to_mask_px(hb[2], hb[3], cb, up)
            hb_px = [x1, y1, x2, y2]
        try:
            joined.append(JoinedFrame(
                frame=fr, time_s=float(s.get("time_s", 0.0)), yaw=float(s.get("yaw", 0.0)),
                score=float(s.get("score", 0.0)), view=s.get("view"), head_bbox=hb,
                crop_bbox=[float(v) for v in cb[:4]], upscale=up, mask=mask, prob=prob,
                keypoints_px=keypoints_to_mask_px(kps if isinstance(kps, dict) else None, cb, up),
                head_bbox_px=hb_px, selected=fr in sel_set, method=str(e.get("method", "")),
                stats=e.get("stats") or {}, mask_path=mpath))
        except (TypeError, ValueError) as exc:
            logger.warning("frame %d: malformed entry (%s) -> skipped", fr, exc)
    joined.sort(key=lambda j: j.frame)
    logger.info("canonical io: %d marking entries -> %d joined frames", len(entries), len(joined))
    return joined
