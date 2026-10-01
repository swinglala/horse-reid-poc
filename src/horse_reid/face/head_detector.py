"""Horse head detectors.

This module defines the :class:`HorseHeadDetector` interface, two HEURISTIC
detectors (``bbox_top``, ``mask_top``) and the :func:`build_head_detector`
registry. The real (open-vocabulary) detector is Grounding DINO, implemented
in ``grounded_head_detector.py`` and registered as ``"grounding_dino"``; the
heuristics remain selectable and serve as its fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..types import BBox, HeadDetection, bbox_height, bbox_width, clip_bbox, clip_bbox_to, expand_bbox

logger = logging.getLogger(__name__)

COCO_HORSE = 17


class HorseHeadDetector(ABC):
    """Locate the head of a horse given the frame and the horse's bbox."""

    name: str = "base"

    @abstractmethod
    def detect(self, frame: np.ndarray, horse_bbox: BBox, horse_conf: float = 1.0) -> Optional[HeadDetection]:
        """Return the head detection in frame coordinates, or ``None``.

        Args:
            frame: Full BGR frame (upright).
            horse_bbox: ``(x1, y1, x2, y2)`` of the horse in ``frame``.
            horse_conf: Detector confidence of the horse box (used to derive
                the head confidence for heuristic methods).
        """


# --------------------------------------------------------------------------- #
# Heuristic 1: top band of the horse bbox
# --------------------------------------------------------------------------- #
class BBoxTopHeadDetector(HorseHeadDetector):
    """TEMPORARY FALLBACK: head = upper band of the horse bounding box.

    # TODO(phase1-upgrade): replace with a real head detector (Grounding DINO /
    # OWLv2 / fine-tuned YOLO head model). This heuristic assumes an upright
    # horse whose head is the highest part of the body, which fails for grazing
    # horses (head down) and for frontal close-ups.

    The band is the top ``top_fraction`` of the horse box. Left/right half
    selection: if a horse mask is given, pick the half with higher mask
    density in the band; without a mask, use Canny edge density but only if
    one side is clearly denser (``side_ratio``), otherwise keep full width.
    """

    name = "bbox_top_heuristic"

    def __init__(self, top_fraction: float = 0.42, side_width_fraction: float = 0.6,
                 side_ratio: float = 1.5, conf_factor: float = 0.5) -> None:
        self.top_fraction = top_fraction
        self.side_width_fraction = side_width_fraction
        self.side_ratio = side_ratio
        self.conf_factor = conf_factor

    def _choose_side(self, band: np.ndarray, from_mask: bool) -> Optional[str]:
        """Return "left", "right" or None (= full width)."""
        if band.size == 0 or band.shape[1] < 4:
            return None
        if from_mask:
            density = band.astype(np.float32)
        else:
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
            density = (cv2.Canny(gray, 50, 150) > 0).astype(np.float32)
        mid = density.shape[1] // 2
        left, right = float(density[:, :mid].mean()), float(density[:, mid:].mean())
        # TODO(phase1-upgrade): density-based side choice is a weak cue.
        if from_mask:
            if max(left, right) <= 0:
                return None
            return "left" if left >= right else "right"
        lo, hi = min(left, right), max(left, right)
        if hi <= 0 or (lo > 0 and hi / lo < self.side_ratio):
            return None
        return "left" if left > right else "right"

    def detect(self, frame: np.ndarray, horse_bbox: BBox, horse_conf: float = 1.0,
               mask: Optional[np.ndarray] = None) -> Optional[HeadDetection]:
        """See class docstring. ``mask`` (optional) is a bool mask of the horse
        box region (shape = horse box h x w)."""
        h_img, w_img = frame.shape[:2]
        hb = clip_bbox(horse_bbox, w_img, h_img)
        bw, bh = bbox_width(hb), bbox_height(hb)
        if bw < 4 or bh < 4:
            return None
        band_h = max(2, int(round(bh * self.top_fraction)))
        x1, y1, x2 = hb[0], hb[1], hb[2]
        if mask is not None and mask.shape[:2] == (bh, bw):
            side = self._choose_side(mask[:band_h], from_mask=True)
        else:
            side = self._choose_side(frame[y1:y1 + band_h, x1:x2], from_mask=False)
        sw = int(round(bw * self.side_width_fraction))
        if side == "left":
            x2 = x1 + sw
        elif side == "right":
            x1 = x2 - sw
        box = clip_bbox_to((x1, y1, x2, y1 + band_h), hb)
        return HeadDetection(bbox=box, confidence=float(horse_conf) * self.conf_factor,
                             method=self.name, keypoints=None)


# --------------------------------------------------------------------------- #
# Heuristic 2: topmost extremity of the instance mask
# --------------------------------------------------------------------------- #
class MaskTopHeadDetector(HorseHeadDetector):
    """HEURISTIC: head = region below the topmost extremity of the horse mask.

    # TODO(phase1-upgrade): replace with a real head detector. The "highest
    # point of the mask is the poll/ears" assumption only holds for a standing
    # horse with its head up; grazing horses and riders break it.

    The YOLO11 segmentation model is run on the (slightly padded) horse crop
    only, at a small ``imgsz``, to keep CPU cost low. ``last_mask`` keeps the
    horse mask (bool, horse-bbox sized) of the most recent call and
    ``last_mask_bbox`` the frame-coordinate box it covers, for Phase 2.
    """

    name = "mask_top_heuristic"

    def __init__(
        self,
        weights: str | Path = "models/yolo11s-seg.pt",
        device: str = "cpu",
        imgsz: int = 416,
        conf: float = 0.2,
        crop_pad: float = 0.05,
        top_band_fraction: float = 0.12,
        head_height_fraction: float = 0.40,
        head_width_ratio: float = 0.55,
        conf_factor: float = 0.7,
        fallback: Optional[BBoxTopHeadDetector] = None,
    ) -> None:
        from ultralytics import YOLO

        weights = Path(weights)
        if weights.parent and str(weights.parent) not in ("", "."):
            weights.parent.mkdir(parents=True, exist_ok=True)
        # Ultralytics downloads known asset names to the given path if missing.
        self.model = YOLO(str(weights))
        if getattr(self.model, "task", None) != "segment":
            raise ValueError(f"{weights} is not a segmentation model (task={self.model.task})")
        self.device = device
        self.fixed_shape = not str(device).startswith("cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.crop_pad = crop_pad
        self.top_band_fraction = top_band_fraction
        self.head_height_fraction = head_height_fraction
        self.head_width_ratio = head_width_ratio
        self.conf_factor = conf_factor
        self.fallback = fallback or BBoxTopHeadDetector()
        self.last_mask: Optional[np.ndarray] = None
        self.last_mask_bbox: Optional[BBox] = None
        logger.info("Loaded head-seg model %s (imgsz=%d) on %s", weights, imgsz, device)

    def horse_mask(self, frame: np.ndarray, horse_bbox: BBox) -> Optional[np.ndarray]:
        """Segment the horse inside ``horse_bbox``; returns a bool mask with the
        shape of the horse box, or ``None`` if no horse instance is found."""
        h_img, w_img = frame.shape[:2]
        hb = clip_bbox(horse_bbox, w_img, h_img)
        bw, bh = bbox_width(hb), bbox_height(hb)
        if bw < 8 or bh < 8:
            return None
        pb = expand_bbox(hb, self.crop_pad, w_img, h_img)
        crop_img = np.ascontiguousarray(frame[pb[1]:pb[3], pb[0]:pb[2]])
        if self.fixed_shape:
            # Letterbox to a fixed square input ourselves: a constant input shape
            # avoids per-shape kernel recompilation on GPU backends (notably MPS).
            ch, cw = crop_img.shape[:2]
            scale = self.imgsz / max(ch, cw)
            nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
            net_in = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            net_in[:nh, :nw] = cv2.resize(crop_img, (nw, nh), interpolation=interp)
        else:
            # CPU: let Ultralytics use its minimal rectangular letterbox (fewer pixels).
            scale, net_in = 1.0, crop_img
        r = self.model.predict(net_in, imgsz=self.imgsz, conf=self.conf, classes=[COCO_HORSE],
                               device=self.device, verbose=False, retina_masks=False)[0]
        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return None
        # Horse box in crop coordinates; pick the instance that best overlaps it.
        target = np.array([hb[0] - pb[0], hb[1] - pb[1], hb[2] - pb[0], hb[3] - pb[1]], dtype=np.float32)
        boxes = r.boxes.xyxy.cpu().numpy() / scale
        ix1 = np.maximum(boxes[:, 0], target[0]); iy1 = np.maximum(boxes[:, 1], target[1])
        ix2 = np.minimum(boxes[:, 2], target[2]); iy2 = np.minimum(boxes[:, 3], target[3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        t_area = (target[2] - target[0]) * (target[3] - target[1])
        iou = inter / np.maximum(areas + t_area - inter, 1e-6)
        best = int(np.argmax(iou))
        if iou[best] < 0.3:
            return None
        poly = r.masks.xy[best]
        if poly is None or len(poly) < 3:
            return None
        crop_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(crop_mask, [np.round(np.asarray(poly) / scale).astype(np.int32)], 1)
        ox, oy = hb[0] - pb[0], hb[1] - pb[1]
        mask = crop_mask[oy:oy + bh, ox:ox + bw].astype(bool)
        return mask if mask.any() else None

    def detect(self, frame: np.ndarray, horse_bbox: BBox, horse_conf: float = 1.0) -> Optional[HeadDetection]:
        h_img, w_img = frame.shape[:2]
        hb = clip_bbox(horse_bbox, w_img, h_img)
        self.last_mask, self.last_mask_bbox = None, None
        mask = self.horse_mask(frame, hb)
        if mask is None:
            det = self.fallback.detect(frame, hb, horse_conf)
            if det is not None:
                det.method = f"{self.fallback.name}(no_mask)"
            return det
        self.last_mask, self.last_mask_bbox = mask, hb
        bh = bbox_height(hb)
        rows = np.flatnonzero(mask.any(axis=1))
        top, bottom = int(rows[0]), int(rows[-1])
        mask_h = bottom - top + 1
        band_end = top + max(1, int(round(self.top_band_fraction * mask_h)))
        ys, xs = np.nonzero(mask[top:band_end])
        cx = float(xs.mean()) if xs.size else mask.shape[1] / 2.0
        # TODO(phase1-upgrade): fixed head proportions relative to the horse box.
        head_h = self.head_height_fraction * bh
        head_w = self.head_width_ratio * head_h
        x1 = hb[0] + cx - head_w / 2.0
        y1 = hb[1] + top
        box = (int(round(x1)), int(round(y1)), int(round(x1 + head_w)), int(round(y1 + head_h)))
        box = clip_bbox_to(box, hb)
        if bbox_width(box) < 2 or bbox_height(box) < 2:
            return None
        coverage = float(mask[box[1] - hb[1]:box[3] - hb[1], box[0] - hb[0]:box[2] - hb[0]].mean())
        return HeadDetection(
            bbox=box,
            confidence=float(horse_conf) * self.conf_factor,
            method=self.name,
            keypoints={"poll_top": [[float(hb[0] + cx), float(hb[1] + top), 1.0]]},
        ) if coverage > 0 else None


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _build_bbox_top(device: str, **kw: object) -> HorseHeadDetector:
    return BBoxTopHeadDetector()


def _build_mask_top(device: str, **kw: object) -> HorseHeadDetector:
    weights = kw.get("seg_weights", "models/yolo11s-seg.pt")
    imgsz = int(kw.get("seg_imgsz", 416))  # type: ignore[arg-type]
    try:
        return MaskTopHeadDetector(weights=str(weights), device=device, imgsz=imgsz)
    except Exception as e:  # download / load failure
        logger.warning("Could not load segmentation model %s (%s); falling back to bbox_top", weights, e)
        return BBoxTopHeadDetector()


def _build_grounding_dino(device: str, **kw: object) -> HorseHeadDetector:
    """Grounding DINO head detector (model loaded lazily on first ``detect``).

    kwargs: ``cache_dir`` (HF hub cache; default ``<project>/models/hf/hub``),
    ``fallback_name`` ("mask_top" (default) | "bbox_top" | "none"), plus any
    kwargs of the fallback builder (``seg_weights``, ``seg_imgsz``).
    """
    from ..config import resolve_project_path
    from .grounded_head_detector import GroundingDinoHeadDetector

    cache_dir = resolve_project_path(str(kw.get("cache_dir") or "models/hf/hub"))
    fallback_name = str(kw.get("fallback_name") or "mask_top")
    fallback: Optional[HorseHeadDetector] = None
    if fallback_name.lower() != "none":
        if fallback_name == "grounding_dino" or fallback_name not in _REGISTRY:
            raise ValueError(f"Invalid head fallback {fallback_name!r}; use mask_top, bbox_top or none")
        fb_kw = {k: v for k, v in kw.items() if k not in ("cache_dir", "fallback_name")}
        fallback = _REGISTRY[fallback_name](device, **fb_kw)
    return GroundingDinoHeadDetector(device=device, cache_dir=cache_dir, fallback=fallback)


_REGISTRY: dict[str, Callable[..., HorseHeadDetector]] = {
    "bbox_top": _build_bbox_top,
    "mask_top": _build_mask_top,
    "grounding_dino": _build_grounding_dino,
}


def available_head_detectors() -> list[str]:
    return list(_REGISTRY)


def build_head_detector(name: str = "grounding_dino", device: str = "cpu", **kwargs: object) -> HorseHeadDetector:
    """Instantiate a head detector by name ("bbox_top", "mask_top", "grounding_dino")."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown head detector {name!r}; available: {available_head_detectors()}")
    return _REGISTRY[name](device, **kwargs)
