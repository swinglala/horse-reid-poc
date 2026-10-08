"""Face-region extraction for Phase 2: crop + face mask that excludes background."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

from ..types import BBox, clip_bbox, expand_bbox
from .adaptive_color import landmark_hull_mask

logger = logging.getLogger(__name__)


@dataclass
class FaceRegion:
    """Result of :meth:`FaceRegionExtractor.extract`.

    Unpacks as ``(face_crop_bgr, face_mask_bool, crop_bbox)``. ``crop`` and
    ``face_mask`` are already upscaled by ``upscale``; ``crop_bbox`` is in
    original frame coordinates (``frame[y1:y2, x1:x2]`` before upscaling).
    """

    crop: np.ndarray
    face_mask: np.ndarray
    crop_bbox: BBox
    upscale: float = 1.0
    face_mask_source: str = "seg"   # "seg" | "seg+landmark_hull" | "landmark_hull" | "ellipse_fallback"
    hull_added_frac: float = 0.0    # fraction of the final face mask contributed by the landmark hull only
    coat_mask: Optional[np.ndarray] = None  # seg-derived part of face_mask when the hull extended it (coat reference)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.crop, self.face_mask, self.crop_bbox))


def ellipse_mask(h: int, w: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Bool ``h x w`` mask of the ellipse inscribed in ``box`` (x1, y1, x2, y2)."""
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2 - 1) / 2.0, (y1 + y2 - 1) / 2.0
    ax, ay = max(1.0, (x2 - x1) / 2.0), max(1.0, (y2 - y1) / 2.0)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))), 0, 0, 360, 1, -1)
    return m.astype(bool)


class FaceRegionExtractor:
    """Crop the head (head bbox + margin) and build a face mask.

    The face mask is the YOLO11-seg horse instance mask, united with the
    convex hull of the ear/eye/nose landmarks when ``keypoints`` are given
    (the seg mask can stop at a halter noseband and drop the muzzle; the
    landmarks say where the face really ends). The seg mask is computed with
    :meth:`MaskTopHeadDetector.horse_mask` - same code path as Phase 1's
    ``mask_top`` head heuristic) intersected with the crop and eroded by
    ``erode_frac`` x crop width to drop the coat/background boundary.
    If no horse mask is found, the inscribed ellipse of the head bbox is used
    (``face_mask_source="ellipse_fallback"``).

    # TODO(phase2-upgrade): the ellipse fallback and "horse mask inside head
    # box" are approximations of a true face mask; replace with a face/head
    # part segmenter (e.g. SegFormer head model trained on AI-Hub data).

    Crops narrower than ``min_width`` are upscaled by ``upscale`` (cubic for
    the image, nearest for the mask) because markings are small.

    ``seg_model_path=None`` disables segmentation (always ellipse fallback).
    The seg model is loaded lazily on the first ``extract`` call.
    """

    def __init__(self, seg_model_path: Optional[str | Path] = "models/yolo11s-seg.pt", device: str = "cpu",
                 imgsz: int = 416, upscale: float = 2.0, min_width: int = 300, erode_frac: float = 0.02) -> None:
        self.seg_model_path = seg_model_path
        self.device = device
        self.imgsz = imgsz
        self.upscale = upscale
        self.min_width = min_width
        self.erode_frac = erode_frac
        self._seg: Any = None
        self._seg_failed = False

    def _segmenter(self) -> Any:
        if self.seg_model_path is None or self._seg_failed:
            return None
        if self._seg is None:
            try:
                from ..config import resolve_project_path
                from ..face.head_detector import MaskTopHeadDetector

                self._seg = MaskTopHeadDetector(weights=str(resolve_project_path(self.seg_model_path)),
                                                device=self.device, imgsz=self.imgsz)
            except Exception as e:
                logger.warning("Could not load seg model %s (%s); face masks fall back to ellipses",
                               self.seg_model_path, e)
                self._seg_failed = True
                return None
        return self._seg

    def horse_mask_in_crop(self, frame: np.ndarray, horse_bbox: Optional[BBox], cb: BBox) -> Optional[np.ndarray]:
        """Horse instance mask restricted to crop box ``cb`` (bool, crop-sized) or None."""
        seg = self._segmenter()
        if seg is None or horse_bbox is None:
            return None
        h_img, w_img = frame.shape[:2]
        hb = clip_bbox(tuple(int(v) for v in horse_bbox), w_img, h_img)  # type: ignore[arg-type]
        # The tracker's horse box can be narrower than the head (e.g. a handler
        # standing next to the face truncates it). The seg mask is returned in
        # horse-box coordinates, so widen the box to cover the whole head crop;
        # otherwise the part of the face outside the horse box is cut off.
        hb = clip_bbox((min(hb[0], cb[0]), min(hb[1], cb[1]), max(hb[2], cb[2]), max(hb[3], cb[3])), w_img, h_img)
        hm = seg.horse_mask(frame, hb)
        if hm is None:
            return None
        full = np.zeros((h_img, w_img), bool)
        full[hb[1]:hb[1] + hm.shape[0], hb[0]:hb[0] + hm.shape[1]] = hm
        m = full[cb[1]:cb[3], cb[0]:cb[2]]
        return m if m.any() else None

    def extract(self, frame: np.ndarray, horse_bbox: Optional[BBox], head_bbox: BBox,
                margin: float = 0.15, keypoints: Optional[dict] = None,
                hull_margin_frac: float = 0.06, hull_parts: tuple[str, ...] = ("eye", "nose")) -> FaceRegion:
        """Return ``(face_crop_bgr, face_mask_bool, crop_bbox)`` (a :class:`FaceRegion`).

        ``keypoints`` are frame-coordinate part landmarks ``{name: [[x, y, ...], ...]}``;
        the convex hull of the parts named in ``hull_parts`` (default eyes + nose:
        the central face, so the hull stays on the horse; ears would pull in the
        background beside the head), dilated by ``hull_margin_frac`` x crop width,
        is added to the face mask.
        """
        h_img, w_img = frame.shape[:2]
        head = clip_bbox(tuple(int(v) for v in head_bbox), w_img, h_img)  # type: ignore[arg-type]
        cb = expand_bbox(head, margin, w_img, h_img)
        crop_img = frame[cb[1]:cb[3], cb[0]:cb[2]].copy()
        ch, cw = crop_img.shape[:2]
        if ch == 0 or cw == 0:
            raise ValueError(f"empty crop for head bbox {head_bbox}")

        mask = self.horse_mask_in_crop(frame, horse_bbox, cb)
        source = "seg"
        if mask is not None:
            k = max(1, int(round(self.erode_frac * cw)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            # Border-replicate so that the crop edge itself is not treated as background.
            eroded = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_REPLICATE).astype(bool)
            mask = eroded if eroded.sum() > 0.05 * mask.sum() else mask
        hull_added = 0.0
        hull = None
        coat: Optional[np.ndarray] = None
        if keypoints:
            kc: dict[str, list[list[float]]] = {}
            for k, v in keypoints.items():
                if not any(part in str(k).lower() for part in hull_parts):
                    continue
                try:
                    kc[k] = [[float(pt[0]) - cb[0], float(pt[1]) - cb[1]] for pt in v]
                except (TypeError, IndexError, ValueError):
                    continue
            hull = landmark_hull_mask(kc, (ch, cw), hull_margin_frac * cw)
        if hull is not None:
            if mask is None or not mask.any():
                mask, source, hull_added = hull, "landmark_hull", 1.0
            else:
                added = hull & ~mask
                if added.any():
                    coat = mask
                    mask = mask | hull
                    source = "seg+landmark_hull"
                    hull_added = float(added.sum()) / float(mask.sum())
        if mask is None or not mask.any():
            # TODO(phase2-upgrade): ellipse fallback = no real face segmentation.
            source = "ellipse_fallback"
            hx = (head[0] - cb[0], head[1] - cb[1], head[2] - cb[0], head[3] - cb[1])
            mask = ellipse_mask(ch, cw, hx)

        scale = 1.0
        if cw < self.min_width and self.upscale and self.upscale > 1:
            scale = float(self.upscale)
            nw, nh = int(round(cw * scale)), int(round(ch * scale))
            crop_img = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
            if coat is not None:
                coat = cv2.resize(coat.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST).astype(bool)
        return FaceRegion(crop=crop_img, face_mask=mask, crop_bbox=cb, upscale=scale, face_mask_source=source,
                          hull_added_frac=round(hull_added, 4), coat_mask=coat)



__all__ = ["FaceRegion", "FaceRegionExtractor", "ellipse_mask"]
