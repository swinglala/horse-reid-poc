"""Core data types shared across pipeline stages.

Bounding boxes are always ``(x1, y1, x2, y2)`` integer pixel coordinates in
the (rotated, upright) frame, with ``x2``/``y2`` exclusive-ish (we treat them
as the slicing end, i.e. ``frame[y1:y2, x1:x2]``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

BBox = tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# BBox helpers
# --------------------------------------------------------------------------- #
def bbox_from_xyxy(xyxy: Sequence[float]) -> BBox:
    """Round a float xyxy sequence to an int bbox."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy[:4])
    return (x1, y1, x2, y2)


def bbox_width(b: BBox) -> int:
    return max(0, b[2] - b[0])


def bbox_height(b: BBox) -> int:
    return max(0, b[3] - b[1])


def bbox_area(b: BBox) -> int:
    return bbox_width(b) * bbox_height(b)


def bbox_center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_intersection(a: BBox, b: BBox) -> int:
    """Intersection area of two boxes (pixels)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def bbox_iou(a: BBox, b: BBox) -> float:
    inter = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def clip_bbox(b: BBox, width: int, height: int) -> BBox:
    """Clip a box to image bounds ``[0, width] x [0, height]``."""
    x1 = min(max(b[0], 0), width)
    y1 = min(max(b[1], 0), height)
    x2 = min(max(b[2], 0), width)
    y2 = min(max(b[3], 0), height)
    return (x1, y1, max(x1, x2), max(y1, y2))


def clip_bbox_to(b: BBox, container: BBox) -> BBox:
    """Clip a box to lie inside another box."""
    x1 = min(max(b[0], container[0]), container[2])
    y1 = min(max(b[1], container[1]), container[3])
    x2 = min(max(b[2], container[0]), container[2])
    y2 = min(max(b[3], container[1]), container[3])
    return (x1, y1, max(x1, x2), max(y1, y2))


def expand_bbox(b: BBox, margin: float, width: int, height: int) -> BBox:
    """Expand a box by ``margin`` (fraction of its size) on every side, clipped to the image."""
    mw = int(round(bbox_width(b) * margin))
    mh = int(round(bbox_height(b) * margin))
    return clip_bbox((b[0] - mw, b[1] - mh, b[2] + mw, b[3] + mh), width, height)


def crop(frame: np.ndarray, b: BBox) -> np.ndarray:
    """Return ``frame[y1:y2, x1:x2]`` after clipping (may be empty)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox(b, w, h)
    return frame[y1:y2, x1:x2]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One (optionally tracked) object detection in one frame."""

    frame: int
    track_id: Optional[int]
    bbox: BBox
    confidence: float
    cls_name: str
    time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "time_s": round(self.time_s, 4),
            "track_id": self.track_id,
            "cls": self.cls_name,
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class HeadDetection:
    """A horse head localisation inside a frame."""

    bbox: BBox
    confidence: float
    method: str
    # part name -> list of ``[x, y, score]`` in frame coordinates
    keypoints: Optional[dict[str, list[list[float]]]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [int(v) for v in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "keypoints": round_keypoints(self.keypoints),
        }


def round_keypoints(kps: Optional[dict[str, Any]], xy_digits: int = 1,
                    score_digits: int = 4) -> Optional[dict[str, list[list[float]]]]:
    """JSON-friendly copy of a keypoint dict (``[x, y, score]`` lists, rounded)."""
    if kps is None:
        return None
    out: dict[str, list[list[float]]] = {}
    for part, pts in kps.items():
        rows = []
        for p in pts:
            p = [float(v) for v in p]
            rows.append([round(p[0], xy_digits), round(p[1], xy_digits)]
                        + [round(v, score_digits) for v in p[2:]])
        out[str(part)] = rows
    return out


@dataclass
class FrameScore:
    """Quality assessment of the primary horse's head in one frame."""

    frame: int
    time_s: float
    track_id: Optional[int]
    horse_bbox: BBox
    head_bbox: BBox
    head_conf: float
    head_method: str
    size: float
    blur_var: float
    blur: float
    exposure: float
    occlusion: float
    yaw: float
    yaw_conf: float
    view: float
    det_conf: float
    score: float
    visibility: float = 0.0
    gated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def abs_yaw(self) -> float:
        return abs(self.yaw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameScore":
        """Inverse of :meth:`to_dict` (unknown keys are ignored)."""
        names = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in names}
        kw["horse_bbox"] = tuple(int(v) for v in d["horse_bbox"])
        kw["head_bbox"] = tuple(int(v) for v in d["head_bbox"])
        kw["extra"] = dict(d.get("extra") or {})
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["horse_bbox"] = [int(v) for v in self.horse_bbox]
        d["head_bbox"] = [int(v) for v in self.head_bbox]
        for k, v in list(d.items()):
            if isinstance(v, (float, np.floating)):
                d[k] = round(float(v), 4)
        if isinstance(d["extra"].get("keypoints"), dict):
            d["extra"]["keypoints"] = round_keypoints(d["extra"]["keypoints"])
        return d
