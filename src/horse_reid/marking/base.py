"""Interface and result type for horse white-marking segmenters (Phase 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def coat_class(coat_L_median: float) -> str:
    """Coat brightness class from the median face L (0..100 scale): dark < 45 <= medium < 65 <= light."""
    if coat_L_median < 45:
        return "dark"
    if coat_L_median < 65:
        return "medium"
    return "light"


@dataclass
class MarkingResult:
    """Output of a :class:`HorseMarkingSegmenter` on one face crop.

    Attributes:
        mask: uint8 ``HxW`` binary mask (0/1) of white markings.
        prob: float32 ``HxW`` marking probability in ``[0, 1]``.
        face_mask: bool ``HxW`` face-region mask the segmenter worked inside.
        components: accepted marking components; dicts with ``area_px``,
            ``area_frac_of_face``, ``bbox`` (x1, y1, x2, y2), ``centroid``
            (x, y), ``mean_z``, ``accepted_by`` ("candidate" | "sam").
        excluded: rejected regions; same keys plus ``reason``
            ("strap_shape" | "high_chroma" | "too_small" | "sam_flood").
        method: name of the segmenter that produced the result.
        stats: ``marking_area_frac``, ``n_components``, ``coat_L_median``,
            ``coat_L_mad`` (+ segmenter-specific extras).
        debug: non-serialised arrays for visualisation, e.g.
            ``excluded_mask`` (uint8 HxW) and ``candidate_prob`` (float32 HxW).
    """

    mask: np.ndarray
    prob: np.ndarray
    face_mask: np.ndarray
    components: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    method: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (no arrays)."""
        return {
            "method": self.method,
            "stats": self.stats,
            "components": self.components,
            "excluded": self.excluded,
        }


class HorseMarkingSegmenter(ABC):
    """Segment white markings (star, stripe, blaze, snip, ...) on a horse face crop.

    Implementations must be drop-in replaceable: the heuristic
    ``adaptive_color`` / ``sam_refined`` segmenters and a trained model
    (``yolo_seg``) share this interface.
    """

    name: str = "base"

    @abstractmethod
    def predict(self, image: np.ndarray, face_mask: Optional[np.ndarray] = None,
                keypoints: Optional[dict] = None) -> MarkingResult:
        """
        Returns:
            mask: binary or probability mask
        """


def full_face_mask(image: np.ndarray, face_mask: Optional[np.ndarray]) -> np.ndarray:
    """Bool face mask with the image's ``HxW``; all-True when ``face_mask`` is None."""
    h, w = image.shape[:2]
    if face_mask is None:
        return np.ones((h, w), dtype=bool)
    fm = np.asarray(face_mask).astype(bool)
    if fm.shape != (h, w):
        raise ValueError(f"face_mask shape {fm.shape} does not match image {(h, w)}")
    return fm
