"""Trained-model marking segmenter (YOLO segmentation fine-tuned on AI-Hub data).

# TODO(phase2-upgrade): no weights exist yet - the AI-Hub dataset (dataSetSn=71707)
# has not been downloaded (login + approval, Korean nationals only). Train with
# ``scripts/convert_aihub.py`` + ``yolo segment train`` once access is granted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .adaptive_color import coat_zmap, region_info
from .base import HorseMarkingSegmenter, MarkingResult, coat_class, full_face_mask

logger = logging.getLogger(__name__)


class YoloSegMarkingSegmenter(HorseMarkingSegmenter):
    """Union of the instance masks of ``class_ids`` predicted by an Ultralytics seg model.

    ``prob`` is the per-pixel max of (mask x confidence) over the kept instances.
    """

    name = "yolo_seg"

    def __init__(self, weights_path: str | Path, class_ids: Optional[Sequence[int]] = (0,), device: str = "cpu",
                 conf: float = 0.25, imgsz: int = 640) -> None:
        from ..config import resolve_project_path

        path = resolve_project_path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Marking segmentation weights not found: {path}. No trained white-marking model exists yet: "
                "obtain the AI-Hub dataset (see docs/aihub_dataset.md), convert it with scripts/convert_aihub.py "
                "and fine-tune an Ultralytics *-seg model, then pass its weights here."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.class_ids = list(class_ids) if class_ids is not None else None
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image: np.ndarray, face_mask: Optional[np.ndarray] = None,
                keypoints: Optional[dict] = None) -> MarkingResult:
        """
        Returns:
            mask: binary or probability mask
        """
        fm = full_face_mask(image, face_mask)
        h, w = fm.shape
        r = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids,
                               device=self.device, verbose=False, retina_masks=True)[0]
        prob = np.zeros((h, w), np.float32)
        if r.masks is not None and r.boxes is not None:
            data = r.masks.data.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for m, c in zip(data, confs):
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                prob = np.maximum(prob, (m > 0.5).astype(np.float32) * float(c))
        prob[~fm] = 0.0
        mask = (prob > 0).astype(np.uint8)
        z, med, mad = coat_zmap(image, fm)
        face_area = int(fm.sum())
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        comps: list[dict[str, Any]] = [region_info(labels == i, face_area, z, accepted_by="model")
                                       for i in range(1, n)]
        stats = {"marking_area_frac": round(float(mask.sum()) / max(face_area, 1), 6), "n_components": n - 1,
                 "coat_L_median": round(med, 3), "coat_class": coat_class(med), "coat_L_mad": round(mad, 3), "face_area_px": face_area}
        return MarkingResult(mask=mask, prob=prob, face_mask=fm, components=comps, excluded=[],
                             method=self.name, stats=stats, debug={"candidate_prob": prob})
