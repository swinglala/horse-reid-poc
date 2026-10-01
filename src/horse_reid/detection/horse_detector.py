"""Ultralytics YOLO horse + person detector with built-in multi-object tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from ..types import Detection, bbox_from_xyxy, clip_bbox

logger = logging.getLogger(__name__)

COCO_PERSON = 0
COCO_HORSE = 17
_VALID_TRACKERS = ("bytetrack.yaml", "botsort.yaml")


class YoloHorseDetector:
    """Detect and track horses (and persons, for occlusion reasoning).

    Wraps ``ultralytics.YOLO.track`` with ``persist=True`` so track IDs are
    stable across sequential calls. Call :meth:`reset` before processing a new
    video.
    """

    def __init__(
        self,
        weights: str | Path = "models/yolo11s.pt",
        device: str = "cpu",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.25,
        classes: Sequence[int] = (COCO_PERSON, COCO_HORSE),
        imgsz: int = 640,
    ) -> None:
        from ultralytics import YOLO

        if tracker not in _VALID_TRACKERS and not Path(tracker).exists():
            raise ValueError(f"tracker must be one of {_VALID_TRACKERS} or a yaml path, got {tracker!r}")
        self.weights = str(weights)
        self.device = device
        self.tracker = tracker
        self.conf = conf
        self.classes = list(classes)
        self.imgsz = imgsz
        self.model = YOLO(self.weights)
        self.names: dict[int, str] = dict(self.model.names)
        logger.info("Loaded detector %s on %s (tracker=%s, conf=%.2f)", self.weights, device, tracker, conf)

    def reset(self) -> None:
        """Drop tracker state (new video)."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for t in predictor.trackers:
                t.reset()
        # Force re-initialisation of trackers on next call.
        if predictor is not None and hasattr(predictor, "trackers"):
            del predictor.trackers

    def track(self, frame: np.ndarray, frame_idx: int = 0, time_s: float = 0.0) -> list[Detection]:
        """Run detection + tracking on one frame (must be called sequentially).

        Returns all detections of the configured classes. Detections the
        tracker has not (yet) confirmed have ``track_id=None``.
        """
        r = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=self.classes,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        h, w = frame.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if (boxes.is_track and boxes.id is not None) else None
        dets: list[Detection] = []
        for i in range(len(xyxy)):
            dets.append(
                Detection(
                    frame=frame_idx,
                    track_id=int(ids[i]) if ids is not None else None,
                    bbox=clip_bbox(bbox_from_xyxy(xyxy[i]), w, h),
                    confidence=float(confs[i]),
                    cls_name=self.names.get(int(clss[i]), str(int(clss[i]))),
                    time_s=time_s,
                )
            )
        return dets
