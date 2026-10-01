"""Object detection / tracking wrappers."""

from .horse_detector import COCO_HORSE, COCO_PERSON, YoloHorseDetector

__all__ = ["YoloHorseDetector", "COCO_HORSE", "COCO_PERSON"]
