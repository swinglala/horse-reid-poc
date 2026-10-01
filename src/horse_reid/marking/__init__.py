"""Phase 2: white-marking segmentation of selected horse-face frames."""

from .adaptive_color import AdaptiveColorParams, AdaptiveColorSegmenter
from .base import HorseMarkingSegmenter, MarkingResult
from .face_region import FaceRegion, FaceRegionExtractor
from .registry import available_marking_segmenters, build_marking_segmenter
from .sam_refined import SamRefinedSegmenter

__all__ = [
    "HorseMarkingSegmenter",
    "MarkingResult",
    "AdaptiveColorParams",
    "AdaptiveColorSegmenter",
    "SamRefinedSegmenter",
    "FaceRegion",
    "FaceRegionExtractor",
    "build_marking_segmenter",
    "available_marking_segmenters",
]
