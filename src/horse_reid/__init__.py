"""horse_reid: horse re-identification proof of concept.

Phase 1: detect + track a horse in a video, localise its head, score every
frame's head crop for quality and select a temporally diverse set of good
head frames.
"""

import os as _os

# Must be set before torch is first imported: lets ops missing on MPS (e.g.
# torchvision::nms used by Ultralytics) fall back to CPU instead of raising.
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

__all__ = ["__version__"]
