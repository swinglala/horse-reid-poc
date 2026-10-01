"""Phase 3: canonical white-marking representation + identity reference.

* :class:`PlanarCanonicalMapper` - 2D similarity warp onto a frontal template (works on CPU today).
* :class:`FourDEquineMapper` - per-vertex mapping on the 4DEquine/VAREN surface (external artifacts).
* :func:`build_reference` - orchestration -> ``<output>/reference/horse_reference.json``.
"""

from .base import CanonicalMapper, CanonicalObservation, CanonicalReference, aggregate_weighted
from .fourdequine import FourDEquineMapper, FourDEquineNotAvailable
from .planar import CANONICAL_H, CANONICAL_LANDMARKS, CANONICAL_W, PlanarCanonicalMapper


def build_reference(*args, **kwargs):  # lazy: keeps package import light
    from .reference import build_reference as _br

    return _br(*args, **kwargs)


__all__ = [
    "CANONICAL_H", "CANONICAL_LANDMARKS", "CANONICAL_W", "CanonicalMapper", "CanonicalObservation",
    "CanonicalReference", "FourDEquineMapper", "FourDEquineNotAvailable", "PlanarCanonicalMapper",
    "aggregate_weighted", "build_reference",
]
