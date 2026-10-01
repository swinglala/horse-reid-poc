"""Name -> marking segmenter registry."""

from __future__ import annotations

from typing import Any, Callable

from .base import HorseMarkingSegmenter


def _adaptive(device: str, **kw: Any) -> HorseMarkingSegmenter:
    from .adaptive_color import AdaptiveColorParams, AdaptiveColorSegmenter

    params = kw.get("params")
    if isinstance(params, dict):
        params = AdaptiveColorParams(**params)
    return AdaptiveColorSegmenter(params)


def _sam_refined(device: str, **kw: Any) -> HorseMarkingSegmenter:
    from .sam_refined import SamRefinedSegmenter

    base = kw.pop("base", None) or _adaptive(device, **{k: v for k, v in kw.items() if k == "params"})
    kw.pop("params", None)
    sam_weights = kw.pop("sam_weights", None) or "models/mobile_sam.pt"
    return SamRefinedSegmenter(base=base, sam_weights=sam_weights, device=device, **kw)


def _yolo_seg(device: str, **kw: Any) -> HorseMarkingSegmenter:
    from .yolo_seg import YoloSegMarkingSegmenter

    weights = kw.pop("weights_path", None) or kw.pop("weights", None) or "models/marking_yolo_seg.pt"
    return YoloSegMarkingSegmenter(weights, device=device, **kw)


_REGISTRY: dict[str, Callable[..., HorseMarkingSegmenter]] = {
    "sam_refined": _sam_refined,
    "adaptive_color": _adaptive,
    "yolo_seg": _yolo_seg,
}


def available_marking_segmenters() -> list[str]:
    return list(_REGISTRY)


def build_marking_segmenter(name: str = "sam_refined", device: str = "cpu", **kw: Any) -> HorseMarkingSegmenter:
    """Instantiate a marking segmenter: "sam_refined" (default), "adaptive_color", "yolo_seg".

    kwargs: ``params`` (AdaptiveColorParams or dict) for adaptive_color /
    sam_refined; ``sam_weights``, ``max_components`` for sam_refined;
    ``weights_path``, ``class_ids`` for yolo_seg.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown marking segmenter {name!r}; available: {available_marking_segmenters()}")
    return _REGISTRY[name](device, **kw)
