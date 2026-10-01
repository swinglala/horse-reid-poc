"""Interface and data types for canonical marking mappers (Phase 3).

A *canonical mapper* takes the per-frame 2D white-marking segmentation of a
horse face (Phase 2) and maps it into ONE shared reference frame, so that
observations from many video frames / view angles can be aggregated into a
single identity reference.

Principle: the reference is built ONLY by aggregating real observations. No
generated / hallucinated / "beautified" face is ever produced; pixels (or
vertices) that were never observed stay unobserved (coverage 0).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class CanonicalObservation:
    """One frame's marking evidence expressed in the canonical frame.

    Attributes:
        frame: video frame index.
        yaw: signed head yaw in degrees (negative = horse faces image-left).
        view: yaw bin ("frontal" | "three_quarter" | "profile").
        prob: float32 ``HxW`` marking probability in canonical coordinates
            (for vertex-based mappers: ``1xN`` per-vertex values).
        weight: float32 array, same shape as ``prob``; 0 where unobserved.
        quality: scalar frame quality used in the weight.
        mask: optional float32 ``HxW`` warped BINARY accepted-component mask
            (0/1) - the pixels the Phase-2 segmenter actually accepted as
            marking in this frame. Drives ``support`` / ``n_support``.
        meta: free-form diagnostics (transform, residual, landmarks used, ...).
    """

    frame: int
    yaw: float
    view: str
    prob: np.ndarray
    weight: np.ndarray
    quality: float
    mask: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalReference:
    """Aggregated canonical marking reference.

    Attributes:
        prob: float32 ``HxW`` weighted mean marking probability.
        coverage: float32 ``HxW`` in ``[0, 1]`` = sum of weights / max sum.
        mask: uint8 ``HxW`` (0/1) final canonical marking mask.
        consistency: float32 ``HxW`` = 1 - weighted std of the per-frame prob
            (0 where unobserved).
        view_count: number of observations that contributed (any weight > 0).
        views: count of contributing observations per yaw bin.
        frames: contributing frame indices.
        mapper: name of the mapper that produced it.
        weight_sum: float32 ``HxW`` raw sum of weights.
        support: float32 ``HxW`` = ``sum(w * mask) / sum(w)``: weighted fraction
            of observations whose accepted mask covers the pixel (0 where
            unobserved). ``None`` when the mapper provides no binary masks.
        n_support: int32 ``HxW`` = number of observations whose warped mask is
            1 at the pixel (counted only where that observation has weight > 0).
        support_stack: optional uint8 ``NxHxW`` per-observation (mask & weight>0),
            aligned with ``frames`` (used to list the frames supporting a peak).
    """

    prob: np.ndarray
    coverage: np.ndarray
    mask: np.ndarray
    consistency: np.ndarray
    view_count: int
    views: dict[str, int]
    frames: list[int]
    mapper: str
    weight_sum: Optional[np.ndarray] = None
    support: Optional[np.ndarray] = None
    n_support: Optional[np.ndarray] = None
    support_stack: Optional[np.ndarray] = None


def aggregate_weighted(prob_stack: Sequence[np.ndarray], weight_stack: Sequence[np.ndarray],
                       min_coverage: float = 0.15, prob_threshold: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared weighted aggregation used by all mappers.

    Returns ``(prob_ref, coverage, consistency, mask, weight_sum)`` where
    ``prob_ref = sum(w p) / sum(w)`` (0 where ``sum(w) == 0``),
    ``coverage = sum(w) / max(sum(w))``, ``consistency = 1 - weighted std(p)``
    (0 where unobserved) and ``mask = (prob_ref > prob_threshold) &
    (coverage > min_coverage)``. NaN probabilities are treated as weight 0.
    """
    if len(prob_stack) == 0:
        raise ValueError("no observations to aggregate")
    shape = np.asarray(prob_stack[0]).shape
    sw = np.zeros(shape, np.float64)
    swp = np.zeros(shape, np.float64)
    swp2 = np.zeros(shape, np.float64)
    for p, w in zip(prob_stack, weight_stack):
        p = np.asarray(p, np.float64)
        w = np.asarray(w, np.float64)
        if p.shape != shape or w.shape != shape:
            raise ValueError(f"observation shape mismatch: {p.shape}/{w.shape} vs {shape}")
        bad = ~np.isfinite(p) | ~np.isfinite(w) | (w < 0)
        w = np.where(bad, 0.0, w)
        p = np.where(bad, 0.0, p)
        sw += w
        swp += w * p
        swp2 += w * p * p
    obs = sw > 0
    prob = np.zeros(shape, np.float64)
    prob[obs] = swp[obs] / sw[obs]
    var = np.zeros(shape, np.float64)
    var[obs] = np.maximum(swp2[obs] / sw[obs] - prob[obs] ** 2, 0.0)
    consistency = np.where(obs, 1.0 - np.sqrt(var), 0.0)
    mx = float(sw.max()) if sw.size else 0.0
    coverage = sw / mx if mx > 0 else np.zeros(shape, np.float64)
    mask = ((prob > prob_threshold) & (coverage > min_coverage)).astype(np.uint8)
    return (prob.astype(np.float32), coverage.astype(np.float32), consistency.astype(np.float32),
            mask, sw.astype(np.float32))


def aggregate_support(mask_stack: Sequence[Optional[np.ndarray]], weight_stack: Sequence[np.ndarray]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binary-mask support of the aggregated observations.

    Returns ``(support, n_support, stack)``:

    * ``support = sum(w * m) / sum(w)`` (0 where ``sum(w) == 0``) - the weighted
      fraction of observations whose ACCEPTED mask covers the pixel;
    * ``n_support`` (int32) = number of observations with ``m == 1`` and
      ``w > 0`` at the pixel;
    * ``stack`` (uint8 ``NxHxW``) = the per-observation ``(m == 1) & (w > 0)``.

    A missing mask (``None``) counts as "no marking accepted" (all zeros); masks
    given as 0/255 are binarised with ``> 0.5`` after scaling.
    """
    if len(weight_stack) == 0:
        raise ValueError("no observations to aggregate")
    shape = np.asarray(weight_stack[0]).shape
    sw = np.zeros(shape, np.float64)
    swm = np.zeros(shape, np.float64)
    stack = np.zeros((len(weight_stack),) + shape, np.uint8)
    for i, (m, w) in enumerate(zip(mask_stack, weight_stack)):
        w = np.asarray(w, np.float64)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if m is None:
            mb = np.zeros(shape, bool)
        else:
            m = np.asarray(m, np.float64)
            if m.shape != shape:
                raise ValueError(f"mask shape mismatch: {m.shape} vs {shape}")
            if np.nanmax(m) > 1.0 + 1e-6:
                m = m / 255.0
            mb = np.nan_to_num(m) > 0.5
        sw += w
        swm += w * mb
        stack[i] = mb & (w > 0)
    support = np.zeros(shape, np.float64)
    obs = sw > 0
    support[obs] = swm[obs] / sw[obs]
    return support.astype(np.float32), stack.sum(0, dtype=np.int32), stack


def find_peaks(prob: np.ndarray, coverage: np.ndarray, support: np.ndarray, min_support: float = 0.02,
               min_distance_px: int = 12, max_peaks: int = 8, peak_sigma: float = 4.0,
               n_support: Optional[np.ndarray] = None, support_stack: Optional[np.ndarray] = None,
               frames: Optional[Sequence[int]] = None) -> list[dict[str, Any]]:
    """Candidate marking locations backed by ACCEPTED masks.

    ``support_s = gaussian_filter(support, peak_sigma)`` (``peak_sigma`` absorbs
    the ~10 px landmark-alignment error between frames); peaks are local maxima
    of ``support_s`` (:func:`skimage.feature.peak_local_max`, ``min_distance =
    min_distance_px``, ``threshold_abs = min_support``), ranked by ``support_s``,
    at most ``max_peaks``. A peak is kept only if at least one observation's
    warped mask is 1 within radius ``2 * peak_sigma``; ``frames`` = every such
    observation and ``n_support = len(frames)``. ``prob`` / ``coverage`` /
    ``support`` at the peak pixel are reported attributes only (not used for
    ranking). Without ``support_stack``, the radius check uses ``support > 0``
    and ``n_support`` = max of the ``n_support`` map in the disk.

    A peak is a CANDIDATE location, not a confirmed marking: the final mask
    still requires ``prob > 0.5``.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max

    sup = np.nan_to_num(np.asarray(support, np.float32))
    if sup.ndim != 2 or sup.size == 0 or float(sup.max()) <= 0:
        return []
    p = np.nan_to_num(np.asarray(prob, np.float32))
    c = np.nan_to_num(np.asarray(coverage, np.float32))
    sup_s = gaussian_filter(sup, sigma=float(peak_sigma), mode="constant") if peak_sigma > 0 else sup
    if float(sup_s.max()) < min_support:
        return []
    coords = peak_local_max(sup_s, min_distance=max(1, int(min_distance_px)), threshold_abs=float(min_support),
                            exclude_border=False)
    H, W = sup.shape
    r = 2.0 * float(peak_sigma)
    ri = int(np.ceil(r))
    cand = sorted(((float(sup_s[y, x]), int(y), int(x)) for y, x in coords), reverse=True)
    peaks: list[dict[str, Any]] = []
    for ss, y, x in cand:
        y0, y1, x0, x1 = max(0, y - ri), min(H, y + ri + 1), max(0, x - ri), min(W, x + ri + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        if support_stack is not None:
            hit = (support_stack[:, y0:y1, x0:x1] > 0) & disk[None]
            idx = np.nonzero(hit.reshape(hit.shape[0], -1).any(1))[0]
            sup_frames = [int(frames[i]) for i in idx] if frames is not None else [int(i) for i in idx]
            ns = len(idx)
        else:
            sup_frames = []
            ns = int(n_support[y0:y1, x0:x1][disk].max()) if n_support is not None else \
                int((sup[y0:y1, x0:x1][disk] > 0).any())
        if ns < 1:
            continue  # not backed by any accepted mask -> not a peak
        peaks.append({"x": x, "y": y, "support_s": round(ss, 4), "support": round(float(sup[y, x]), 4),
                      "prob": round(float(p[y, x]), 4), "coverage": round(float(c[y, x]), 4),
                      "n_support": ns, "frames": sup_frames})
        if len(peaks) >= max_peaks:
            break
    for i, d in enumerate(peaks):
        d["id"] = f"p{i}"
    return peaks


class CanonicalMapper(ABC):
    """Maps per-frame 2D marking masks into a canonical representation."""

    name: str = "base"

    @abstractmethod
    def map_frame(self, mask: np.ndarray, prob: Optional[np.ndarray],
                  keypoints_px: Optional[dict[str, list[list[float]]]], yaw: float, quality: float,
                  frame: int = -1, head_bbox_px: Optional[Sequence[float]] = None,
                  ) -> Optional[CanonicalObservation]:
        """Map one frame's marking mask / probability into the canonical frame.

        Args:
            mask: uint8 ``HxW`` binary mask (0/255 or 0/1) in mask-pixel coords.
            prob: uint8 (0..255) or float (0..1) ``HxW`` probability; ``None``
                -> the binary mask is used as probability.
            keypoints_px: ``{"eye"|"nose"|"ear": [[x, y, score], ...]}`` in
                mask-pixel coordinates (may be ``None`` / empty).
            yaw: signed yaw in degrees (negative = horse faces image-left).
            quality: frame quality score in ``[0, 1]``.
            frame: frame index (bookkeeping only).
            head_bbox_px: head box ``(x1, y1, x2, y2)`` in mask-pixel coords
                (used by fallbacks).

        Returns:
            The observation, or ``None`` when the frame cannot be mapped.
        """

    @abstractmethod
    def aggregate(self, observations: Sequence[CanonicalObservation]) -> CanonicalReference:
        """Fuse observations into one :class:`CanonicalReference`."""
