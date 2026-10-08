"""Coat-adaptive colour candidate generator for white markings.

BASELINE, not the final method: it is kept as the candidate generator of
:class:`~horse_reid.marking.sam_refined.SamRefinedSegmenter`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import math

import numpy as np

from .base import HorseMarkingSegmenter, MarkingResult, coat_class, full_face_mask

MAD_TO_SIGMA = 1.4826
log = logging.getLogger(__name__)


@dataclass
class AdaptiveColorParams:
    """Thresholds of :class:`AdaptiveColorSegmenter` (Lab units: L 0..100, a/b ~ -128..127)."""

    exclude_top_frac: float = 0.05     # brightest face pixels ignored for coat statistics
    min_mad: float = 4.0               # floor of the L MAD: z0 * 1.4826 * 4 ~ 15 L units minimum contrast
                                       # (avoids z blow-up on flat / over-exposed crops)
    z0: float = 2.5                    # brightness z-score at prob 0.5 (global illumination path)
    z0_local: float = 3.0              # same, used instead of z0 when local_illumination is on
    zs: float = 0.6                    # brightness sigmoid scale
    chroma_max: float = 22.0           # chroma at prob 0.5 / component high-chroma limit
    cs: float = 4.0                    # chroma sigmoid scale
    prob_thresh: float = 0.5
    open_ksize: int = 3
    close_ksize: int = 5
    min_area_px: int = 12
    min_area_frac: float = 0.0005      # of face area
    strap_ratio: float = 4.0           # min-area-rect long/short side
    strap_min_len_frac: float = 0.25   # long side relative to face-mask width
    local_illumination: bool = True    # detrend L by a large-kernel illumination field before z-scoring
    local_kernel_frac: float = 0.25    # illumination kernel size / face-mask width (min 15 px, odd)
    min_area_frac_local: float = 0.0008  # of face area, replaces min_area_frac when local illumination is on
    solidity_min: float = 0.55         # area / convex hull area ("low_solidity")
    strong_z: float = 6.5              # a component this far above the coat and ...
    strong_area_frac: float = 0.01     # ... covering this fraction of the face is a "strong" white region
                                       # (white hair; coat sheen and halter webbing stay below ~6 z)
    stripe_axis_max_deg: float = 30.0  # a strong elongated component whose long axis is within this angle of
                                       # the face axis (ear/eye midpoint -> nose) is a blaze/stripe, not a strap
    stripe_solidity_min: float = 0.35  # solidity floor for strong regions (blazes taper, bend, get cut by straps)
    edge_margin_px: int = 2            # component within this many px of the face-mask boundary ...
    edge_max_area_frac: float = 0.003  # ... and smaller than this face fraction -> "edge_fragment"
    eye_zone_frac: float = 0.08        # centroid within this x face width of an eye keypoint -> "eye_glint"
    ear_line_margin_frac: float = 0.02  # centroid above (lowest ear y + this x face height) -> "above_ears"
    ear_zone_frac: float = 0.10        # centroid within this x face width of an ear keypoint -> "above_ears"
    hull_margin_frac: float = 0.12     # dilation of the ear/eye/nose keypoint hull x face width -> "outside_landmark_hull"
    clip_L: float = 245.0              # 8-bit-scale L (0..255) counted as sensor-clipped
    clip_frac_max: float = 0.5         # >= this fraction of clipped pixels -> "blown_highlight"
    halo_ring_px: int = 3              # dilation radius of the halo ring
    halo_min_z: float = 1.0            # ring pixels must be brighter than coat by this z to count as halo
    halo_chroma_delta: float = 12.0    # mean halo chroma - component chroma above this -> "blown_highlight"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def lab_float(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(L, a, b)`` float32 planes (L in 0..100)."""
    lab = cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return lab[..., 0], lab[..., 1], lab[..., 2]


def coat_statistics(L: np.ndarray, face_mask: np.ndarray, exclude_top_frac: float = 0.05,
                    min_mad: float = 4.0) -> tuple[float, float]:
    """Robust coat brightness: median and MAD of L over the face mask after
    dropping the brightest ``exclude_top_frac`` (reduces marking influence)."""
    vals = L[face_mask]
    if vals.size == 0:
        vals = L.ravel()
    if vals.size > 20 and exclude_top_frac > 0:
        vals = vals[vals <= np.quantile(vals, 1.0 - exclude_top_frac)]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, max(mad, min_mad)


def coat_zmap(image_bgr: np.ndarray, face_mask: Optional[np.ndarray] = None,
              exclude_top_frac: float = 0.05, min_mad: float = 4.0) -> tuple[np.ndarray, float, float]:
    """Brightness z-score map relative to the coat: ``(L - med) / (1.4826 * MAD)``."""
    fm = full_face_mask(image_bgr, face_mask)
    L, _, _ = lab_float(image_bgr)
    med, mad = coat_statistics(L, fm, exclude_top_frac, min_mad)
    return (L - med) / (MAD_TO_SIGMA * mad), med, mad


def mask_width(mask: np.ndarray) -> int:
    """Horizontal extent (pixels) of a bool mask."""
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[-1] - cols[0] + 1) if cols.size else int(mask.shape[1])


def illumination_field(L: np.ndarray, face_mask: np.ndarray, kernel_frac: float = 0.25,
                       exclude_top_frac: float = 0.05) -> np.ndarray:
    """Smooth illumination of L inside the face mask (masked box blur, no background leakage).

    The brightest ``exclude_top_frac`` face pixels get zero weight so markings do
    not lift the field. Pixels outside the mask get the nearest-weight estimate
    (blur ratio) or the global median where no weight is in reach.
    """
    w = face_mask.astype(np.float32)
    vals = L[face_mask]
    if vals.size > 20 and exclude_top_frac > 0:
        w = w * (L <= np.quantile(vals, 1.0 - exclude_top_frac))
    k = max(15, int(kernel_frac * mask_width(face_mask)))
    k |= 1  # odd
    num = cv2.blur((L * w).astype(np.float32), (k, k))
    den = cv2.blur(w.astype(np.float32), (k, k))
    gmed = float(np.median(vals)) if vals.size else float(np.median(L))
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), gmed).astype(np.float32)


def solidity(region: np.ndarray) -> float:
    """Area / convex-hull area (pixel counts) of a bool region."""
    pts = cv2.findNonZero(region.astype(np.uint8))
    if pts is None:
        return 0.0
    x, y, bw, bh = cv2.boundingRect(pts)
    hull = cv2.convexHull(pts)
    canvas = np.zeros((bh, bw), np.uint8)
    cv2.fillConvexPoly(canvas, hull - np.array([[x, y]], hull.dtype), 1)
    return float(region.sum()) / max(float(canvas.sum()), 1.0)


def keypoint_xy(keypoints: Optional[dict], name: str) -> list[tuple[float, float]]:
    """``(x, y)`` of all keypoint entries whose key contains ``name``."""
    out: list[tuple[float, float]] = []
    for k, v in (keypoints or {}).items():
        if name not in str(k).lower() or v is None:
            continue
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr[None, :]
        if arr.ndim == 2 and arr.shape[1] >= 2:
            out.extend((float(r[0]), float(r[1])) for r in arr)
    return out


def eye_points(keypoints: Optional[dict]) -> list[tuple[float, float]]:
    """``(x, y)`` of all keypoint entries whose key contains "eye"."""
    return keypoint_xy(keypoints, "eye")


def landmark_hull_mask(keypoints: Optional[dict], shape: tuple[int, int], margin_px: float) -> Optional[np.ndarray]:
    """Filled convex hull of all ear/eye/nose keypoints, dilated by ``margin_px``
    (elliptical kernel). ``None`` when fewer than 3 distinct points are available."""
    pts = {(round(x, 1), round(y, 1)) for n in ("ear", "eye", "nose") for x, y in keypoint_xy(keypoints, n)}
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(np.array(sorted(pts), np.float32)).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    r = int(round(margin_px))
    if r > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m.astype(bool)


def mask_height(mask: np.ndarray) -> int:
    """Vertical extent (pixels) of a bool mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else int(mask.shape[0])


def blown_highlight_stats(region: np.ndarray, L: np.ndarray, chroma: np.ndarray, z: np.ndarray,
                          ring_px: int = 3, min_z: float = 1.0) -> tuple[float, float]:
    """``(clip_frac, halo_delta)`` of a component.

    HEURISTIC (TODO: trained model). ``clip_frac`` = fraction of pixels with
    8-bit-scale L >= ``clip_L`` is computed by the caller from ``L``; here
    ``halo_delta`` = mean chroma of the dilated ring pixels that are brighter
    than the coat (z > ``min_z``, i.e. a lit halo, not plain dark coat) minus the
    component's mean chroma. A bright core inside a strongly coloured sunlit
    halo is a specular on coat; a real white marking has a low-chroma edge.
    """
    k = np.ones((2 * ring_px + 1,) * 2, np.uint8)
    ring = cv2.dilate(region.astype(np.uint8), k).astype(bool) & ~region & (z > min_z)
    if int(ring.sum()) < 5:
        return 0.0, 0.0
    return 0.0, float(chroma[ring].mean() - chroma[region].mean())


def face_axis(keypoints: Optional[dict]) -> tuple[float, float]:
    """Unit vector of the face's long axis in image coordinates: from the mean
    ear/eye keypoint to the mean nose keypoint. Falls back to straight down
    (0, 1) - the head crops are roughly upright - when either end is missing."""
    top = keypoint_xy(keypoints, "ear") + keypoint_xy(keypoints, "eye")
    nose = keypoint_xy(keypoints, "nose")
    if not top or not nose:
        return 0.0, 1.0
    tx, ty = float(np.mean([x for x, _ in top])), float(np.mean([y for _, y in top]))
    nx, ny = float(np.mean([x for x, _ in nose])), float(np.mean([y for _, y in nose]))
    dx, dy = nx - tx, ny - ty
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else (0.0, 1.0)


def region_axis_angle(region: np.ndarray, axis: tuple[float, float]) -> float:
    """Angle in degrees (0..90) between the principal axis of a bool region
    (PCA of its pixel coordinates) and ``axis``."""
    ys, xs = np.nonzero(region)
    if xs.size < 3:
        return 90.0
    pts = np.stack([xs - xs.mean(), ys - ys.mean()], axis=1).astype(np.float64)
    cov = pts.T @ pts / pts.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, int(np.argmax(vals))]
    cos = abs(float(v[0] * axis[0] + v[1] * axis[1])) / max(float(np.hypot(*v)), 1e-9)
    return float(math.degrees(math.acos(min(1.0, cos))))


def region_info(region: np.ndarray, face_area: int, z: np.ndarray, chroma: Optional[np.ndarray] = None,
                **extra: Any) -> dict[str, Any]:
    """Descriptor dict of a bool region (see :class:`MarkingResult`)."""
    ys, xs = np.nonzero(region)
    area = int(xs.size)
    d: dict[str, Any] = {
        "area_px": area,
        "area_frac_of_face": round(area / max(face_area, 1), 6),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if area else [0, 0, 0, 0],
        "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)] if area else [0.0, 0.0],
        "mean_z": round(float(z[region].mean()), 3) if area else 0.0,
    }
    if chroma is not None and area:
        d["mean_chroma"] = round(float(chroma[region].mean()), 2)
    d.update(extra)
    return d


def rect_sides(region: np.ndarray) -> tuple[float, float]:
    """(long, short) side of the min-area rectangle of a bool region."""
    pts = cv2.findNonZero(region.astype(np.uint8))
    if pts is None:
        return 0.0, 0.0
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    # minAreaRect measures pixel centres; +1 gives the pixel extent.
    w, h = w + 1.0, h + 1.0
    return max(w, h), min(w, h)


def contour_shape(region: np.ndarray, eps: float = 2.0) -> tuple[float, float, float]:
    """Curvature-robust shape of the largest outer contour of a bool region.

    Returns ``(elongation, length, thickness)`` with ``elongation = P^2 / (4 A)``
    (circle ~3.1, square 4, k:1 rectangle ~ (k+1)^2 / k ... ~k for long
    strips), ``length ~ P / 2`` and ``thickness = 2 A / P``, computed on the
    hole-filled contour after ``approxPolyDP(eps)``. Unlike the min-area
    rectangle this also flags curved straps (e.g. a noseband seen frontally).
    """
    cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0, 0.0, 0.0
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps, True)
    per = float(cv2.arcLength(c, True))
    area = max(float(cv2.contourArea(c)), 1.0)
    if per <= 0:
        return 0.0, 0.0, 0.0
    return per * per / (4.0 * area), per / 2.0, 2.0 * area / per


class AdaptiveColorSegmenter(HorseMarkingSegmenter):
    """BASELINE coat-adaptive colour segmenter (NOT the final method; kept as
    the candidate generator for ``sam_refined``).

    # TODO(phase2-upgrade): replace by a model trained on AI-Hub white-marking polygons

    Steps (Lab colour space, statistics restricted to the face mask):

    1. coat statistics: median / MAD of L over the face mask, ignoring the
       brightest 5 %; ``z = (L - med) / (1.4826 * MAD)``;
       ``chroma = sqrt(a^2 + b^2)``.
    2. ``prob = sigmoid((z - z0)/zs) * sigmoid((chroma_max - chroma)/cs)``.
    3. Candidate regions = brightness term > 0.5 inside the face mask, open
       3x3 then close 5x5, connected components. (Gating on the brightness
       term alone - instead of ``prob`` - lets bright but coloured objects such
       as the green halter tag surface as components so their rejection is
       recorded as ``high_chroma``; achromatic pixels have brightness term ==
       prob up to the chroma factor.)
    4. Component filters: ``too_small``, ``high_chroma`` (mean chroma >
       ``chroma_max``), ``strap_shape`` (min-area-rect long/short > 4 and long
       side > 25 % of the face-mask width: halter / noseband straps). A
       *strong* elongated component (>= ``strong_area_frac`` of the face and
       mean z >= ``strong_z``) whose principal axis lies within
       ``stripe_axis_max_deg`` of the face axis (ear/eye midpoint -> nose) is a
       blaze / stripe, not a strap: straps cross the face, blazes run along it.
       ``low_solidity`` (area / convex hull < ``solidity_min``) uses the lower
       ``stripe_solidity_min`` for strong components (blazes taper and bend).
    5. Anatomy / exposure gates: ``above_ears`` (centroid above the lowest ear
       keypoint or near any ear keypoint; face markings lie below the ear
       bases) and ``blown_highlight`` (HEURISTIC, TODO: trained model: >= 50 %
       sensor-clipped pixels, or a bright core in a strongly coloured lit halo;
       strong components are exempt: large white hair clips in direct sun).
    """

    name = "adaptive_color"

    def __init__(self, params: Optional[AdaptiveColorParams] = None) -> None:
        self.params = params or AdaptiveColorParams()

    def predict(self, image: np.ndarray, face_mask: Optional[np.ndarray] = None,
                keypoints: Optional[dict] = None) -> MarkingResult:
        """
        Returns:
            mask: binary or probability mask
        """
        p = self.params
        fm = full_face_mask(image, face_mask)
        h, w = fm.shape
        face_area = int(fm.sum())
        L, a, b = lab_float(image)
        if p.local_illumination and face_area > 0:
            illum = illumination_field(L, fm, p.local_kernel_frac, p.exclude_top_frac)
            gmed, _ = coat_statistics(L, fm, p.exclude_top_frac, p.min_mad)
            L_eff = L - illum + gmed
            illum_mode = "local"
        else:
            illum = np.full(L.shape, float(np.median(L[fm])) if face_area else float(np.median(L)), np.float32)
            L_eff = L
            illum_mode = "global"
        med, mad = coat_statistics(L_eff, fm, p.exclude_top_frac, p.min_mad)
        z = (L_eff - med) / (MAD_TO_SIGMA * mad)
        chroma = np.sqrt(a * a + b * b)
        z0 = p.z0_local if illum_mode == "local" else p.z0
        bright = _sigmoid((z - z0) / p.zs)
        prob = (bright * _sigmoid((p.chroma_max - chroma) / p.cs)).astype(np.float32)
        prob[~fm] = 0.0

        cand = ((bright > p.prob_thresh) & fm).astype(np.uint8)
        # median filter instead of opening: drops isolated pixels without eroding small blobs
        cand = cv2.medianBlur(cand, p.open_ksize)
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((p.close_ksize, p.close_ksize), np.uint8))
        cand &= fm.astype(np.uint8)

        n, labels = cv2.connectedComponents(cand, connectivity=8)
        min_area = max(p.min_area_px, (p.min_area_frac_local if illum_mode == "local" else p.min_area_frac) * face_area)
        eyes = eye_points(keypoints)
        outside = ~fm
        edge_k = np.ones((2 * p.edge_margin_px + 1,) * 2, np.uint8)
        face_w = mask_width(fm)
        eye_r = p.eye_zone_frac * face_w
        ears = keypoint_xy(keypoints, "ear")
        ear_r = p.ear_zone_frac * face_w
        face_h = mask_height(fm)
        ear_line = (max(ey for _, ey in ears) + p.ear_line_margin_frac * face_h) if ears else None
        L255 = L * 2.55
        hull_mask = landmark_hull_mask(keypoints, (h, w), p.hull_margin_frac * face_w)
        axis = face_axis(keypoints)
        n_kp = sum(len(keypoint_xy(keypoints, n)) for n in ("ear", "eye", "nose"))
        mask = np.zeros((h, w), np.uint8)
        excl_mask = np.zeros((h, w), np.uint8)
        components: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        for lab_id in range(1, n):
            region = labels == lab_id
            area = int(region.sum())
            reason: Optional[str] = None
            long_side, short_side = rect_sides(region)
            mean_chroma = float(chroma[region].mean())
            elongated = short_side > 0 and long_side / short_side > p.strap_ratio and long_side > p.strap_min_len_frac * face_w
            strong = bool(area >= p.strong_area_frac * face_area and z[region].mean() >= p.strong_z)
            axis_deg = region_axis_angle(region, axis) if elongated else None
            # Halter / noseband straps cross the face; a blaze or stripe runs along it. Only a
            # strongly white strip qualifies: sunlit sheen along the nasal bone is axis-aligned too.
            stripe = bool(elongated and strong and axis_deg is not None and axis_deg <= p.stripe_axis_max_deg)
            if area < min_area:
                reason = "too_small"
            elif mean_chroma > p.chroma_max:
                reason = "high_chroma"
            elif elongated and not stripe:
                reason = "strap_shape"
            sol = solidity(region)
            if reason is None and sol < (p.stripe_solidity_min if strong else p.solidity_min):
                reason = "low_solidity"
            if reason is None and area < p.edge_max_area_frac * face_area and bool(
                    (cv2.dilate(region.astype(np.uint8), edge_k).astype(bool) & outside).any()):
                reason = "edge_fragment"
            if reason is None and eyes:
                ys_, xs_ = np.nonzero(region)
                cx_, cy_ = float(xs_.mean()), float(ys_.mean())
                if any((cx_ - ex) ** 2 + (cy_ - ey) ** 2 <= eye_r ** 2 for ex, ey in eyes):
                    reason = "eye_glint"
            if reason is None and ears:
                ys_, xs_ = np.nonzero(region)
                cx_, cy_ = float(xs_.mean()), float(ys_.mean())
                if cy_ < ear_line or any((cx_ - ex) ** 2 + (cy_ - ey) ** 2 <= ear_r ** 2 for ex, ey in ears):
                    reason = "above_ears"
            if reason is None and hull_mask is not None:
                ys_, xs_ = np.nonzero(region)
                if not hull_mask[min(int(round(ys_.mean())), h - 1), min(int(round(xs_.mean())), w - 1)]:
                    reason = "outside_landmark_hull"
            exposure: dict[str, Any] = {}
            if reason is None:
                clip = float((L255[region] >= p.clip_L).mean())
                _, halo = blown_highlight_stats(region, L, chroma, z, p.halo_ring_px, p.halo_min_z)
                exposure = {"clip_frac": round(clip, 3), "halo_delta": round(halo, 2)}
                # A large, strongly white, achromatic region clips in direct sun
                # because it IS white hair; speculars on dark coat are small.
                if not strong and (clip >= p.clip_frac_max or halo > p.halo_chroma_delta):
                    reason = "blown_highlight"
            info = region_info(region, face_area, z, chroma,
                               rect_long=round(long_side, 1), rect_short=round(short_side, 1),
                               solidity=round(sol, 3), strong=strong, **exposure,
                               **({"axis_deg": round(axis_deg, 1), "stripe": stripe} if axis_deg is not None else {}))
            if reason is None:
                info["accepted_by"] = "candidate"
                components.append(info)
                mask[region] = 1
            else:
                info["reason"] = reason
                excluded.append(info)
                excl_mask[region] = 1
        components.sort(key=lambda c: -c["area_px"])
        by_reason: dict[str, int] = {}
        for e in excluded:
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        log.info("adaptive_color: %d ear/eye/nose keypoints, hull gate %s, removed by gate: %s",
                 n_kp, "on" if hull_mask is not None else "off", by_reason)
        stats = {
            "n_excluded_by_reason": by_reason,
            "marking_area_frac": round(float(mask.sum()) / max(face_area, 1), 6),
            "n_components": len(components),
            "coat_L_median": round(med, 3),
            "coat_class": coat_class(med),
            "coat_L_mad": round(mad, 3),
            "face_area_px": face_area,
            "illumination": illum_mode,
        }
        return MarkingResult(mask=mask, prob=prob, face_mask=fm, components=components, excluded=excluded,
                             method=self.name, stats=stats,
                             debug={"excluded_mask": excl_mask, "candidate_prob": prob, "z": z.astype(np.float32),
                                    "illumination": illum,
                                    "params": asdict(p)})
