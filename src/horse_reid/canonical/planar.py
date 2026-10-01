"""Planar (2D similarity) canonical mapper - works today on CPU.

The canonical frame is a fixed 256 x 320 px *frontal* horse-face template
(:data:`CANONICAL_LANDMARKS`). It is a 2D STAND-IN for the 3D head surface:
each frame's marking map is warped onto it with a similarity transform
estimated from coarse eye / nose / ear keypoints. This is exact only for a
planar face seen frontally; for 3/4 and profile views the far side of the face
is foreshortened / hidden, which is why such views are down-weighted (or zeroed)
on the far half via a visibility profile instead of being "unwrapped". A proper
surface model (see :mod:`horse_reid.canonical.fourdequine`) replaces this.

Left/right convention (IMPORTANT)
---------------------------------
Template landmark names are IMAGE sides of the canonical frontal view, i.e. as
seen by a camera looking straight at the horse's face (no mirroring):

* ``left_eye``  (80, 120) = canonical image-left  = the horse's anatomical RIGHT eye
* ``right_eye`` (176, 120) = canonical image-right = the horse's anatomical LEFT eye

Yaw is signed, negative = the horse faces image-left (``quality.metrics``).
A horse whose nose points to image-left shows the camera its anatomical LEFT
side (picture standing at the horse's left shoulder: its nose points to your
left). Hence:

* yaw < 0: near (visible) side = horse's left = canonical image-RIGHT half
  (x >= 128); a single visible eye is ``right_eye``; far half = x < 128.
* yaw > 0: near side = horse's right = canonical image-LEFT half (x < 128); a
  single visible eye is ``left_eye``; far half = x >= 128.

Equivalently, in canonical coordinates the far (hidden) half lies on the side
the nose turns towards. Two detected eyes are assigned by image x order (a
yaw rotation below 90 deg never swaps their image order). A single eye with
``|yaw| <= 60`` is ambiguous: both assignments are fitted and the one with
the smaller residual wins; when the residual cannot discriminate (only 2
points -> exact fit) the prior decides: eye right of the nose in the image ->
``right_eye`` (in a left-facing profile the eye sits image-right of the nose,
consistent with the yaw rule above), else the yaw sign.

Because the Phase-1 yaw SIGN is heuristic, when exactly one eye ends up
matched its template side (not the yaw sign) selects the near half for the
visibility profile (:meth:`PlanarCanonicalMapper.near_side`); ``|yaw|`` still
comes from Phase 1.

Robustness: two eye detections closer than 0.25 x the eye-nose distance are
one eye (the detector often fires twice); a fit with RMS residual > 20
canonical px is retried leaving one landmark out (5 px penalty per dropped
landmark); residuals between 20 and 50 px scale the weight linearly to 0 and
fits >= 50 px are skipped.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ..quality.metrics import view_score, yaw_bin
from .base import CanonicalMapper, CanonicalObservation, CanonicalReference, aggregate_support, aggregate_weighted

logger = logging.getLogger(__name__)

CANONICAL_W = 256
CANONICAL_H = 320
CENTER_X = CANONICAL_W / 2.0

#: Fixed landmark targets of the canonical frontal template (x, y) in px.
#: Names are image sides of the frontal view (see module docstring).
CANONICAL_LANDMARKS: dict[str, tuple[float, float]] = {
    "left_eye": (80.0, 120.0),
    "right_eye": (176.0, 120.0),
    "nose": (128.0, 290.0),
    "left_ear_base": (72.0, 30.0),
    "right_ear_base": (184.0, 30.0),
}

#: Schematic frontal horse-face outline (visualisation only).
TEMPLATE_OUTLINE: list[tuple[int, int]] = [
    (96, 40), (74, 2), (58, 44), (42, 96), (44, 150), (66, 220), (80, 270), (92, 312),
    (164, 312), (176, 270), (190, 220), (212, 150), (214, 96), (198, 44), (182, 2), (160, 40),
]

FALLBACK_WEIGHT = 0.3       # weight factor for 1-landmark (bbox-scale) mapping
MIN_KP_SCORE = 0.25
PRIOR_PENALTY_PX = 2.0      # residual-equivalent penalty for disagreeing with the yaw prior
ROT_PENALTY_START_DEG = 45.0
ROT_PENALTY_PER_DEG = 0.1
SCALE_RANGE = (0.02, 50.0)
GOOD_RESIDUAL_PX = 20.0     # canonical px: fits up to this keep full weight
REJECT_RESIDUAL_PX = 50.0   # fits at/above this are dropped (linear weight in between)
DROP_PENALTY_PX = 5.0       # cost of discarding one landmark in the leave-one-out refit
EYE_DUP_FRAC = 0.25         # two eyes closer than this x mean eye-nose distance = one eye


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares similarity (Umeyama, no reflection) mapping ``src -> dst``.

    Returns ``(M 2x3, rms_residual_in_dst_px)``. Needs >= 2 distinct points.
    """
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    if len(src) < 2 or len(src) != len(dst):
        raise ValueError("need >= 2 point pairs")
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    var = float((s0 ** 2).sum()) / len(src)
    if var < 1e-9:
        raise ValueError("degenerate source points")
    cov = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1.0
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D)) / var
    t = md - scale * R @ ms
    M = np.hstack([scale * R, t[:, None]])
    pred = src @ M[:, :2].T + M[:, 2]
    rms = float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))
    return M, rms


def transform_params(M: np.ndarray) -> tuple[float, float]:
    """``(scale, rotation_deg)`` of a 2x3 similarity matrix."""
    a, b = float(M[0, 0]), float(M[1, 0])
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)))


def visibility_profile(yaw: float, width: int = CANONICAL_W, height: int = CANONICAL_H,
                       band_px: float = 24.0) -> np.ndarray:
    """Per-pixel visibility weight in canonical coordinates.

    * ``|yaw| <= 30``: 1 everywhere;
    * ``30 < |yaw| <= 60``: far half 0.4, near half 1;
    * ``|yaw| > 60``: far half 0, near half 1, and a ``band_px`` wide band
      centred on the midline 0.5.

    Far half: ``x < 128`` for yaw < 0, ``x >= 128`` for yaw > 0 (module docstring).
    """
    a = abs(float(yaw))
    prof = np.ones((height, width), np.float32)
    if a <= 30.0:
        return prof
    xs = np.arange(width, dtype=np.float32) + 0.5
    cx = width / 2.0
    far_cols = xs < cx if yaw < 0 else xs >= cx
    if a <= 60.0:
        prof[:, far_cols] = 0.4
        return prof
    prof[:, far_cols] = 0.0
    band = np.abs(xs - cx) <= band_px / 2.0
    prof[:, band] = 0.5
    return prof


def _parts(kps: Optional[dict], part: str, min_score: float) -> list[list[float]]:
    if not kps:
        return []
    pts = []
    for p in kps.get(part) or []:
        if p is None or len(p) < 2:
            continue
        q = [float(v) for v in p]
        if len(q) < 3:
            q.append(1.0)
        if q[2] >= min_score and np.isfinite(q[0]) and np.isfinite(q[1]):
            pts.append(q)
    return sorted(pts, key=lambda q: -q[2])


def _as_prob(mask: np.ndarray, prob: Optional[np.ndarray]) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if prob is None:
        return (m > 0).astype(np.float32)
    p = np.asarray(prob)
    if p.ndim == 3:
        p = p[..., 0]
    if p.dtype == np.uint8 or (p.size and float(np.nanmax(p)) > 1.0 + 1e-6):
        p = p.astype(np.float32) / 255.0
    p = np.clip(p.astype(np.float32), 0.0, 1.0)
    if p.shape != m.shape[:2]:
        p = cv2.resize(p, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
    return p


# --------------------------------------------------------------------------- #
# Mapper
# --------------------------------------------------------------------------- #
class PlanarCanonicalMapper(CanonicalMapper):
    """Similarity warp of each frame onto a 2D frontal template + weighted fusion."""

    name = "planar"

    def __init__(self, min_coverage: float = 0.15, prob_threshold: float = 0.5,
                 min_kp_score: float = MIN_KP_SCORE, band_px: float = 24.0) -> None:
        self.min_coverage = min_coverage
        self.prob_threshold = prob_threshold
        self.min_kp_score = min_kp_score
        self.band_px = band_px
        self.size = (CANONICAL_W, CANONICAL_H)
        self.landmarks = dict(CANONICAL_LANDMARKS)

    # ------------------------------------------------------------------ #
    def correspondences(self, keypoints_px: Optional[dict], yaw: float,
                        head_bbox_px: Optional[Sequence[float]] = None) -> list[list[tuple[str, tuple[float, float]]]]:
        """Candidate landmark assignments ``[(template_name, (x, y)), ...]``.

        More than one candidate is returned only for ambiguous single eyes /
        ears; each candidate is ordered with the prior-preferred one first.
        """
        ms = self.min_kp_score
        eyes = _parts(keypoints_px, "eye", ms)[:2]
        noses = _parts(keypoints_px, "nose", ms)[:1]
        ears = _parts(keypoints_px, "ear", ms)[:2]
        if len(eyes) == 2 and self._duplicate_eyes(eyes, noses, head_bbox_px):
            logger.debug("planar: two eye detections %.1f px apart treated as one eye",
                         float(np.hypot(eyes[0][0] - eyes[1][0], eyes[0][1] - eyes[1][1])))
            eyes = eyes[:1]  # keep the higher-scoring one
        fixed: list[tuple[str, tuple[float, float]]] = []
        options: list[list[list[tuple[str, tuple[float, float]]]]] = []

        if noses:
            fixed.append(("nose", (noses[0][0], noses[0][1])))
        if len(eyes) == 2:
            e = sorted(eyes, key=lambda q: q[0])
            fixed += [("left_eye", (e[0][0], e[0][1])), ("right_eye", (e[1][0], e[1][1]))]
        elif len(eyes) == 1:
            ex, ey = eyes[0][0], eyes[0][1]
            if abs(yaw) > 60.0 and yaw != 0:
                # horse faces image-left (yaw<0) -> its anatomical LEFT eye is
                # visible -> canonical image-right eye.
                fixed.append(("right_eye" if yaw < 0 else "left_eye", (ex, ey)))
            else:
                prior = self._side_prior(ex, yaw, noses, head_bbox_px)
                first, second = ("right_eye", "left_eye") if prior == "right" else ("left_eye", "right_eye")
                options.append([[(first, (ex, ey))], [(second, (ex, ey))]])
        if len(ears) == 2:
            e = sorted(ears, key=lambda q: q[0])
            fixed += [("left_ear_base", (e[0][0], e[0][1])), ("right_ear_base", (e[1][0], e[1][1]))]
        elif len(ears) == 1:
            ex, ey = ears[0][0], ears[0][1]
            prior = self._side_prior(ex, yaw, noses or eyes, head_bbox_px)
            first, second = (("right_ear_base", "left_ear_base") if prior == "right"
                             else ("left_ear_base", "right_ear_base"))
            options.append([[(first, (ex, ey))], [(second, (ex, ey))]])

        if not options:
            return [fixed]
        cands = []
        for combo in itertools.product(*options):
            c = list(fixed)
            for part in combo:
                c += part
            cands.append(c)
        return cands

    @staticmethod
    def _duplicate_eyes(eyes: list[list[float]], noses: list[list[float]],
                        head_bbox_px: Optional[Sequence[float]]) -> bool:
        """Two eye boxes on the same eye (the detector often fires twice)."""
        d = float(np.hypot(eyes[0][0] - eyes[1][0], eyes[0][1] - eyes[1][1]))
        if noses:
            ref = float(np.mean([np.hypot(e[0] - noses[0][0], e[1] - noses[0][1]) for e in eyes]))
            return d < EYE_DUP_FRAC * ref
        if head_bbox_px is not None:
            return d < 0.06 * (float(head_bbox_px[3]) - float(head_bbox_px[1]))
        return False

    @staticmethod
    def _side_prior(x: float, yaw: float, refs: list[list[float]],
                    head_bbox_px: Optional[Sequence[float]]) -> str:
        """'left' / 'right' canonical side for a single ambiguous landmark."""
        if refs:
            rx = float(np.mean([r[0] for r in refs]))
            if abs(x - rx) > 1e-6:
                return "right" if x > rx else "left"
        if yaw < 0:
            return "right"
        if yaw > 0:
            return "left"
        if head_bbox_px is not None:
            return "right" if x > (head_bbox_px[0] + head_bbox_px[2]) / 2.0 else "left"
        return "left"

    def estimate_transform(self, keypoints_px: Optional[dict], yaw: float,
                           head_bbox_px: Optional[Sequence[float]] = None) -> Optional[dict[str, Any]]:
        """Pick the best similarity transform (mask px -> canonical px).

        Returns ``{"M", "residual", "scale", "rotation_deg", "landmarks", "fallback"}``
        or ``None`` when no landmark is usable.
        """
        cands = self.correspondences(keypoints_px, yaw, head_bbox_px)
        trials = [(rank, cand, 0) for rank, cand in enumerate(cands)]
        best = self._best_fit(trials, len(cands))
        if best is not None and best["residual"] > GOOD_RESIDUAL_PX:
            # leave-one-out: one spurious landmark (e.g. a false second eye in
            # a profile view) can wreck the fit; keep >= 3 points.
            loo = [(rank, [c for k, c in enumerate(cand) if k != i], 1)
                   for rank, cand in enumerate(cands) if len(cand) >= 4 for i in range(len(cand))]
            alt = self._best_fit(loo, len(cands))
            if alt is not None and alt["cost"] < best["cost"]:
                logger.debug("planar: leave-one-out refit %.1f -> %.1f px (dropped landmark)",
                             best["residual"], alt["residual"])
                best = alt
        if best is not None:
            return best
        # ---- single-landmark fallback: scale from head bbox height ----
        cand = cands[0] if cands else []
        if not cand or head_bbox_px is None:
            return None
        head_h = float(head_bbox_px[3]) - float(head_bbox_px[1])
        if head_h <= 1:
            return None
        name, (x, y) = cand[0]
        s = CANONICAL_H / head_h
        tx, ty = self.landmarks[name]
        M = np.array([[s, 0.0, tx - s * x], [0.0, s, ty - s * y]], np.float64)
        logger.info("planar: single landmark (%s) -> bbox-height scale %.3f, weight x%.1f",
                    name, s, FALLBACK_WEIGHT)
        return {"M": M, "residual": None, "scale": s, "rotation_deg": 0.0,
                "landmarks": [name], "fallback": True, "cost": None, "n_candidates": len(cands)}

    def _best_fit(self, trials, n_cands: int) -> Optional[dict[str, Any]]:
        best: Optional[dict[str, Any]] = None
        for rank, cand, n_drop in trials:
            if len(cand) < 2:
                continue
            src = np.array([p for _, p in cand], np.float64)
            dst = np.array([self.landmarks[n] for n, _ in cand], np.float64)
            try:
                M, res = estimate_similarity(src, dst)
            except ValueError:
                continue
            scale, rot = transform_params(M)
            if not (SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]):
                continue
            # rank 0 is the prior-preferred assignment; each set bit of the
            # rank = one landmark assigned against the prior.
            n_disagree = bin(rank).count("1") if n_cands > 1 else 0
            cost = (res + PRIOR_PENALTY_PX * n_disagree + DROP_PENALTY_PX * n_drop
                    + ROT_PENALTY_PER_DEG * max(0.0, abs(rot) - ROT_PENALTY_START_DEG))
            if best is None or cost < best["cost"]:
                best = {"M": M, "residual": res, "scale": scale, "rotation_deg": rot,
                        "landmarks": [n for n, _ in cand], "fallback": False, "cost": cost,
                        "n_candidates": n_cands, "dropped": n_drop}
        return best

    @staticmethod
    def near_side(yaw: float, landmarks: Sequence[str]) -> tuple[float, str]:
        """Signed yaw used for the visibility profile and where its sign came from.

        The Phase-1 yaw SIGN is heuristic and unreliable, so when exactly one
        eye was matched its template side decides: ``right_eye`` (horse's
        anatomical left visible) behaves like yaw < 0, ``left_eye`` like yaw > 0.
        The magnitude always comes from Phase 1.
        """
        eyes = [n for n in landmarks if n.endswith("_eye")]
        if len(eyes) == 1 and abs(yaw) > 30.0:
            sign = -1.0 if eyes[0] == "right_eye" else 1.0
            return sign * abs(yaw), "single_eye"
        return float(yaw), "yaw"

    # ------------------------------------------------------------------ #
    def map_frame(self, mask, prob, keypoints_px, yaw, quality, frame=-1, head_bbox_px=None):
        if mask is None or np.asarray(mask).size == 0:
            logger.warning("planar: frame %s has an empty mask -> skipped", frame)
            return None
        tr = self.estimate_transform(keypoints_px, float(yaw), head_bbox_px)
        if tr is None:
            logger.warning("planar: frame %s has no usable landmarks -> skipped", frame)
            return None
        res = tr["residual"]
        if res is not None and res >= REJECT_RESIDUAL_PX:
            logger.warning("planar: frame %s landmark fit residual %.1f px >= %.0f -> skipped",
                           frame, res, REJECT_RESIDUAL_PX)
            return None
        res_w = 1.0 if res is None or res <= GOOD_RESIDUAL_PX else \
            (REJECT_RESIDUAL_PX - res) / (REJECT_RESIDUAL_PX - GOOD_RESIDUAL_PX)
        vis_yaw, side_src = self.near_side(float(yaw), tr["landmarks"])
        M = tr["M"].astype(np.float64)
        W, H = self.size
        p = _as_prob(mask, prob)
        m = (np.asarray(mask)[..., 0] if np.asarray(mask).ndim == 3 else np.asarray(mask)) > 0
        warped_p = cv2.warpAffine(p, M, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # binary accepted-component mask, nearest-neighbour warped -> stays 0/1
        warped_m = cv2.warpAffine(m.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        valid = cv2.warpAffine(np.ones(p.shape, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        q = float(np.clip(quality, 0.0, 1.0))
        factor = FALLBACK_WEIGHT if tr["fallback"] else 1.0
        weight = (q * view_score(yaw) * factor * res_w) * visibility_profile(vis_yaw, W, H, self.band_px) * valid
        meta = {k: tr.get(k) for k in ("residual", "scale", "rotation_deg", "landmarks", "fallback",
                                       "n_candidates", "dropped")}
        meta.update({"residual_weight": round(float(res_w), 4), "visibility_yaw": round(vis_yaw, 2),
                     "near_side_source": side_src})
        meta["M"] = [[round(float(v), 6) for v in row] for row in M]
        return CanonicalObservation(frame=int(frame), yaw=float(yaw), view=yaw_bin(yaw),
                                    prob=np.clip(warped_p, 0, 1).astype(np.float32),
                                    weight=weight.astype(np.float32), quality=q,
                                    mask=warped_m, meta=meta)

    def aggregate(self, observations: Sequence[CanonicalObservation]) -> CanonicalReference:
        obs = [o for o in observations if o is not None and float(np.max(o.weight)) > 0]
        if not obs:
            W, H = self.size
            z = np.zeros((H, W), np.float32)
            return CanonicalReference(prob=z, coverage=z.copy(), mask=z.astype(np.uint8), consistency=z.copy(),
                                      view_count=0, views={}, frames=[], mapper=self.name, weight_sum=z.copy(),
                                      support=z.copy(), n_support=np.zeros((H, W), np.int32),
                                      support_stack=np.zeros((0, H, W), np.uint8))
        prob, cov, cons, mask, sw = aggregate_weighted([o.prob for o in obs], [o.weight for o in obs],
                                                       self.min_coverage, self.prob_threshold)
        support, n_support, stack = aggregate_support([o.mask for o in obs], [o.weight for o in obs])
        views: dict[str, int] = {}
        for o in obs:
            views[o.view] = views.get(o.view, 0) + 1
        return CanonicalReference(prob=prob, coverage=cov, mask=mask, consistency=cons,
                                  view_count=len(obs), views=views, frames=[o.frame for o in obs],
                                  mapper=self.name, weight_sum=sw, support=support, n_support=n_support,
                                  support_stack=stack)
