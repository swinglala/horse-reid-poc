"""SAM boundary refinement of colour-candidate white markings (default segmenter)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .adaptive_color import (AdaptiveColorSegmenter, contour_shape, coat_zmap, face_axis, rect_sides,
                             region_axis_angle, region_info)
from .base import HorseMarkingSegmenter, MarkingResult, full_face_mask

logger = logging.getLogger(__name__)


class SamRefinedSegmenter(HorseMarkingSegmenter):
    """Candidate generator (default :class:`AdaptiveColorSegmenter`) + MobileSAM refinement.

    For the ``max_components`` largest accepted candidate components:

    * positive point: candidate pixel nearest to the component centroid;
    * 2 negative points: face-mask pixels with coat z < ``neg_z_max`` at least
      ``neg_min_dist_frac`` x face width away from the component (nearest such
      pixel, then the nearest one on a clearly different side);
    * one SAM prompt per component (all prompts batched in one SAM call,
      ``points=[[pos, neg1, neg2], ...]``, ``labels=[[1, 0, 0], ...]``).

    A SAM mask is accepted only if (a) area < ``max_area_frac`` x face area,
    (b) mean z inside > ``min_mean_z`` and (c) IoU with the candidate dilated
    by ``iou_dilate_px`` > ``min_iou``; otherwise the candidate component is
    kept and the SAM mask is recorded in ``excluded`` as ``"sam_flood"``.
    Strap prior on SAM's object extent (extension of the colour shape prior):
    SAM tends to return the *whole object* a bright fragment belongs to; if
    that mask is strap-like - contour elongation ``P^2/(4A)`` >
    ``strap_elongation``, length > ``strap_min_len_frac`` x face width and
    thickness < ``strap_max_thick_frac`` x face width - the candidate is a
    piece of a halter/noseband strap (straight or curved) and is excluded as
    ``"strap_shape"`` (``strap_source="sam"``) instead of being kept.
    Final mask = union of accepted SAM masks (inside the face mask) and kept
    candidates; ``prob = max(base prob, 0.9 * accepted SAM masks)``.

    SAM is lazy-loaded; on any SAM failure the base result is returned (with a
    warning and ``stats["sam_error"]``).
    """

    name = "sam_refined"

    def __init__(self, base: Optional[HorseMarkingSegmenter] = None, sam_weights: str | Path = "models/mobile_sam.pt",
                 device: str = "cpu", max_components: int = 5, max_area_frac: float = 0.20,
                 min_mean_z: float = 1.5, min_iou: float = 0.3, iou_dilate_px: int = 5,
                 neg_z_max: float = 0.5, neg_min_dist_frac: float = 0.15, strap_elongation: float = 6.25,
                 strap_min_len_frac: float = 0.25, strap_max_thick_frac: float = 0.15,
                 stripe_axis_max_deg: float = 30.0, strong_z: float = 6.5, strong_area_frac: float = 0.01,
                 strap_min_cover: float = 0.5, compact_max_ratio: float = 2.5,
                 swallow_area_ratio: float = 10.0) -> None:
        self.base = base if base is not None else AdaptiveColorSegmenter()
        self.sam_weights = Path(sam_weights)
        self.device = device
        self.max_components = max_components
        self.max_area_frac = max_area_frac
        self.min_mean_z = min_mean_z
        self.min_iou = min_iou
        self.iou_dilate_px = iou_dilate_px
        self.neg_z_max = neg_z_max
        self.neg_min_dist_frac = neg_min_dist_frac
        self.strap_elongation = strap_elongation
        self.strap_min_len_frac = strap_min_len_frac
        self.strap_max_thick_frac = strap_max_thick_frac
        # A strongly white (strong_z, strong_area_frac) candidate whose SAM mask runs along the
        # face axis is a blaze, not a strap; weaker axis-aligned strips (coat sheen) stay straps.
        self.stripe_axis_max_deg = stripe_axis_max_deg
        self.strong_z = strong_z
        self.strong_area_frac = strong_area_frac
        self.strap_min_cover = strap_min_cover   # SAM mask must cover this fraction of the candidate to call it a strap
        self.compact_max_ratio = compact_max_ratio     # candidate min-area-rect long/short below this = compact blob
        self.swallow_area_ratio = swallow_area_ratio   # SAM mask > this x candidate area on a compact blob = merged with neighbour
        self._sam: Any = None

    # ------------------------------------------------------------------ #
    def _load_sam(self) -> Any:
        if self._sam is None:
            from ..config import resolve_project_path
            from ultralytics import SAM

            path = resolve_project_path(self.sam_weights)
            if not path.exists():
                raise FileNotFoundError(f"SAM weights not found: {path}")
            self._sam = SAM(str(path))
            logger.info("Loaded SAM %s on %s", path, self.device)
        return self._sam

    @staticmethod
    def _positive_point(region: np.ndarray) -> list[float]:
        ys, xs = np.nonzero(region)
        cx, cy = xs.mean(), ys.mean()
        i = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        return [float(xs[i]), float(ys[i])]

    def _negative_points(self, region: np.ndarray, fm: np.ndarray, z: np.ndarray, face_w: int,
                         pos: list[float]) -> list[list[float]]:
        dist = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5)
        eligible = fm & (z < self.neg_z_max) & (dist >= self.neg_min_dist_frac * face_w)
        ys, xs = np.nonzero(eligible)
        if xs.size == 0:
            return []
        d = dist[ys, xs]
        i1 = int(np.argmin(d))
        negs = [[float(xs[i1]), float(ys[i1])]]
        ang = np.arctan2(ys - pos[1], xs - pos[0])
        a1 = ang[i1]
        diff = np.abs((ang - a1 + math.pi) % (2 * math.pi) - math.pi)
        other = np.flatnonzero(diff > math.pi / 2)
        if other.size:
            i2 = int(other[np.argmin(d[other])])
            negs.append([float(xs[i2]), float(ys[i2])])
        return negs

    def _run_sam(self, image: np.ndarray, prompts: list[tuple[list[list[float]], list[int]]]) -> list[np.ndarray]:
        """Run SAM on ``(points, labels)`` prompts; one bool mask per prompt.

        Prompts with the same number of points are batched into one call using
        the nested form ``points=[[p1, p2, p3], ...]``, ``labels=[[1, 0, 0], ...]``
        (ultralytics ``Predictor._prepare_prompts``: a 3-D points array is
        (num_prompts, points_per_prompt, 2); a flat 2-D list would instead make
        every point its own prompt).
        """
        sam = self._load_sam()
        h, w = image.shape[:2]
        out: list[Optional[np.ndarray]] = [None] * len(prompts)
        groups: dict[int, list[int]] = {}
        for i, (pts, _) in enumerate(prompts):
            groups.setdefault(len(pts), []).append(i)
        for _, idxs in groups.items():
            pts = [prompts[i][0] for i in idxs]
            lbl = [prompts[i][1] for i in idxs]
            r = sam.predict(image, points=pts, labels=lbl, device=self.device, verbose=False)[0]
            if r.masks is None:
                raise RuntimeError("SAM returned no masks")
            data = r.masks.data.cpu().numpy()
            if data.shape[0] != len(idxs):
                raise RuntimeError(f"SAM returned {data.shape[0]} masks for {len(idxs)} prompts")
            for j, i in enumerate(idxs):
                m = data[j]
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                out[i] = m > 0.5
        return [m for m in out if m is not None]

    # ------------------------------------------------------------------ #
    def predict(self, image: np.ndarray, face_mask: Optional[np.ndarray] = None,
                keypoints: Optional[dict] = None, coat_mask: Optional[np.ndarray] = None) -> MarkingResult:
        """
        Returns:
            mask: binary or probability mask
        """
        base = self.base.predict(image, face_mask, keypoints, coat_mask=coat_mask)
        fm = full_face_mask(image, face_mask)
        n, labels = cv2.connectedComponents(base.mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            base.stats["base_method"] = base.method
            base.method = self.name
            base.stats["sam_calls"] = 0
            return base
        try:
            return self._refine(image, fm, base, n, labels, keypoints)
        except Exception as e:  # SAM load / inference failure -> degrade gracefully
            logger.warning("SAM refinement failed (%s); returning %s result", e, base.method)
            base.stats["sam_error"] = str(e)
            base.stats["base_method"] = base.method
            base.method = f"{self.name}:degraded"
            return base

    def _refine(self, image: np.ndarray, fm: np.ndarray, base: MarkingResult, n: int,
                labels: np.ndarray, keypoints: Optional[dict] = None) -> MarkingResult:
        axis = face_axis(keypoints)
        z = base.debug.get("z")
        if z is None or z.shape != fm.shape:
            z, _, _ = coat_zmap(image, fm)
        face_area = int(fm.sum())
        cols = np.flatnonzero(fm.any(axis=0))
        face_w = int(cols[-1] - cols[0] + 1) if cols.size else fm.shape[1]
        regions = [labels == i for i in range(1, n)]
        order = sorted(range(len(regions)), key=lambda i: -int(regions[i].sum()))
        top = order[: self.max_components]

        prompts, prompt_comp = [], []
        for i in top:
            pos = self._positive_point(regions[i])
            negs = self._negative_points(regions[i], fm, z, face_w, pos)
            prompts.append(([pos] + negs, [1] + [0] * len(negs)))
            prompt_comp.append(i)
        sam_masks = self._run_sam(image, prompts)

        k = 2 * self.iou_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        final = np.zeros(fm.shape, bool)
        sam_union = np.zeros(fm.shape, bool)
        excl_mask = base.debug.get("excluded_mask", np.zeros(fm.shape, np.uint8)).copy()
        components: list[dict[str, Any]] = []
        excluded = list(base.excluded)
        sam_by_comp = dict(zip(prompt_comp, zip(sam_masks, prompts)))
        for i, region in enumerate(regions):
            if i not in sam_by_comp:
                final |= region
                components.append(region_info(region, face_area, z, accepted_by="candidate"))
                continue
            sm, (pts, lbl) = sam_by_comp[i]
            area_frac = float(sm.sum()) / max(face_area, 1)
            mean_z = float(z[sm].mean()) if sm.any() else -1e9
            dil = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)
            inter = float((sm & dil).sum())
            union = float((sm | dil).sum())
            iou = inter / union if union > 0 else 0.0
            elong, length, thick = contour_shape(sm) if sm.any() else (0.0, 0.0, 0.0)
            sam_info = {"sam_area_frac": round(area_frac, 6), "sam_mean_z": round(mean_z, 3),
                        "sam_iou": round(iou, 3), "sam_elongation": round(elong, 2),
                        "sam_length": round(length, 1), "sam_thickness": round(thick, 1),
                        "sam_points": pts, "sam_labels": lbl}
            axis_deg = region_axis_angle(sm, axis) if sm.any() else 90.0
            region_axis_deg = region_axis_angle(region, axis)
            strong = bool(region.sum() >= self.strong_area_frac * face_area and float(z[region].mean()) >= self.strong_z)
            sam_info["sam_axis_deg"] = round(axis_deg, 1)
            sam_info["axis_deg"] = round(region_axis_deg, 1)
            sam_info["strong"] = strong
            # The SAM mask describes the candidate only if it actually covers it. Prompted on a
            # small star next to a white halter, SAM returns the halter (covers ~0 % of the
            # star); that mask must not re-label the star as a strap.
            cover = float((sm & region).sum()) / max(float(region.sum()), 1.0)
            sam_info["sam_cover"] = round(cover, 3)
            # A compact blob (a star) whose SAM mask is many times larger than itself: SAM
            # merged it with the adjacent halter. The mask shape says nothing about the blob.
            r_long, r_short = rect_sides(region)
            compact = r_short > 0 and r_long / r_short < self.compact_max_ratio
            swallowed = compact and float(sm.sum()) > self.swallow_area_ratio * float(region.sum())
            sam_info["sam_swallowed"] = swallowed
            # A strong candidate that itself runs along the face axis is a blaze; the SAM
            # mask may have wandered onto a strap (low IoU) and must not re-label it.
            blaze_like = strong and min(axis_deg, region_axis_deg) <= self.stripe_axis_max_deg
            strap_like = (elong > self.strap_elongation and length > self.strap_min_len_frac * face_w
                          and thick < self.strap_max_thick_frac * face_w and cover >= self.strap_min_cover
                          and not blaze_like and not swallowed)
            if strap_like:
                excluded.append(region_info(region, face_area, z, reason="strap_shape", strap_source="sam",
                                            **sam_info))
                excl_mask[region] = 1
                excl_mask[sm & fm] = 1
            elif area_frac < self.max_area_frac and mean_z > self.min_mean_z and iou > self.min_iou:
                refined = (sm & fm) | region
                final |= refined
                sam_union |= sm & fm
                components.append(region_info(refined, face_area, z, accepted_by="sam", **sam_info))
            else:
                final |= region
                components.append(region_info(region, face_area, z, accepted_by="candidate", **sam_info))
                if sm.any():
                    excluded.append(region_info(sm, face_area, z, reason="sam_flood", **sam_info))
                    excl_mask[sm & fm] = 1
        components.sort(key=lambda c: -c["area_px"])
        mask = final.astype(np.uint8)
        prob = np.maximum(base.prob, 0.9 * sam_union.astype(np.float32)).astype(np.float32)
        n_final = cv2.connectedComponents(mask, connectivity=8)[0] - 1
        stats = dict(base.stats)
        stats.update({
            "base_method": base.method,
            "marking_area_frac": round(float(mask.sum()) / max(face_area, 1), 6),
            "n_components": int(n_final),
            "candidate_area_frac": base.stats.get("marking_area_frac"),
            "sam_calls": len({len(p[0]) for p in prompts}),
            "sam_prompts": len(prompts),
            "sam_accepted": sum(c["accepted_by"] == "sam" for c in components),
        })
        debug = dict(base.debug)
        debug["excluded_mask"] = excl_mask
        debug["candidate_prob"] = base.prob
        return MarkingResult(mask=mask, prob=prob, face_mask=fm, components=components, excluded=excluded,
                             method=self.name, stats=stats, debug=debug)
