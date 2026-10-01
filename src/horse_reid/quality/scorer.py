"""Combine individual metrics into a single per-frame quality score."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import numpy as np

from ..config import QualityParams, QualityWeights
from ..types import BBox, Detection, FrameScore, HeadDetection, bbox_height, bbox_width, crop
from . import metrics as M

logger = logging.getLogger(__name__)


def combine_score(components: dict[str, Any], params: QualityParams, weights: QualityWeights) -> tuple[float, bool]:
    """Pure "components -> (score, gated)" combination.

    ``components`` needs size, blur, exposure, occlusion, view, det_conf,
    visibility, head_method and optionally ``visibility_measured`` (default:
    True only for gd_fallback heads). Heads whose method contains
    "gd_fallback" (the open-vocabulary detector found no head) use
    ``params.fallback_visibility`` and are always gated.
    """
    c = dict(components)
    w, p = weights, params
    method = str(c.get("head_method") or "")
    head_conf = c.get("head_conf")
    if head_conf is not None:
        horse_conf = float(c.get("horse_conf", c["det_conf"]))
        c["det_conf"] = horse_conf * float(np.clip(head_conf / max(p.head_conf_ref, 1e-6), 0.0, 1.0))
    low_head_conf = (head_conf is not None and "heuristic" not in method
                     and head_conf < p.gate_min_head_conf)
    ratio = c.get("head_area_ratio")
    implausible = ratio is not None and ratio > p.gate_max_head_area_ratio
    fallback = "gd_fallback" in method
    visibility = p.fallback_visibility if fallback else float(c["visibility"])
    measured = bool(c.get("visibility_measured", False)) or fallback
    s = (w.size * c["size"] + w.blur * c["blur"] + w.exposure * c["exposure"]
         + w.occlusion * c["occlusion"] + w.view * c["view"] + w.det_conf * c["det_conf"]
         + w.visibility * visibility) / w.total()
    gated = bool(c["occlusion"] < p.gate_min_occlusion or c["blur"] < p.gate_min_blur
                 or c["exposure"] < p.gate_min_exposure
                 or (measured and visibility < p.gate_min_visibility)
                 or fallback or low_head_conf or implausible)
    if gated:
        s *= p.gate_factor
    if "heuristic" in method and not fallback:
        s *= p.heuristic_head_factor
    return float(s), gated


def _area_ratio(head: BBox, horse: BBox) -> float:
    ha = max(0.0, head[2] - head[0]) * max(0.0, head[3] - head[1])
    za = max(1e-6, (horse[2] - horse[0]) * (horse[3] - horse[1]))
    return float(ha / za)


def resolve_blur_ref(blur_vars: Iterable[float], params: QualityParams) -> float:
    """Laplacian-variance reference used to turn ``blur_var`` into the blur score.

    ``"absolute"`` mode: ``params.blur_ref``. ``"adaptive"`` mode:
    ``max(params.blur_ref_min, median(blur_vars))`` (finite values only;
    ``params.blur_ref`` if there are none), i.e. the median candidate of the
    run maps to a blur score of ``1 - 1/e ~= 0.63``.
    """
    if params.blur_ref_mode == "absolute":
        return float(params.blur_ref)
    v = np.asarray([float(b) for b in blur_vars], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(params.blur_ref)
    return float(max(params.blur_ref_min, float(np.median(v))))


def rescore(fs: FrameScore, params: QualityParams, weights: QualityWeights,
            blur_ref: Optional[float] = None, keep_original: bool = True) -> FrameScore:
    """Recompute ``fs.blur`` / ``score`` / ``gated`` / ``visibility`` in place from
    stored components.

    ``fs.blur`` is recomputed from the raw ``fs.blur_var`` with ``blur_ref``
    (default ``params.blur_ref``, i.e. absolute mode; see
    :func:`resolve_blur_ref`). With ``keep_original`` the previous score is kept
    once in ``extra["score_original"]``.
    """
    if keep_original:
        fs.extra.setdefault("score_original", fs.score)
    ref = float(params.blur_ref if blur_ref is None else blur_ref)
    fs.blur = M.blur_score_from_var(fs.blur_var, ref)
    measured = _landmark_keypoints((fs.extra or {}).get("keypoints")) is not None
    horse_conf = float(fs.extra.setdefault("horse_conf", fs.det_conf))
    fs.extra["head_conf"] = fs.head_conf
    comps = {"size": fs.size, "blur": fs.blur, "exposure": fs.exposure, "occlusion": fs.occlusion,
             "view": fs.view, "det_conf": horse_conf, "horse_conf": horse_conf,
             "head_conf": fs.head_conf, "visibility": fs.visibility,
             "head_method": fs.head_method, "visibility_measured": measured,
             "head_area_ratio": _area_ratio(fs.head_bbox, fs.horse_bbox)}
    s, gated = combine_score(comps, params, weights)
    fs.det_conf = horse_conf * float(np.clip(fs.head_conf / max(params.head_conf_ref, 1e-6), 0.0, 1.0))
    if "gd_fallback" in (fs.head_method or ""):
        fs.visibility = params.fallback_visibility
    fs.score, fs.gated = s, gated
    return fs


def rescore_all(scores: Iterable[FrameScore], params: QualityParams, weights: QualityWeights,
                keep_original: bool = True) -> float:
    """Resolve the blur reference over all ``scores`` (see :func:`resolve_blur_ref`)
    and :func:`rescore` each of them in place. Returns the ``blur_ref`` used.

    This is the single "components -> score" step shared by ``run_phase1``
    (after the detection pass, ``keep_original=False``) and ``run_reselect``.
    """
    scores = list(scores)
    ref = resolve_blur_ref((s.blur_var for s in scores), params)
    for fs in scores:
        rescore(fs, params, weights, blur_ref=ref, keep_original=keep_original)
    logger.info("blur_ref=%.2f (mode %s, %d candidates)", ref, params.blur_ref_mode, len(scores))
    return ref


class FrameQualityScorer:
    """Score the head of one horse in one frame.

    ``score = sum_i w_i * metric_i / sum_i w_i`` over size, blur, exposure,
    occlusion, view, detection confidence and face visibility. Hard gates
    (occlusion < 0.5, blur < 0.15, exposure < 0.2, visibility < 0.25 by
    default) demote a frame by ``gate_factor`` (0.2) instead of discarding it.

    The score returned here uses the absolute ``blur_ref``; it is provisional.
    The pipeline stores the raw ``blur_var`` and re-combines every score with the
    run-level (adaptive) ``blur_ref`` via :func:`rescore_all` before selection.
    Heuristic head boxes (method contains "heuristic") are further multiplied
    by ``heuristic_head_factor`` (0.85) so real detections win ties.

    Yaw comes from the eye/nose keypoints when available
    (:func:`metrics.yaw_from_landmarks`), else from the appearance heuristic
    (:func:`metrics.yaw_estimate`).

    The visibility gate only applies when face visibility is actually measured:
    the head comes from a landmark detector (keypoints contain ear/eye/nose
    entries) or is a fallback box produced inside a landmark detector
    ("gd_fallback" in the method, i.e. the real detector found no head). A
    stand-alone heuristic run (no landmarks at all) gets visibility 0.1 in the
    weighted sum but is not gated, otherwise every frame would be demoted.

    # TODO(phase1-upgrade): learn the weights from labelled "good for ReID" frames.
    """

    def __init__(self, weights: Optional[QualityWeights] = None, params: Optional[QualityParams] = None) -> None:
        self.w = weights or QualityWeights()
        self.p = params or QualityParams()
        if self.w.total() <= 0:
            raise ValueError("quality weights must sum to > 0")

    def score(
        self,
        frame: np.ndarray,
        horse: Detection,
        head: HeadDetection,
        others: Iterable[BBox] = (),
    ) -> FrameScore:
        """Compute a :class:`FrameScore`.

        Args:
            frame: Full BGR frame.
            horse: The horse detection the head belongs to.
            head: Head detection inside ``horse.bbox``.
            others: Boxes of potential occluders (persons, other horses).
        """
        p, w = self.p, self.w
        head_crop = crop(frame, head.bbox)
        size = M.face_size_score(head.bbox, frame.shape, p.size_ref_px)
        blur_var = M.blur_variance(head_crop, p.blur_resize_width)
        # Provisional (absolute blur_ref); run_phase1 re-combines all scores with the
        # run-level blur_ref via rescore_all() once the pass is done.
        blur = M.blur_score_from_var(blur_var, p.blur_ref)
        exposure = M.exposure_score(head_crop)
        occlusion = M.occlusion_score(head.bbox, others, frame.shape)
        landmarks = _landmark_keypoints(head.keypoints)
        yl = M.yaw_from_landmarks(head.bbox, landmarks) if landmarks is not None else None
        if yl is not None:
            (yaw, yaw_conf), yaw_source = yl, "landmarks"
        else:
            yaw, yaw_conf = M.yaw_estimate(head_crop, (bbox_width(head.bbox), bbox_height(head.bbox)))
            yaw_source = "appearance"
        view = M.view_score(yaw)
        det_conf = float(np.clip(horse.confidence, 0.0, 1.0))
        visibility = M.face_visibility_score(landmarks)
        if "gd_fallback" in head.method:
            visibility = p.fallback_visibility
        comps = {"size": size, "blur": blur, "exposure": exposure, "occlusion": occlusion, "view": view,
                 "det_conf": det_conf, "horse_conf": det_conf, "head_conf": head.confidence,
                 "visibility": visibility, "head_method": head.method,
                 "visibility_measured": landmarks is not None,
                 "head_area_ratio": _area_ratio(head.bbox, horse.bbox)}
        s, gated = combine_score(comps, p, w)
        horse_conf = det_conf
        det_conf = horse_conf * float(np.clip(head.confidence / max(p.head_conf_ref, 1e-6), 0.0, 1.0))
        extra: dict[str, Any] = {"yaw_source": yaw_source, "horse_conf": horse_conf,
                                 "head_conf": float(head.confidence)}
        if head.keypoints is not None:
            extra["keypoints"] = head.keypoints
        return FrameScore(
            frame=horse.frame, time_s=horse.time_s, track_id=horse.track_id,
            horse_bbox=horse.bbox, head_bbox=head.bbox, head_conf=head.confidence,
            head_method=head.method, size=size, blur_var=blur_var, blur=blur,
            exposure=exposure, occlusion=occlusion, yaw=yaw, yaw_conf=yaw_conf,
            view=view, det_conf=det_conf, score=float(s), visibility=visibility, gated=gated,
            extra=extra,
        )


_LANDMARK_PARTS = ("ear", "eye", "nose")


def _landmark_keypoints(kps: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The ear/eye/nose subset of ``kps``, or ``None`` if the detector does not
    produce landmarks at all (e.g. the heuristic ``poll_top`` point)."""
    if not kps or not any(k in kps for k in _LANDMARK_PARTS):
        return None
    return {k: kps.get(k) or [] for k in _LANDMARK_PARTS}
