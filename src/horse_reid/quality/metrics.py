"""Individual quality metrics for a horse head crop. All scores are in [0, 1]
(higher = better) unless stated otherwise."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..types import BBox, bbox_area, bbox_height, bbox_intersection, bbox_width


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def face_size_score(head_bbox: BBox, frame_shape: Sequence[int], ref_px: float = 300.0) -> float:
    """``sqrt(area) / ref_px`` clipped to [0, 1] (a 300x300 px head -> 1.0).

    ``frame_shape`` is accepted for future resolution normalisation.
    """
    del frame_shape  # TODO(phase1-upgrade): normalise by frame resolution if inputs vary.
    a = bbox_area(head_bbox)
    if a <= 0:
        return 0.0
    return float(np.clip(math.sqrt(a) / ref_px, 0.0, 1.0))


def blur_variance(crop: np.ndarray, resize_width: int = 256) -> float:
    """Variance of the Laplacian on the grayscale crop resized to a fixed width."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = _to_gray(crop)
    h, w = gray.shape[:2]
    if w < 2 or h < 2:
        return 0.0
    nh = max(2, int(round(h * resize_width / w)))
    interp = cv2.INTER_AREA if w > resize_width else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (resize_width, nh), interpolation=interp)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_score_from_var(blur_var: float, ref: float = 150.0) -> float:
    """Sharpness in [0, 1] from a Laplacian variance: ``1 - exp(-blur_var / ref)``."""
    return float(1.0 - math.exp(-max(float(blur_var), 0.0) / max(float(ref), 1e-6)))


def blur_score(crop: np.ndarray, ref: float = 150.0, resize_width: int = 256) -> float:
    """Sharpness in [0, 1]: ``1 - exp(-laplacian_var / ref)``."""
    return blur_score_from_var(blur_variance(crop, resize_width), ref)


def exposure_score(crop: np.ndarray, dark: float = 50.0, bright: float = 205.0,
                   clip_tol: float = 0.15) -> float:
    """Exposure quality from grayscale mean and the clipped-pixel fraction.

    * mean in [dark, bright] -> 1, decaying linearly to 0 at 0 / 255;
    * fraction of pixels <= 10 or >= 245 above ``clip_tol`` is penalised
      linearly (0 at ``clip_tol + 0.35``).
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = _to_gray(crop)
    m = float(gray.mean())
    if m < dark:
        mean_s = m / dark
    elif m > bright:
        mean_s = (255.0 - m) / (255.0 - bright)
    else:
        mean_s = 1.0
    clipped = float(((gray <= 10) | (gray >= 245)).mean())
    clip_s = 1.0 if clipped <= clip_tol else max(0.0, 1.0 - (clipped - clip_tol) / 0.35)
    return float(np.clip(mean_s, 0.0, 1.0) * clip_s)


def occlusion_score(head_bbox: BBox, other_bboxes: Iterable[BBox],
                    frame_shape: Optional[Sequence[int]] = None,
                    border_margin: int = 2, border_penalty: float = 0.7) -> float:
    """``1 - max_i(|head ∩ other_i| / |head|)``; multiplied by ``border_penalty``
    if the head box touches the frame border (likely truncated).

    # TODO(phase1-upgrade): use instance masks instead of boxes (a person box
    # overlapping the head box does not always occlude it).
    """
    area = bbox_area(head_bbox)
    if area <= 0:
        return 0.0
    worst = 0.0
    for ob in other_bboxes:
        worst = max(worst, bbox_intersection(head_bbox, ob) / area)
    score = 1.0 - min(1.0, worst)
    if frame_shape is not None:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = head_bbox
        if x1 <= border_margin or y1 <= border_margin or x2 >= w - border_margin or y2 >= h - border_margin:
            score *= border_penalty
    return float(score)


def yaw_estimate(crop: np.ndarray, bbox_wh: Optional[tuple[int, int]] = None) -> tuple[float, float]:
    """HEURISTIC head yaw estimate. Returns ``(yaw_deg, confidence)``.

    # TODO(phase1-upgrade): replace with a landmark-based estimate (eyes,
    # nostrils, ears from a horse keypoint model) or a learned yaw regressor.

    Cues:
      1. aspect ratio w/h of the head box: profile heads are wide
         (w/h > 1.1 -> 90 deg), frontal heads narrow (w/h < 0.7 -> 0 deg),
         linear in between;
      2. bilateral symmetry: normalised cross-correlation between the
         grayscale crop and its horizontal mirror (high -> frontal).

    ``|yaw|`` is the mean of both cues (0..90 deg); the sign (+ = facing
    right in the image) comes from which half has more edge energy (the nose
    side) and is unreliable. Confidence is the agreement between the cues,
    halved because both cues are weak.
    """
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return 0.0, 0.0
    h, w = crop.shape[:2]
    bw, bh = bbox_wh if bbox_wh is not None else (w, h)
    ratio = bw / max(bh, 1)
    yaw_aspect = float(np.clip((ratio - 0.7) / (1.1 - 0.7), 0.0, 1.0)) * 90.0

    gray = cv2.resize(_to_gray(crop), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = gray - gray.mean()
    m = g[:, ::-1]
    denom = float(np.sqrt((g * g).sum() * (m * m).sum()))
    ncc = float((g * m).sum() / denom) if denom > 1e-6 else 1.0
    # ncc >= 0.85 -> frontal (0 deg); ncc <= 0.30 -> profile (90 deg).
    yaw_sym = float(np.clip((0.85 - ncc) / (0.85 - 0.30), 0.0, 1.0)) * 90.0

    abs_yaw = 0.5 * yaw_aspect + 0.5 * yaw_sym
    conf = 0.5 * (1.0 - abs(yaw_aspect - yaw_sym) / 90.0)

    edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    left, right = float(edges[:, :32].sum()), float(edges[:, 32:].sum())
    sign = 1.0 if right >= left else -1.0
    return float(sign * abs_yaw), float(conf)


def _parts(keypoints: Optional[Mapping[str, Any]], part: str, min_score: float = 0.0) -> list[list[float]]:
    """Keypoints ``[x, y, score]`` of ``part`` with ``score >= min_score``, best first."""
    if not keypoints:
        return []
    pts = [list(map(float, p)) for p in (keypoints.get(part) or []) if len(p) >= 2]
    pts = [p if len(p) >= 3 else p + [1.0] for p in pts]
    return sorted((p for p in pts if p[2] >= min_score), key=lambda p: -p[2])


def yaw_from_landmarks(head_bbox: BBox, keypoints: Optional[Mapping[str, Any]],
                       min_score: float = 0.25) -> Optional[tuple[float, float]]:
    """HEURISTIC yaw from coarse eye / nose keypoints. Returns ``(yaw_deg, confidence)``
    (signed; negative = the horse faces image-left) or ``None`` if unusable.

    # TODO(phase1-upgrade): replace with a trained horse head-pose / keypoint
    # model; these rules only use the horizontal layout of detector boxes.

    Rules (eyes / nose with score >= ``min_score``; ``w`` = head box width,
    ``cx`` = head box centre x):

    * 2 eyes: ``r = |x_e1 - x_e2| / w``; ``|yaw| = clip(90 * (1 - r / 0.45), 0, 90)``,
      conf 0.8 (both eyes far apart -> frontal);
    * nose (+ 0 or 1 eye): ``o = clip((x_nose - cx) / (w / 2), -1, 1)``;
      ``|yaw| = 45 + 45|o|`` with one eye, ``30 + 60|o|`` with none; conf 0.6;
    * 1 eye only: ``|yaw| = 60``, conf 0.3;
    * otherwise ``None``.

    Sign: negative when the nose (else the mean eye position) lies left of ``cx``.
    """
    w = float(bbox_width(head_bbox))
    if w <= 0 or not keypoints:
        return None
    cx = (head_bbox[0] + head_bbox[2]) / 2.0
    eyes = _parts(keypoints, "eye", min_score)[:2]
    noses = _parts(keypoints, "nose", min_score)[:1]
    if len(eyes) == 2:
        r = abs(eyes[0][0] - eyes[1][0]) / w
        abs_yaw = float(np.clip(90.0 * (1.0 - r / 0.45), 0.0, 90.0))
        conf = 0.8
    elif noses:
        o = float(np.clip((noses[0][0] - cx) / (w / 2.0), -1.0, 1.0))
        abs_yaw = 45.0 + 45.0 * abs(o) if len(eyes) == 1 else 30.0 + 60.0 * abs(o)
        conf = 0.6
    elif len(eyes) == 1:
        abs_yaw, conf = 60.0, 0.3
    else:
        return None
    ref_x = noses[0][0] if noses else float(np.mean([e[0] for e in eyes]))
    sign = -1.0 if ref_x < cx else 1.0
    return float(sign * abs_yaw), float(conf)


def face_visibility_score(keypoints: Optional[Mapping[str, Any]]) -> float:
    """How much face evidence the detector found, in [0, 1].

    ``clip((max_eye_score + max_nose_score) / 0.8, 0, 1)`` (missing parts count
    as 0) + 0.1 if any ear is present (clipped to 1); 0.1 when ``keypoints`` is
    ``None`` (no landmark detector). Rear views get a head box but eye/nose
    scores < 0.3, hence a low visibility.
    """
    if keypoints is None:
        return 0.1
    eye = max((p[2] for p in _parts(keypoints, "eye")), default=0.0)
    nose = max((p[2] for p in _parts(keypoints, "nose")), default=0.0)
    v = float(np.clip((eye + nose) / 0.8, 0.0, 1.0))
    if _parts(keypoints, "ear"):
        v += 0.1
    return float(min(1.0, v))


def view_score(yaw_deg: float) -> float:
    """View usefulness: 1.0 for |yaw| <= 60 (frontal and 3/4 are equally
    valuable), linearly down to 0.4 at 90 (profile is still useful)."""
    a = min(abs(float(yaw_deg)), 90.0)
    if a <= 60.0:
        return 1.0
    return float(1.0 - 0.6 * (a - 60.0) / 30.0)


def yaw_bin(yaw_deg: float, thresholds: tuple[float, float] = (30.0, 60.0)) -> str:
    """"frontal" (< t0), "three_quarter" (t0..t1) or "profile" (> t1)."""
    a = abs(float(yaw_deg))
    if a < thresholds[0]:
        return "frontal"
    if a <= thresholds[1]:
        return "three_quarter"
    return "profile"
