"""Draw a canonical marking reference onto a user-supplied head diagram.

The canonical reference lives on the 256x320 frontal template whose five
landmarks are :data:`~horse_reid.canonical.planar.CANONICAL_LANDMARKS`. Given
the same five landmarks on any frontal drawing (e.g. a registration diagram
sheet), a least-squares affine map template -> drawing is estimated and the
aggregated ``support`` / ``mask`` / peaks are warped onto the drawing. Nothing
is re-detected: the drawing is only a background and the landmarks only fix the
alignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .planar import CANONICAL_H, CANONICAL_LANDMARKS, CANONICAL_W

LANDMARK_ORDER = ("left_ear_base", "right_ear_base", "left_eye", "right_eye", "nose")


def load_diagram(spec_path: str | Path) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """``(image_bgr, landmarks)`` from a JSON spec ``{"image": ..., "landmarks": {name: [x, y]}}``.
    Transparent PNGs are flattened onto white."""
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    img_path = spec_path.parent / spec["image"]
    raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(img_path)
    if raw.ndim == 2:
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        a = raw[:, :, 3:4].astype(np.float32) / 255.0
        img = (raw[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).round().astype(np.uint8)
    else:
        img = raw[:, :, :3]
    lm = {k: (float(v[0]), float(v[1])) for k, v in spec["landmarks"].items()}
    missing = [n for n in LANDMARK_ORDER if n not in lm]
    if missing:
        raise ValueError(f"diagram spec {spec_path} lacks landmarks {missing}")
    return img, lm


def estimate_affine(src: dict[str, tuple[float, float]], dst: dict[str, tuple[float, float]],
                    names=LANDMARK_ORDER) -> tuple[np.ndarray, float]:
    """Least-squares full affine (2x3) mapping ``src[name] -> dst[name]`` and the RMS
    residual in destination pixels. An affine (not a similarity) is used on purpose:
    drawings are not to scale and the eye-nose / ear-eye ratios differ from the template."""
    S = np.array([src[n] for n in names], np.float64)
    D = np.array([dst[n] for n in names], np.float64)
    A = np.hstack([S, np.ones((len(names), 1))])
    X, *_ = np.linalg.lstsq(A, D, rcond=None)       # 3x2
    M = X.T                                          # 2x3
    pred = A @ X
    rms = float(np.sqrt(((pred - D) ** 2).sum(1).mean()))
    return M, rms


def load_reference_maps(ref_dir: str | Path) -> dict[str, Any]:
    """Canonical maps written by ``build_reference`` (PNG encodings) plus the peaks."""
    ref_dir = Path(ref_dir)
    support = cv2.imread(str(ref_dir / "canonical_support.png"), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(ref_dir / "canonical_mask.png"), cv2.IMREAD_GRAYSCALE)
    nsup = cv2.imread(str(ref_dir / "canonical_nsupport.png"), cv2.IMREAD_GRAYSCALE)
    prob = cv2.imread(str(ref_dir / "canonical_prob.png"), cv2.IMREAD_GRAYSCALE)
    if support is None or mask is None:
        raise FileNotFoundError(f"canonical_support.png / canonical_mask.png not found in {ref_dir}")
    meta = json.loads((ref_dir / "horse_reference.json").read_text(encoding="utf-8"))
    r = meta.get("reference", meta)
    return {"support": support.astype(np.float32) / 255.0, "mask": mask > 0,
            "n_support": nsup.astype(np.int32) if nsup is not None else None,
            "prob": prob.astype(np.float32) / 255.0 if prob is not None else None,
            "peaks": r.get("canonical_peaks", []), "frames": r.get("frames", []),
            "horse_id": meta.get("horse_id", "unknown"), "views": r.get("views", {})}


def _warp(a: np.ndarray, M: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    return cv2.warpAffine(a, M, size, flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def draw_on_diagram(diagram_bgr: np.ndarray, landmarks: dict[str, tuple[float, float]], maps: dict[str, Any],
                    scale: float = 4.0, min_support: float = 0.15, fill_support: float = 0.5,
                    show_landmarks: bool = False, show_peaks: bool = False, show_tint: bool = False,
                    label: Optional[str] = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Warp the canonical ``support`` / ``mask`` / peaks onto ``diagram_bgr`` (upscaled by
    ``scale``). Returns ``(image, info)``; ``info`` has the affine, RMS residual and the
    peaks in diagram pixels.

    Rendering (default): support >= ``fill_support`` (accepted marking in at least that
    weighted fraction of the frames) is filled white with a black outline, nothing else.
    Optional: ``show_tint`` tints support >= ``min_support`` (yellow = weak .. red =
    strong), ``show_peaks`` draws the candidate peaks as circles labelled with their frame
    count, ``show_landmarks`` marks the five alignment points. The prob-based canonical
    mask is NOT used for the fill: prob is the base candidate probability and also lights
    up rejected objects such as a white halter.
    """
    h, w = diagram_bgr.shape[:2]
    W, H = int(round(w * scale)), int(round(h * scale))
    bg = cv2.resize(diagram_bgr, (W, H), interpolation=cv2.INTER_CUBIC)
    dst = {k: (v[0] * scale, v[1] * scale) for k, v in landmarks.items()}
    M, rms = estimate_affine(CANONICAL_LANDMARKS, dst)

    sup = _warp(maps["support"], M, (W, H))
    msk = sup >= fill_support
    out = bg.copy()
    if show_tint:
        # support tint: colour from yellow (0,220,255 BGR) at min_support to red (0,0,230) at 1.0
        t = np.clip((sup - min_support) / max(1e-6, 1.0 - min_support), 0, 1)
        tint = np.stack([np.zeros_like(t), 220 * (1 - t), 255 * (1 - t) + 230 * t], axis=-1)
        alpha = np.where(sup >= min_support, 0.35 + 0.5 * t, 0.0)[..., None]
        out = (out.astype(np.float32) * (1 - alpha) + tint * alpha).round().astype(np.uint8)
    # final mask: white fill + dark outline
    if msk.any():
        out[msk] = (255, 255, 255)
        cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 0, 0), max(1, int(round(scale / 2))), cv2.LINE_AA)
    # peaks (always reported in info; drawn only on request)
    peaks_px = []
    for p in maps.get("peaks", []):
        x, y = M @ np.array([p["x"], p["y"], 1.0])
        n = int(p.get("n_support", 0))
        peaks_px.append({**p, "diagram_x": float(x), "diagram_y": float(y)})
        if show_peaks:
            r = int(round(3 * scale + 1.5 * scale * min(n, 10) ** 0.5))
            cv2.circle(out, (int(round(x)), int(round(y))), r, (255, 200, 0), max(1, int(round(scale / 2))), cv2.LINE_AA)
            cv2.putText(out, f"{p.get('id', '')} n={n}", (int(round(x)) + r + 2, int(round(y)) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.08 * scale, (160, 90, 0), 1, cv2.LINE_AA)
    if show_landmarks:
        for name in LANDMARK_ORDER:
            x, y = dst[name]
            cv2.circle(out, (int(round(x)), int(round(y))), max(2, int(round(scale))), (0, 160, 255), -1, cv2.LINE_AA)
            fx, fy = M @ np.array([*CANONICAL_LANDMARKS[name], 1.0])
            cv2.drawMarker(out, (int(round(fx)), int(round(fy))), (0, 100, 255), cv2.MARKER_CROSS,
                           max(4, int(round(2 * scale))), 1, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (4, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.09 * scale, (0, 0, 0), 1, cv2.LINE_AA)
    info = {"affine": M.tolist(), "rms_px": rms, "scale": scale, "peaks": peaks_px,
            "mask_px": int(msk.sum()), "landmarks_diagram_px": {k: list(v) for k, v in dst.items()}}
    return out, info
