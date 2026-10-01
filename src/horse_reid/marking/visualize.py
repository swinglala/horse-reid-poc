"""Visualisation helpers for Phase 2 marking results."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .base import MarkingResult

GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 230, 255)


def mask_overlay(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = GREEN, alpha: float = 0.45,
                 contour: bool = True) -> np.ndarray:
    """Blend ``color`` into ``img`` where ``mask`` and draw its outline."""
    out = img.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (out[m] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
        if contour:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, color, 1, cv2.LINE_AA)
    return out


def outline(img: np.ndarray, mask: Optional[np.ndarray], color: tuple[int, int, int], thickness: int = 1) -> np.ndarray:
    if mask is not None and mask.any():
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, color, thickness, cv2.LINE_AA)
    return img


def _title(panel: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((22, panel.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def four_panel(crop: np.ndarray, res: MarkingResult, height: int = 360) -> np.ndarray:
    """face crop | face-mask overlay | candidate prob heatmap | final mask (green) + excluded (red)."""
    fm = res.face_mask.astype(bool)
    p1 = crop.copy()
    p2 = (crop * 0.35).astype(np.uint8)
    p2[fm] = crop[fm]
    outline(p2, fm, YELLOW, 2)
    cand = res.debug.get("candidate_prob", res.prob)
    heat = cv2.applyColorMap(np.clip(cand * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    p3 = cv2.addWeighted(crop, 0.35, heat, 0.65, 0)
    outline(p3, fm, (255, 255, 255), 1)
    p4 = mask_overlay(crop, res.mask, GREEN, 0.5)
    outline(p4, res.debug.get("excluded_mask"), RED, 2)
    titles = ["face crop", "face mask", "candidate prob", f"final ({res.method})"]
    panels = []
    for p, t in zip([p1, p2, p3, p4], titles):
        s = height / p.shape[0]
        p = cv2.resize(p, (max(1, int(round(p.shape[1] * s))), height),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        panels.append(_title(p, t))
    sep = np.full((panels[0].shape[0], 4, 3), 255, np.uint8)
    row = []
    for i, p in enumerate(panels):
        row += [p] + ([sep] if i < len(panels) - 1 else [])
    return np.hstack(row)
