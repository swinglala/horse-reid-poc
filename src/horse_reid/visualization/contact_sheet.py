"""Grid of labelled head crops."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

TILE = 256
LINE_H = 18
PAD = 6


def letterbox(img: np.ndarray, size: int = TILE, color: tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    """Resize keeping aspect ratio and pad to ``size x size``."""
    out = np.full((size, size, 3), color, dtype=np.uint8)
    if img is None or img.size == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def make_contact_sheet(
    items: Sequence[tuple[np.ndarray, Sequence[str]]],
    tile: int = TILE,
    columns: Optional[int] = None,
    save_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Build a grid image from ``(crop_bgr, label_lines)`` items.

    Each tile is ``tile x tile`` (letterboxed) with a label block below it.
    ``columns`` defaults to ``ceil(sqrt(n))``. Saves a JPEG if ``save_path``.
    """
    n = len(items)
    if n == 0:
        img = np.full((tile, tile, 3), 255, np.uint8)
        cv2.putText(img, "no frames", (10, tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        cols = columns or math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        max_lines = max(len(lines) for _, lines in items)
        cell_h = tile + PAD + max_lines * LINE_H + PAD
        cell_w = tile + 2 * PAD
        img = np.full((rows * cell_h, cols * cell_w, 3), 255, np.uint8)
        for i, (crop_img, lines) in enumerate(items):
            r, c = divmod(i, cols)
            x0, y0 = c * cell_w + PAD, r * cell_h + PAD
            img[y0:y0 + tile, x0:x0 + tile] = letterbox(crop_img, tile)
            for j, line in enumerate(lines):
                y = y0 + tile + (j + 1) * LINE_H
                cv2.putText(img, str(line), (x0 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return img
