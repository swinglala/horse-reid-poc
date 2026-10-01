"""Auxiliary face embedding for the identity reference.

IMPORTANT: the embedding is an AUXILIARY descriptor (coarse appearance:
coat colour, head shape, lighting). The canonical white-marking mask is the
PRIMARY identity evidence; do not match horses on the embedding alone.

Methods:
  * ``clip``: OpenAI CLIP ViT-B/32 image embedding (``clip`` package), the
    L2-normalised mean of the per-crop L2-normalised embeddings.
  * ``hog_canonical`` (fallback): HOG of the canonical marking probability map
    (``skimage.feature.hog``), L2-normalised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..config import resolve_project_path

logger = logging.getLogger(__name__)

CLIP_MODEL = "ViT-B/32"
#: Local checkpoints tried before downloading (relative -> project root).
CLIP_LOCAL_CANDIDATES = ("models/weights/clip/ViT-B-32.pt", "models/clip/ViT-B-32.pt")
CLIP_DOWNLOAD_ROOT = "models/clip"

_CLIP_CACHE: dict[str, tuple] = {}


def _load_clip():
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE["model"]
    import clip  # ultralytics fork of OpenAI CLIP

    for cand in CLIP_LOCAL_CANDIDATES:
        p = resolve_project_path(cand)
        if p.exists():
            logger.info("CLIP: loading local checkpoint %s", p)
            model, preprocess = clip.load(str(p), device="cpu")
            break
    else:
        root = resolve_project_path(CLIP_DOWNLOAD_ROOT)
        logger.info("CLIP: downloading %s to %s (~340 MB, first use only)", CLIP_MODEL, root)
        model, preprocess = clip.load(CLIP_MODEL, device="cpu", download_root=str(root))
    model.eval()
    _CLIP_CACHE["model"] = (model, preprocess)
    return model, preprocess


def clip_embedding(face_crops: Sequence[np.ndarray]) -> list[float]:
    """Mean of L2-normalised CLIP image embeddings of BGR crops (L2-normalised)."""
    import torch
    from PIL import Image

    model, preprocess = _load_clip()
    imgs = [Image.fromarray(np.ascontiguousarray(c[..., ::-1] if c.ndim == 3 else c))
            for c in face_crops if c is not None and c.size > 0]
    if not imgs:
        raise ValueError("no valid face crops")
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            batch = torch.stack([preprocess(im) for im in imgs[i:i + 16]])
            f = model.encode_image(batch).float()
            feats.append(f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    v = torch.cat(feats).mean(0)
    v = v / v.norm().clamp_min(1e-12)
    return [float(x) for x in v.cpu().numpy()]


def hog_embedding(canonical_prob: np.ndarray) -> list[float]:
    """L2-normalised HOG of the canonical probability map (resized to 128x160)."""
    import cv2
    from skimage.feature import hog

    p = np.nan_to_num(np.asarray(canonical_prob, np.float32))
    p = cv2.resize(p, (128, 160), interpolation=cv2.INTER_AREA)
    h = hog(p, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    n = float(np.linalg.norm(h))
    return [float(x) for x in (h / n if n > 0 else h)]


def face_embedding(face_crops: Sequence[np.ndarray], method: str = "clip",
                   canonical_prob: Optional[np.ndarray] = None) -> tuple[list[float], str]:
    """Return ``(vector, method_str)``; falls back to ``hog_canonical`` when CLIP
    is unavailable / fails / there are no crops. Returns ``([], "none")`` if
    no fallback input is available either."""
    if method == "clip" and face_crops:
        try:
            return clip_embedding(face_crops), f"clip_{CLIP_MODEL.replace('/', '-')}_mean_of_{len(face_crops)}"
        except Exception as exc:  # noqa: BLE001 - any failure -> fallback
            logger.warning("CLIP embedding failed (%s) -> falling back to hog_canonical", exc)
    if canonical_prob is not None:
        return hog_embedding(canonical_prob), "hog_canonical"
    return [], "none"
