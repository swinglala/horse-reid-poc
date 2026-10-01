"""4DEquine adapter: per-vertex canonical marking on the VAREN horse surface.

4DEquine (https://github.com/luoxue-star/4DEquine, CVPR 2026, arXiv 2603.10125)
reconstructs a 4D horse (VAREN parametric model pose/shape per frame) from a
monocular video. With it, every frame's 2D marking probability can be sampled
at the visible head vertices of one shared mesh topology -> a TRUE canonical
(per-vertex) marking map, instead of the planar 2D stand-in.

Status on this machine: NOT runnable (needs CUDA 12.1, torch 2.5.1, Python
3.10, pytorch3d, Google-Drive checkpoints and the registration-gated VAREN
model; BSL-1.1 license). Everything that does not need the external model -
projection, visibility, sampling, per-vertex aggregation and UV rasterisation -
is implemented and tested on synthetic data here. See
``docs/4dequine_integration.md`` for the integration plan.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..quality.metrics import view_score, yaw_bin
from .base import CanonicalMapper, CanonicalObservation, CanonicalReference, aggregate_weighted

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/luoxue-star/4DEquine"
VAREN_URL = "https://varen.is.tue.mpg.de"

INSTALL_HINT = (
    "4DEquine artifacts are not available. To enable the '4dequine' mapper:\n"
    f"  1. Clone {REPO_URL} on a Linux machine with an NVIDIA GPU "
    "(Python 3.10, torch 2.5.1 + CUDA 12.1, pytorch3d); CPU/macOS is unsupported.\n"
    "  2. Download the 4DEquine checkpoints (Google Drive links in its README).\n"
    f"  3. Register at {VAREN_URL} and download the VAREN horse model (.pkl).\n"
    "  4. Run 4DEquine's post_optimization_from_video.py (--stage1 --stage2) on the video and copy "
    "its refined_results.pt here; pass --fourdequine-results/--varen-model.\n"
    "LICENSE: 4DEquine is BSL-1.1 and VAREN has its own (non-commercial research) license - "
    "check both before any commercial use. See docs/4dequine_integration.md."
)


class FourDEquineNotAvailable(RuntimeError):
    """Raised when the external 4DEquine / VAREN artifacts are missing."""


# --------------------------------------------------------------------------- #
# Pure geometry (no external model needed)
# --------------------------------------------------------------------------- #
def _as_R(camera: dict) -> np.ndarray:
    R = camera.get("R")
    return np.eye(3) if R is None else np.asarray(R, np.float64).reshape(3, 3)


def project_vertices(vertices: np.ndarray, camera: dict) -> np.ndarray:
    """Project ``Nx3`` vertices to ``Nx2`` image pixels.

    Camera dicts:
      * pinhole ``{"K": 3x3, "R": 3x3, "t": 3}``: ``X_c = R X + t``,
        ``(u, v) = (K X_c)[:2] / (K X_c)[2]`` (OpenCV convention: x right,
        y down, camera looks along +z).
      * weak perspective ``{"scale": s, "tx": tx, "ty": ty}`` (optional ``"R"``):
        ``(u, v) = s * (R X)[:2] + (tx, ty)`` - depth ignored, viewing
        direction +z.
    """
    V = np.asarray(vertices, np.float64).reshape(-1, 3)
    if "K" in camera:
        K = np.asarray(camera["K"], np.float64).reshape(3, 3)
        t = np.asarray(camera.get("t", np.zeros(3)), np.float64).reshape(3)
        Xc = V @ _as_R(camera).T + t
        uvw = Xc @ K.T
        z = uvw[:, 2:3]
        z = np.where(np.abs(z) < 1e-12, 1e-12, z)
        return (uvw[:, :2] / z).astype(np.float64)
    if "scale" in camera:
        Xr = V @ _as_R(camera).T
        s = float(camera["scale"])
        return np.stack([s * Xr[:, 0] + float(camera.get("tx", 0.0)),
                         s * Xr[:, 1] + float(camera.get("ty", 0.0))], axis=1)
    raise ValueError("camera must contain 'K' (pinhole) or 'scale' (weak perspective)")


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (faces wound counter-clockwise when
    seen from outside -> outward normals)."""
    V = np.asarray(vertices, np.float64).reshape(-1, 3)
    F = np.asarray(faces, np.int64).reshape(-1, 3)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.where(n < 1e-12, 1.0, n)


def vertex_visibility(vertices: np.ndarray, faces: np.ndarray, camera: dict) -> np.ndarray:
    """Bool ``N``: vertex normal faces the camera (back-face test).

    Pinhole: ``n . (C - v) > 0`` with camera centre ``C = -R^T t``;
    weak perspective: ``n . (R^T (0, 0, -1)) > 0``.

    TODO: this ignores self-occlusion (e.g. the far cheek behind the muzzle in
    a profile view is front-facing but hidden). Replace by a z-buffer
    (pytorch3d rasteriser / depth-map test) when the real mesh is available.
    """
    V = np.asarray(vertices, np.float64).reshape(-1, 3)
    n = vertex_normals(V, faces)
    R = _as_R(camera)
    if "K" in camera:
        t = np.asarray(camera.get("t", np.zeros(3)), np.float64).reshape(3)
        C = -R.T @ t
        d = C[None, :] - V
    elif "scale" in camera:
        d = np.broadcast_to(R.T @ np.array([0.0, 0.0, -1.0]), V.shape)
    else:
        raise ValueError("camera must contain 'K' or 'scale'")
    return (n * d).sum(1) > 1e-9


def sample_mask_at_vertices(prob: np.ndarray, verts2d: np.ndarray, visible: Optional[np.ndarray] = None
                            ) -> np.ndarray:
    """Bilinear sample of ``prob`` (HxW, uint8 0..255 or float 0..1) at ``Nx2``
    pixel positions. Returns float32 ``N``; NaN for invisible / out-of-image vertices."""
    P = np.asarray(prob)
    if P.ndim == 3:
        P = P[..., 0]
    P = P.astype(np.float32) / 255.0 if P.dtype == np.uint8 else P.astype(np.float32)
    H, W = P.shape
    uv = np.asarray(verts2d, np.float64).reshape(-1, 2)
    out = np.full(len(uv), np.nan, np.float32)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    if visible is not None:
        ok &= np.asarray(visible, bool).reshape(-1)
    if not ok.any():
        return out
    x, y = uv[ok, 0], uv[ok, 1]
    x0 = np.clip(np.floor(x).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, H - 1)
    x1, y1 = np.clip(x0 + 1, 0, W - 1), np.clip(y0 + 1, 0, H - 1)
    fx, fy = x - x0, y - y0
    val = (P[y0, x0] * (1 - fx) * (1 - fy) + P[y0, x1] * fx * (1 - fy)
           + P[y1, x0] * (1 - fx) * fy + P[y1, x1] * fx * fy)
    out[ok] = val.astype(np.float32)
    return out


def aggregate_vertices(observations: Sequence[tuple[np.ndarray, np.ndarray]], min_coverage: float = 0.15,
                       prob_threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Fuse ``[(per_vertex_prob N, per_vertex_weight N), ...]`` (NaN prob = unobserved).

    Returns ``{"prob", "coverage", "consistency", "mask", "weight_sum"}`` (each ``N``).
    """
    if not observations:
        raise ValueError("no observations")
    probs = [np.asarray(p, np.float32).reshape(-1) for p, _ in observations]
    ws = [np.asarray(w, np.float32).reshape(-1) for _, w in observations]
    prob, cov, cons, mask, sw = aggregate_weighted(probs, ws, min_coverage, prob_threshold)
    return {"prob": prob, "coverage": cov, "consistency": cons, "mask": mask, "weight_sum": sw}


def vertices_to_uv_image(values: np.ndarray, uv_coords: np.ndarray, size: int | tuple[int, int] = 256,
                         flip_v: bool = True, max_fill_dist: Optional[float] = None) -> np.ndarray:
    """Rasterise per-vertex values into a UV image (scatter + nearest fill).

    ``uv_coords`` are in ``[0, 1]``; with ``flip_v`` (OpenGL convention) v=0
    is the bottom row. NaN values are not scattered. Empty pixels take the
    value of the nearest scattered pixel (only within ``max_fill_dist`` px if
    given, else NaN). Returns float32 ``HxW`` (NaN where nothing was filled).
    """
    from scipy import ndimage

    H, W = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    vals = np.asarray(values, np.float64).reshape(-1)
    uv = np.asarray(uv_coords, np.float64).reshape(-1, 2)
    img = np.full((H, W), np.nan, np.float64)
    ok = np.isfinite(vals) & np.isfinite(uv).all(1)
    if not ok.any():
        return img.astype(np.float32)
    u = np.clip(uv[ok, 0], 0, 1) * (W - 1)
    v = np.clip(uv[ok, 1], 0, 1)
    v = (1.0 - v) if flip_v else v
    cols = np.rint(u).astype(int)
    rows = np.rint(v * (H - 1)).astype(int)
    acc = np.zeros((H, W)); cnt = np.zeros((H, W))
    np.add.at(acc, (rows, cols), vals[ok])
    np.add.at(cnt, (rows, cols), 1.0)
    filled = cnt > 0
    img[filled] = acc[filled] / cnt[filled]
    dist, (ri, ci) = ndimage.distance_transform_edt(~filled, return_indices=True)
    out = img[ri, ci]
    if max_fill_dist is not None:
        out[dist > max_fill_dist] = np.nan
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Mapper
# --------------------------------------------------------------------------- #
class FourDEquineMapper(CanonicalMapper):
    """Per-vertex canonical marking via 4DEquine/VAREN (needs external artifacts).

    Observations are ``1xN`` per-vertex arrays (``N`` = number of head
    vertices) so they fit :class:`CanonicalObservation`.
    """

    name = "4dequine"

    def __init__(self, results_path: Optional[str | Path] = None, varen_model_path: Optional[str | Path] = None,
                 head_vertex_ids: Optional[Sequence[int]] = None, min_coverage: float = 0.15,
                 prob_threshold: float = 0.5) -> None:
        self.results_path = Path(results_path) if results_path else None
        self.varen_model_path = Path(varen_model_path) if varen_model_path else None
        self._head_ids = None if head_vertex_ids is None else np.asarray(head_vertex_ids, np.int64)
        self.results: Optional[Any] = None
        self.template: Optional[dict[str, Any]] = None
        self.min_coverage = min_coverage
        self.prob_threshold = prob_threshold

    # ---------------- external artifacts ---------------- #
    def load_results(self, path: Optional[str | Path] = None) -> Any:
        """Load 4DEquine's ``refined_results.pt`` (per-frame VAREN pose/shape + camera)."""
        p = Path(path) if path else self.results_path
        if p is None or not p.exists():
            raise FourDEquineNotAvailable(f"refined_results.pt not found ({p}).\n{INSTALL_HINT}")
        try:
            import torch
            self.results = torch.load(str(p), map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - surface as "not available"
            raise FourDEquineNotAvailable(f"could not load {p}: {exc}\n{INSTALL_HINT}") from exc
        keys = list(self.results.keys()) if isinstance(self.results, dict) else type(self.results).__name__
        logger.info("4dequine: loaded %s (keys: %s)", p, keys)
        self.results_path = p
        return self.results

    def load_template(self, varen_pkl: Optional[str | Path] = None) -> dict[str, Any]:
        """Load the VAREN model pickle (template vertices ``v_template``, faces ``f``)."""
        p = Path(varen_pkl) if varen_pkl else self.varen_model_path
        if p is None or not p.exists():
            raise FourDEquineNotAvailable(f"VAREN model not found ({p}); register at {VAREN_URL}.\n{INSTALL_HINT}")
        try:
            with open(p, "rb") as f:
                data = pickle.load(f, encoding="latin1")
        except Exception as exc:  # noqa: BLE001 (chumpy objects etc.)
            raise FourDEquineNotAvailable(f"could not unpickle {p} ({exc}); the VAREN pkl may need "
                                          f"chumpy / the VAREN package.\n{INSTALL_HINT}") from exc
        self.template = {"v_template": np.asarray(data["v_template"]) if "v_template" in data else None,
                         "faces": np.asarray(data["f"]) if "f" in data else None, "raw_keys": list(data)}
        return self.template

    def head_vertex_indices(self) -> np.ndarray:
        """Indices of the VAREN head vertices (from explicit ids or the model's part segmentation)."""
        if self._head_ids is not None:
            return self._head_ids
        raise FourDEquineNotAvailable(
            "head vertex indices are unknown: they must come from the VAREN part segmentation "
            "(open question: key name / part id for 'head' in the VAREN pkl) or be passed as "
            f"head_vertex_ids.\n{INSTALL_HINT}")

    # ---------------- mapping ---------------- #
    def observe(self, prob: np.ndarray, vertices: np.ndarray, faces: np.ndarray, camera: dict, yaw: float,
                quality: float, frame: int = -1, vertex_ids: Optional[np.ndarray] = None) -> CanonicalObservation:
        """Model-free core of :meth:`map_frame`: per-vertex observation from posed
        vertices + camera (mask-pixel camera). Usable with synthetic meshes."""
        vis = vertex_visibility(vertices, faces, camera)
        uv = project_vertices(vertices, camera)
        pv = sample_mask_at_vertices(prob, uv, vis)
        if vertex_ids is not None:
            pv, vis = pv[vertex_ids], vis[vertex_ids]
        w = np.where(np.isfinite(pv), float(np.clip(quality, 0, 1)) * view_score(yaw), 0.0).astype(np.float32)
        return CanonicalObservation(frame=int(frame), yaw=float(yaw), view=yaw_bin(yaw),
                                    prob=np.nan_to_num(pv, nan=0.0)[None, :].astype(np.float32),
                                    weight=w[None, :], quality=float(quality),
                                    meta={"n_visible": int(vis.sum())})

    def frame_geometry(self, frame: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(vertices, faces, camera)`` of a frame from the loaded 4DEquine results.

        Requires a VAREN forward pass (pose/shape -> vertices), which needs the
        VAREN package + model; not wired yet (open question: result keys).
        """
        if self.results is None:
            raise FourDEquineNotAvailable(f"call load_results() first.\n{INSTALL_HINT}")
        raise FourDEquineNotAvailable(
            "VAREN forward pass is not wired: refined_results.pt keys and the VAREN layer API must be "
            f"confirmed on a GPU machine (docs/4dequine_integration.md, open questions).\n{INSTALL_HINT}")

    def map_frame(self, mask, prob, keypoints_px, yaw, quality, frame=-1, head_bbox_px=None):
        if self.results is None:
            raise FourDEquineNotAvailable(f"4DEquine results not loaded.\n{INSTALL_HINT}")
        vertices, faces, camera = self.frame_geometry(frame)
        p = prob if prob is not None else mask
        return self.observe(p, vertices, faces, camera, yaw, quality, frame, self.head_vertex_indices())

    def aggregate(self, observations: Sequence[CanonicalObservation]) -> CanonicalReference:
        obs = [o for o in observations if o is not None and float(np.max(o.weight)) > 0]
        if not obs:
            raise ValueError("no observations with positive weight")
        prob, cov, cons, mask, sw = aggregate_weighted([o.prob for o in obs], [o.weight for o in obs],
                                                       self.min_coverage, self.prob_threshold)
        views: dict[str, int] = {}
        for o in obs:
            views[o.view] = views.get(o.view, 0) + 1
        return CanonicalReference(prob=prob, coverage=cov, mask=mask, consistency=cons, view_count=len(obs),
                                  views=views, frames=[o.frame for o in obs], mapper=self.name, weight_sum=sw)
