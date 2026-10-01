"""Phase 3 orchestration: joined Phase 1/2 outputs -> canonical identity reference.

Writes ``<output>/reference/`` (maps, visualisation, ASCII art, per-frame
warps, ``horse_reference.json``) and ``<output>/final_report.jpg``.

The reference is an aggregation of REAL observations only: nothing is
generated, in-painted or symmetrised; unobserved regions keep coverage 0.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import CanonicalMapper, CanonicalReference, find_peaks
from .io import JoinedFrame, load_joined
from .planar import CANONICAL_LANDMARKS, TEMPLATE_OUTLINE, PlanarCanonicalMapper

logger = logging.getLogger(__name__)

ASCII_LEVELS = " ░▒▓█"
ASCII_COLS, ASCII_ROWS = 32, 40
REPORT_WIDTH = 1400
PEAK_SIGMA = 4.0           # canonical px; ~landmark-alignment error between frames (975 vs 993: ~10 px)
PEAK_MIN_SUPPORT = 0.02
PEAK_MIN_DISTANCE_PX = 12
PEAK_MAX = 8


def build_mapper(name: str, **kwargs: Any) -> CanonicalMapper:
    if name == "planar":
        return PlanarCanonicalMapper(**kwargs)
    if name == "4dequine":
        from .fourdequine import FourDEquineMapper

        m = FourDEquineMapper(**kwargs)
        m.load_results()  # raises FourDEquineNotAvailable with instructions when absent
        return m
    raise ValueError(f"unknown mapper {name!r} (planar | 4dequine)")


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def _u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(np.nan_to_num(a), 0, 1) * 255).round().astype(np.uint8)


def _draw_template(img: np.ndarray, scale: float, color=(255, 255, 255), dots=True) -> None:
    pts = (np.array(TEMPLATE_OUTLINE, np.float32) * scale).round().astype(np.int32)
    cv2.polylines(img, [pts], True, color, max(1, int(round(scale))), cv2.LINE_AA)
    cv2.line(img, (int(128 * scale), 0), (int(128 * scale), img.shape[0] - 1), (90, 90, 90), 1, cv2.LINE_AA)
    if dots:
        for name, (x, y) in CANONICAL_LANDMARKS.items():
            c = (int(round(x * scale)), int(round(y * scale)))
            cv2.circle(img, c, max(3, int(3 * scale)), (0, 200, 255), -1, cv2.LINE_AA)
            cv2.putText(img, name.replace("_base", ""), (c[0] + 6, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35 * scale, (0, 200, 255), 1, cv2.LINE_AA)


def _heat_panel(values: np.ndarray, coverage: np.ndarray, scale: int, cmap: int, title: str,
                mask: Optional[np.ndarray] = None, dots: bool = True) -> np.ndarray:
    H, W = values.shape
    heat = cv2.applyColorMap(_u8(values), cmap)
    base = np.full((H, W, 3), 45, np.uint8)
    a = np.clip(np.nan_to_num(coverage), 0, 1)[..., None]
    a = np.where(a > 0, 0.35 + 0.65 * a, 0.0)
    img = (base * (1 - a) + heat * a).astype(np.uint8)
    img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    if mask is not None and mask.any():
        m = cv2.resize(mask.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)
    _draw_template(img, scale, dots=dots)
    return np.vstack([_title_bar(img.shape[1], title), img])


def _support_of(ref: CanonicalReference) -> tuple[np.ndarray, np.ndarray]:
    """``(support, n_support)`` of a reference; zeros when the mapper had no binary masks."""
    sup = ref.support if ref.support is not None else np.zeros_like(ref.prob, dtype=np.float32)
    ns = ref.n_support if ref.n_support is not None else np.zeros(ref.prob.shape, np.int32)
    return sup, ns


def _title_bar(width: int, title: str, scale: float = 0.65) -> np.ndarray:
    """34 px white title strip; the font shrinks until the title fits."""
    while scale > 0.35 and cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > width - 12:
        scale -= 0.05
    bar = np.full((34, width, 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2 if scale >= 0.6 else 1,
                cv2.LINE_AA)
    return bar


def _draw_peaks(img: np.ndarray, peaks: list[dict], scale: float, labels: bool = True) -> None:
    """Circles with radius ~ n_support; labels alternate right/left of the
    circle and are nudged vertically to avoid overlapping earlier labels."""
    placed: list[tuple[int, int, int, int]] = []
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, (0.45 if labels else 0.35)

    def overlaps(rc):
        return any(not (rc[2] < q[0] or rc[0] > q[2] or rc[3] < q[1] or rc[1] > q[3]) for q in placed)

    geo = []
    for d in peaks:  # circles first; they are obstacles for every label
        c = (int(round((d["x"] + 0.5) * scale)), int(round((d["y"] + 0.5) * scale)))
        r = int(round((3 + 3 * min(d["n_support"], 6)) * scale))
        cv2.circle(img, c, r, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, c, r, (255, 255, 0), 1 if scale < 1.5 else 2, cv2.LINE_AA)
        placed.append((c[0] - r - 1, c[1] - r - 1, c[0] + r + 1, c[1] + r + 1))
        geo.append((c, r))
    for i, (d, (c, r)) in enumerate(zip(peaks, geo)):
        txt = f"{d['id']} n={d['n_support']} s={d['support_s']:.2f}" if labels else d["id"]
        (tw, th), _ = cv2.getTextSize(txt, font, fs, 1)
        right = (i % 2 == 0)
        x0 = c[0] + r + 3 if right else c[0] - r - 3 - tw
        if x0 + tw > img.shape[1] - 2:
            x0 = c[0] - r - 3 - tw
        if x0 < 2:
            x0 = min(c[0] + r + 3, img.shape[1] - tw - 2)
        y0 = min(max(th + 4, c[1] + th // 2), img.shape[0] - 5)
        rc = (x0 - 2, y0 - th - 3, x0 + tw + 2, y0 + 4)
        step = th + 8
        if overlaps(rc):  # try the other side first
            x_alt = c[0] - r - 3 - tw if x0 > c[0] else c[0] + r + 3
            if 2 <= x_alt <= img.shape[1] - tw - 2:
                alt = (x_alt - 2, rc[1], x_alt + tw + 2, rc[3])
                if not overlaps(alt):
                    x0, rc = x_alt, alt
        for k in range(1, 16):
            if not overlaps(rc):
                break
            dy = step * ((k + 1) // 2) * (1 if k % 2 else -1)
            y1 = min(max(th + 4, y0 + dy), img.shape[0] - 5)
            rc = (x0 - 2, y1 - th - 3, x0 + tw + 2, y1 + 4)
        placed.append(rc)
        ly = rc[3] - 4
        if labels:
            cv2.rectangle(img, rc[:2], rc[2:], (0, 0, 0), -1)
        if abs(ly - th // 2 - c[1]) > r:  # leader line when nudged away from the circle
            cv2.line(img, c, (x0 if x0 > c[0] else x0 + tw, ly - th // 2), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, txt, (x0, ly), font, fs, (255, 255, 0), 1, cv2.LINE_AA)


def _support_panel(ref: CanonicalReference, peaks: list[dict], scale: int) -> np.ndarray:
    """Main panel: ``support`` (fraction of weighted observations whose accepted
    mask covers the pixel) over a dark template; colour scale 0..max(support)."""
    sup, _ = _support_of(ref)
    H, W = sup.shape
    smax = float(sup.max())
    base = np.full((H, W, 3), 18, np.uint8)
    base[ref.coverage > 0] = 42                      # observed area slightly lighter
    if smax > 0:
        v = np.clip(sup / smax, 0, 1)
        heat = cv2.applyColorMap(_u8(v), cv2.COLORMAP_HOT)
        a = np.where(sup > 0, 0.35 + 0.65 * v, 0.0)[..., None]
        img = (base * (1 - a) + heat * a).astype(np.uint8)
    else:
        img = base
    img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    _draw_template(img, scale, color=(200, 200, 200))
    _draw_peaks(img, peaks, scale, labels=True)
    title = (f"support = weighted frac. of frames with accepted mask (colour 0..{smax:.3f})" if smax > 0
             else "support: no frame accepted a marking mask")
    return np.vstack([_title_bar(img.shape[1], title, 0.5), img])


def _nsupport_panel(ref: CanonicalReference, peaks: list[dict]) -> np.ndarray:
    _, ns = _support_of(ref)
    nmax = int(ns.max())
    v = ns.astype(np.float32) / max(nmax, 1)
    p = _heat_panel(v, (ref.coverage > 0).astype(np.float32), 1, cv2.COLORMAP_VIRIDIS,
                    f"n_support (0..{nmax} frames)", None, dots=False)
    _draw_peaks(p[34:], peaks, 1.0, labels=False)
    return p


def render_canonical_marking(ref: CanonicalReference, scale: int = 2, peaks: Optional[list[dict]] = None
                             ) -> np.ndarray:
    """Main panel: ``support`` over the dark template with outline, landmarks and
    numbered peak circles. Side panels (2x2): prob (white contour = final mask),
    n_support, coverage, consistency."""
    peaks = peaks or []
    main = _support_panel(ref, peaks, scale)
    obs_area = (ref.coverage > 0).astype(np.float32)
    pp = _heat_panel(ref.prob, ref.coverage, 1, cv2.COLORMAP_INFERNO, "prob (white = mask)", ref.mask,
                     dots=False)
    _draw_peaks(pp[34:], peaks, 1.0, labels=False)
    nsp = _nsupport_panel(ref, peaks)
    cov = _heat_panel(ref.coverage, obs_area, 1, cv2.COLORMAP_VIRIDIS, "coverage", None, dots=False)
    cons = _heat_panel(ref.consistency, obs_area, 1, cv2.COLORMAP_VIRIDIS, "consistency", None, dots=False)
    gap_v = np.full((pp.shape[0], 10, 3), 255, np.uint8)
    row1 = np.hstack([pp, gap_v, nsp])
    row2 = np.hstack([cov, gap_v, cons])
    col = np.vstack([row1, np.full((10, row1.shape[1], 3), 255, np.uint8), row2])
    h = max(main.shape[0], col.shape[0])

    def pad(a: np.ndarray) -> np.ndarray:
        return np.vstack([a, np.full((h - a.shape[0], a.shape[1], 3), 255, np.uint8)])

    body = np.hstack([pad(main), np.full((h, 12, 3), 255, np.uint8), pad(col)])
    sup, ns = _support_of(ref)
    views = ", ".join(f"{k}:{v}" for k, v in sorted(ref.views.items())) or "none"
    lines = [
        (f"mapper={ref.mapper}  views={ref.view_count} ({views})  peaks={len(peaks)}", 0.55, (0, 0, 0)),
        (f"mask px={int(ref.mask.sum())}  observed px={int((ref.coverage > 0).sum())}  "
         f"max support={float(sup.max()):.3f}  max n_support={int(ns.max())}  "
         "(aggregated real observations only)", 0.5, (60, 60, 60)),
        ("peaks are candidate marking locations (mask-backed maxima of smoothed support); "
         "mask requires prob>0.5", 0.5, (60, 60, 60)),
        ("left/right = image sides of the frontal view (image-left = horse's anatomical right)", 0.45,
         (60, 60, 60)),
    ]
    info = np.full((26 * len(lines) + 10, body.shape[1], 3), 255, np.uint8)
    for i, (t, sc, colr) in enumerate(lines):
        cv2.putText(info, t, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, sc, colr, 1, cv2.LINE_AA)
    return np.vstack([body, info])


def _max_pool(v: np.ndarray, cols: int, rows: int) -> np.ndarray:
    H, W = v.shape
    ys = np.linspace(0, H, rows + 1).round().astype(int)
    xs = np.linspace(0, W, cols + 1).round().astype(int)
    out = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            blk = v[ys[r]:max(ys[r + 1], ys[r] + 1), xs[c]:max(xs[c + 1], xs[c] + 1)]
            out[r, c] = float(blk.max()) if blk.size else 0.0
    return out


def ascii_marking(prob: np.ndarray, coverage: np.ndarray, cols: int = ASCII_COLS, rows: int = ASCII_ROWS,
                  title: str = "HORSE MARKING", peaks: Optional[list[dict]] = None) -> str:
    """ASCII map of ``score = prob x coverage`` downsampled with MAX pooling (so
    small peaks survive). Peak cells that would render blank are drawn as '*'."""
    v = np.clip(np.nan_to_num(prob) * np.nan_to_num(coverage), 0, 1).astype(np.float32)
    small = _max_pool(v, cols, rows)
    n = len(ASCII_LEVELS)
    idx = np.clip((small * n).astype(int), 0, n - 1)
    grid = [[ASCII_LEVELS[i] for i in row] for row in idx]
    H, W = v.shape
    for d in peaks or []:
        r = min(rows - 1, int(d["y"] * rows / H))
        c = min(cols - 1, int(d["x"] * cols / W))
        if idx[r, c] == 0:
            grid[r][c] = "*"
    lines = [title.center(cols + 2), "+" + "-" * cols + "+"]
    lines += ["|" + "".join(row) + "|" for row in grid]
    lines.append("+" + "-" * cols + "+")
    lines.append("value = max(prob x coverage) per cell;  ' ' <.2  ░ <.4  ▒ <.6  ▓ <.8  █ >=.8;  "
                 "* = peak below ░")
    if peaks:
        lines.append("peaks (canonical x,y): " + "; ".join(
            f"{d['id']}({d['x']},{d['y']}) n={d['n_support']} s={d['support_s']:.2f}" for d in peaks))
    else:
        lines.append("peaks: none (no mask-backed local max of smoothed support)")
    return "\n".join(lines) + "\n"


def peaks_summary(peaks: list[dict]) -> str:
    """One-line ``canonical_peaks`` summary for logs / report captions."""
    if not peaks:
        return "canonical_peaks: none"
    parts = []
    for d in peaks:
        fr = ",".join(str(f) for f in d["frames"]) or "-"
        parts.append(f"{d['id']}@({d['x']},{d['y']}) s={d['support_s']:.3f} n={d['n_support']} "
                     f"frames=[{fr}] prob={d['prob']:.2f} cov={d['coverage']:.2f}")
    return f"canonical_peaks: {len(peaks)} | " + " | ".join(parts)


def _wrap(text: str, width_chars: int) -> list[str]:
    out, cur = [], ""
    for tok in text.split(" | "):
        cand = tok if not cur else cur + " | " + tok
        if len(cand) > width_chars and cur:
            out.append(cur)
            cur = tok
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def compose_final_report(output_dir: Path, canonical_png: Path, save_path: Path,
                         width: int = REPORT_WIDTH, caption_c: Optional[str] = None) -> np.ndarray:
    panels = [("A. Selected face frames", output_dir / "contact_sheet.jpg"),
              ("B. White-marking segmentation", output_dir / "marking" / "marking_sheet.jpg"),
              ("C. Canonical marking map", canonical_png)]
    parts = []
    for title, p in panels:
        head = np.full((56, width, 3), 255, np.uint8)
        cv2.putText(head, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        img = cv2.imread(str(p)) if p.exists() else None
        if img is None:
            body = np.full((80, width, 3), 235, np.uint8)
            cv2.putText(body, f"(missing: {p.name} - panel skipped)", (16, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 160), 2, cv2.LINE_AA)
        else:
            h = max(1, int(round(img.shape[0] * width / img.shape[1])))
            body = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA if img.shape[1] > width
                              else cv2.INTER_LINEAR)
        parts += [head]
        if title.startswith("C.") and caption_c:
            cap_lines = _wrap(caption_c, max(40, width // 11))
            cap = np.full((26 * len(cap_lines) + 8, width, 3), 255, np.uint8)
            for i, t in enumerate(cap_lines):
                cv2.putText(cap, t, (16, 20 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1,
                            cv2.LINE_AA)
            parts.append(cap)
        parts += [body, np.full((16, width, 3), 255, np.uint8)]
    out = np.vstack(parts)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


# --------------------------------------------------------------------------- #
# Face crops for the auxiliary embedding
# --------------------------------------------------------------------------- #
def _face_crops(output_dir: Path, video_path: Optional[Path], frames: list[JoinedFrame]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    if video_path is not None and Path(video_path).exists():
        try:
            from ..types import clip_bbox
            from ..video import VideoReader

            with VideoReader(video_path) as reader:
                imgs = reader.read_frames([j.frame for j in frames])
            for j in frames:
                f = imgs.get(j.frame)
                if f is None:
                    continue
                x1, y1, x2, y2 = clip_bbox(tuple(int(round(v)) for v in j.crop_bbox), f.shape[1], f.shape[0])
                if x2 > x1 and y2 > y1:
                    crops.append(f[y1:y2, x1:x2].copy())
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read face crops from video (%s)", exc)
    if not crops:
        for j in frames:
            img = cv2.imread(str(output_dir / "face_crops" / f"frame_{j.frame:05d}.jpg"))
            if img is not None:
                crops.append(img)
    return crops


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_reference(output_dir: str | Path, video_path: Optional[str | Path] = None, mapper_name: str = "planar",
                    horse_id: str = "unknown", compute_embedding: bool = True,
                    mapper_kwargs: Optional[dict[str, Any]] = None, selected_only: bool = False) -> dict:
    """Build ``<output>/reference/horse_reference.json`` (+ images) and ``final_report.jpg``."""
    out = Path(output_dir)
    ref_dir = out / "reference"
    pf_dir = ref_dir / "per_frame_canonical"
    pf_dir.mkdir(parents=True, exist_ok=True)
    for old in pf_dir.glob("frame_*.png"):
        old.unlink()

    mapper = build_mapper(mapper_name, **(mapper_kwargs or {}))
    joined = load_joined(out, selected_only=selected_only)
    observations, skipped, frame_meta = [], [], {}
    for j in joined:
        obs = mapper.map_frame(j.mask, j.prob, j.keypoints_px, j.yaw, j.score, frame=j.frame,
                               head_bbox_px=j.head_bbox_px)
        if obs is None:
            skipped.append(j.frame)
            continue
        observations.append(obs)
        frame_meta[j.frame] = obs.meta
        if obs.prob.ndim == 2 and obs.prob.shape[0] > 1:
            wmax = max(float(obs.weight.max()), 1e-6)
            left = cv2.cvtColor(_u8(obs.prob), cv2.COLOR_GRAY2BGR)
            right = cv2.applyColorMap(_u8(obs.weight / wmax), cv2.COLORMAP_VIRIDIS)
            for im in (left, right):
                _draw_template(im, 1.0, color=(0, 255, 0), dots=False)
            tile = np.hstack([left, np.full((left.shape[0], 4, 3), 255, np.uint8), right])
            cv2.putText(tile, f"f{j.frame} yaw {j.yaw:+.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(pf_dir / f"frame_{j.frame:05d}.png"), tile)
    logger.info("phase3: %d joined frames, %d mapped, %d skipped", len(joined), len(observations), len(skipped))
    ref = mapper.aggregate(observations)
    support, n_support = _support_of(ref)
    peaks: list[dict] = []
    if ref.prob.ndim == 2 and ref.prob.shape[0] > 1:
        peaks = find_peaks(ref.prob, ref.coverage, support, min_support=PEAK_MIN_SUPPORT,
                           min_distance_px=PEAK_MIN_DISTANCE_PX, max_peaks=PEAK_MAX, peak_sigma=PEAK_SIGMA,
                           n_support=n_support,
                           support_stack=ref.support_stack, frames=ref.frames)
    summary = peaks_summary(peaks)
    logger.info("phase3: %s", summary)

    paths = {"prob": "canonical_prob.png", "coverage": "canonical_coverage.png", "mask": "canonical_mask.png",
             "consistency": "canonical_consistency.png", "support": "canonical_support.png",
             "nsupport": "canonical_nsupport.png", "marking": "canonical_marking.png",
             "ascii": "canonical_marking.txt"}
    cv2.imwrite(str(ref_dir / paths["prob"]), _u8(ref.prob))
    cv2.imwrite(str(ref_dir / paths["coverage"]), _u8(ref.coverage))
    cv2.imwrite(str(ref_dir / paths["mask"]), (ref.mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(ref_dir / paths["consistency"]), _u8(ref.consistency))
    cv2.imwrite(str(ref_dir / paths["support"]), _u8(support))
    cv2.imwrite(str(ref_dir / paths["nsupport"]), np.clip(n_support, 0, 255).astype(np.uint8))
    if ref.prob.ndim == 2 and ref.prob.shape[0] > 1:
        vis = render_canonical_marking(ref, peaks=peaks)
        txt = ascii_marking(ref.prob, ref.coverage, peaks=peaks)
    else:  # per-vertex reference without UVs: no 2D layout available
        vis = np.full((120, 600, 3), 255, np.uint8)
        cv2.putText(vis, f"{ref.mapper}: per-vertex reference ({ref.prob.size} vertices), no UV image",
                    (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        txt = "HORSE MARKING\n(per-vertex reference; no 2D layout)\n"
    cv2.imwrite(str(ref_dir / paths["marking"]), vis)
    (ref_dir / paths["ascii"]).write_text(txt, encoding="utf-8")

    emb, emb_method = [], "skipped"
    if compute_embedding:
        used = [j for j in joined if j.frame in set(ref.frames)]
        crops = _face_crops(out, Path(video_path) if video_path else None, used)
        from .embedding import face_embedding

        emb, emb_method = face_embedding(crops, "clip", canonical_prob=ref.prob * ref.coverage)

    ok, png = cv2.imencode(".png", (ref.mask > 0).astype(np.uint8) * 255)
    mask_rel = f"reference/{paths['mask']}"
    notes = [
        "Aggregation of REAL observations only: no generated, in-painted or mirrored face; "
        "unobserved canonical pixels have coverage 0.",
        "canonical_marking_mask is the PRIMARY identity evidence; face_embedding is an auxiliary descriptor.",
        f"mapper '{ref.mapper}': " + ("2D similarity warp onto a 256x320 frontal template (stand-in for the "
                                      "3D head surface); 3/4 and profile far halves down-weighted."
                                      if ref.mapper == "planar" else "per-vertex VAREN head surface."),
        "Template sides are image sides of the frontal view: canonical image-left = horse's anatomical right.",
        f"mask = (prob > {getattr(mapper, 'prob_threshold', 0.5)}) & "
        f"(coverage > {getattr(mapper, 'min_coverage', 0.15)})",
    ]
    notes += [
        "support = sum(w * accepted_mask) / sum(w): weighted fraction of observations whose Phase-2 accepted "
        "mask covers the pixel; n_support = number of such frames (canonical_nsupport.png stores raw counts).",
        f"canonical_peaks are CANDIDATE marking locations: local maxima of support_s = gaussian(support, "
        f"sigma={PEAK_SIGMA}) >= {PEAK_MIN_SUPPORT} (min distance {PEAK_MIN_DISTANCE_PX} px, max {PEAK_MAX}), "
        f"ranked by support_s, each backed by >= 1 accepted frame mask within {2 * PEAK_SIGMA:g} px "
        "(frames/n_support = frames with mask pixels in that radius); prob/coverage are reported only. "
        "They are not part of the mask, which "
        "requires prob > 0.5.",
    ]
    if skipped:
        notes.append(f"{len(skipped)} frame(s) skipped (no usable landmarks / mask): {skipped}")
    if ref.view_count == 0:
        notes.append("WARNING: no observation could be mapped; the reference is empty.")
    result = {
        "horse_id": horse_id,
        "reference": {
            "canonical_marking_mask": mask_rel,
            "canonical_marking_mask_b64": base64.b64encode(png.tobytes()).decode("ascii") if ok else "",
            "face_embedding": emb,
            "face_embedding_method": emb_method,
            "view_count": int(ref.view_count),
            "views": ref.views,
            "frames": [int(f) for f in ref.frames],
            "mapper": ref.mapper,
            "canonical_size": [int(ref.prob.shape[1]), int(ref.prob.shape[0])],
            "files": {k: f"reference/{v}" for k, v in paths.items()},
            "canonical_peaks": peaks,
            "params": {"peak_sigma": PEAK_SIGMA, "min_support": PEAK_MIN_SUPPORT,
                       "peak_min_distance_px": PEAK_MIN_DISTANCE_PX, "max_peaks": PEAK_MAX,
                       "peak_support_radius_px": 2 * PEAK_SIGMA,
                       "prob_threshold": getattr(mapper, "prob_threshold", 0.5),
                       "min_coverage": getattr(mapper, "min_coverage", 0.15)},
            "stats": {
                "mask_px": int(ref.mask.sum()),
                "observed_px": int((ref.coverage > 0).sum()),
                "mean_consistency_observed": (round(float(ref.consistency[ref.coverage > 0].mean()), 4)
                                              if (ref.coverage > 0).any() else None),
                "n_peaks": len(peaks),
                "n_peaks_multi": sum(1 for d in peaks if d["n_support"] >= 2),
                "peak_sigma": PEAK_SIGMA,
                "min_support": PEAK_MIN_SUPPORT,
                "max_support": round(float(support.max()), 4) if support.size else 0.0,
                "max_n_support": int(n_support.max()) if n_support.size else 0,
                "joined_frames": len(joined),
                "skipped_frames": skipped,
            },
            "per_frame": {str(k): v for k, v in frame_meta.items()},
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notes": notes,
        },
    }
    with open(ref_dir / "horse_reference.json", "w") as f:
        json.dump(result, f, indent=1, default=_json_default)
    compose_final_report(out, ref_dir / paths["marking"], out / "final_report.jpg", caption_c=summary)
    logger.info("phase3: reference -> %s (views=%d, mask px=%d)", ref_dir, ref.view_count, int(ref.mask.sum()))
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)
