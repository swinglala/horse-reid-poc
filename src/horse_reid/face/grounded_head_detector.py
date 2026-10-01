"""Open-vocabulary horse-head detector based on Grounding DINO.

The model (``IDEA-Research/grounding-dino-tiny``, Apache-2.0) is prompted with
``"horse head. horse ear. horse eye. horse nose."`` on the (slightly expanded)
horse crop. The best ``head`` box becomes the head detection; ``ear`` /
``eye`` / ``nose`` boxes whose centres fall inside the head box become coarse
keypoints (used for landmark-based yaw and a face-visibility score).

Observed behaviour (CPU, torch 2.2.2, transformers 4.49):

* ~3 s per crop on an Intel Mac CPU with ``shortest_edge=480`` (vs ~9 s at the
  processor default of 800);
* the model also returns a *whole-horse* box labelled "horse head" (score
  ~0.3) -> head candidates covering >= 50% of the crop are rejected;
* labels are noisy ("horse horse eye", "ear", "head") -> matched by substring;
* rear views still get a head box, but eye/nose confidences stay < 0.3.

``transformers`` / ``torch`` are imported lazily so the rest of the package
works without them; the weights are loaded on the first :meth:`detect` call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..types import BBox, HeadDetection, bbox_area, clip_bbox, expand_bbox
from .head_detector import HorseHeadDetector

logger = logging.getLogger(__name__)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "horse head. horse ear. horse eye. horse nose."
PARTS: tuple[str, ...] = ("ear", "eye", "nose")
MAX_PER_PART = {"ear": 2, "eye": 2, "nose": 1}

# (label, score, (x1, y1, x2, y2)) in crop coordinates (floats)
_Item = tuple[str, float, tuple[float, float, float, float]]


def _iou_f(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_head(items: list[_Item], crop_w: int, crop_h: int,
              max_area_fraction: float = 0.5) -> Optional[tuple[float, tuple[float, float, float, float]]]:
    """Highest-scoring "head" candidate smaller than ``max_area_fraction`` of the crop
    (larger boxes are the whole horse mislabelled as a head)."""
    crop_area = float(crop_w * crop_h)
    heads = [(s, b) for lbl, s, b in items
             if "head" in lbl and (b[2] - b[0]) * (b[3] - b[1]) < max_area_fraction * crop_area]
    if not heads:
        return None
    return max(heads, key=lambda t: t[0])


def collect_parts(items: list[_Item], head_box: tuple[float, float, float, float],
                  margin: float = 0.15, dedupe_iou: float = 0.3) -> dict[str, list[list[float]]]:
    """Part keypoints ``{"ear"|"eye"|"nose": [[cx, cy, score], ...]}`` whose box
    centres lie inside ``head_box`` expanded by ``margin``. Overlapping boxes of the
    same part (IoU > ``dedupe_iou``) are suppressed, keeping the highest score;
    at most 2 ears, 2 eyes and 1 nose are kept. Coordinates as in ``items``."""
    x1, y1, x2, y2 = head_box
    w, h = x2 - x1, y2 - y1
    ex1, ey1, ex2, ey2 = x1 - margin * w, y1 - margin * h, x2 + margin * w, y2 + margin * h
    cands: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {k: [] for k in PARTS}
    for lbl, s, b in items:
        if "head" in lbl:
            continue
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        if not (ex1 <= cx <= ex2 and ey1 <= cy <= ey2):
            continue
        for part in PARTS:
            if part in lbl:
                cands[part].append((s, b))
    out: dict[str, list[list[float]]] = {}
    for part, lst in cands.items():
        lst.sort(key=lambda t: -t[0])
        keep: list[tuple[float, tuple[float, float, float, float]]] = []
        for s, b in lst:
            if all(_iou_f(b, kb) <= dedupe_iou for _, kb in keep):
                keep.append((s, b))
        out[part] = [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0, float(s)]
                     for s, b in keep[:MAX_PER_PART[part]]]
    return out


class GroundingDinoHeadDetector(HorseHeadDetector):
    """Horse head + ear/eye/nose detection with Grounding DINO (zero-shot).

    Args:
        device: "cpu", "cuda" or "mps". On MPS the first inference is tried; on
            any error the model is moved to CPU permanently (untested backend).
        shortest_edge / longest_edge: processor resize (480/800 -> ~3 s/crop on CPU).
        box_threshold / text_threshold: post-processing thresholds.
        crop_margin: horse bbox expansion before cropping.
        cache_dir: HF hub cache directory (``.../models/hf/hub``); downloaded there if missing.
        fallback: detector used when no head candidate is found (method gets
            the suffix ``"(gd_fallback)"``); ``None`` -> return ``None``.
        max_head_area_fraction: reject head boxes covering >= this fraction of the crop.
    """

    name = "grounding_dino"

    def __init__(
        self,
        device: str = "cpu",
        shortest_edge: int = 480,
        longest_edge: int = 800,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        crop_margin: float = 0.10,
        cache_dir: Optional[str | Path] = None,
        fallback: Optional[HorseHeadDetector] = None,
        max_head_area_fraction: float = 0.5,
        part_margin: float = 0.15,
    ) -> None:
        self.device = str(device)
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.crop_margin = crop_margin
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.fallback = fallback
        self.max_head_area_fraction = max_head_area_fraction
        self.part_margin = part_margin
        self.processor: Any = None
        self.model: Any = None
        self._verified_device = False  # True after the first successful inference
        self.last_items: list[_Item] = []

    # ------------------------------------------------------------------ #
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _from_pretrained(self, cls: Any) -> Any:
        kw: dict[str, Any] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            kw["cache_dir"] = str(self.cache_dir)
        try:  # prefer the local cache (no network round-trip)
            return cls.from_pretrained(MODEL_ID, local_files_only=True, **kw)
        except Exception:  # noqa: BLE001 - not cached yet -> download
            logger.info("Downloading %s to %s", MODEL_ID, kw.get("cache_dir", "default HF cache"))
            return cls.from_pretrained(MODEL_ID, **kw)

    def load(self) -> None:
        """Load processor and model (idempotent)."""
        if self.model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = self._from_pretrained(AutoProcessor)
        self.processor.image_processor.size = {"shortest_edge": self.shortest_edge,
                                               "longest_edge": self.longest_edge}
        self.model = self._from_pretrained(AutoModelForZeroShotObjectDetection).eval()
        try:
            self.model.to(self.device)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not move Grounding DINO to %s (%s); using cpu", self.device, e)
            self.device = "cpu"
            self.model.to("cpu")
        logger.info("Loaded %s on %s (cache %s)", MODEL_ID, self.device, self.cache_dir)

    def _infer(self, crop_bgr: np.ndarray) -> list[_Item]:
        import torch
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inp = self.processor(images=pil, text=PROMPT, return_tensors="pt").to(self.device)
            out = self.model(**inp)
            res = self.processor.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[pil.size[::-1]])[0]
        labels = res["text_labels"] if "text_labels" in res else res["labels"]
        items: list[_Item] = []
        for b, s, lbl in zip(res["boxes"], res["scores"], labels):
            bx = [float(v) for v in b.detach().cpu().tolist()]
            items.append((str(lbl), float(s), (bx[0], bx[1], bx[2], bx[3])))
        return items

    def run(self, crop_bgr: np.ndarray) -> list[_Item]:
        """Raw detections ``(label, score, (x1, y1, x2, y2))`` in crop coordinates."""
        self.load()
        if self._verified_device or self.device == "cpu":
            items = self._infer(crop_bgr)
        else:
            try:
                items = self._infer(crop_bgr)
            except Exception as e:  # noqa: BLE001 - untested backend (MPS)
                logger.warning("Grounding DINO failed on %s (%s); moving to cpu permanently", self.device, e)
                self.device = "cpu"
                self.model.to("cpu")
                items = self._infer(crop_bgr)
        self._verified_device = True
        return items

    # ------------------------------------------------------------------ #
    def _fallback(self, frame: np.ndarray, horse_bbox: BBox, horse_conf: float) -> Optional[HeadDetection]:
        if self.fallback is None:
            return None
        det = self.fallback.detect(frame, horse_bbox, horse_conf)
        if det is not None:
            det.method = f"{det.method}(gd_fallback)"
        return det

    def detect(self, frame: np.ndarray, horse_bbox: BBox, horse_conf: float = 1.0) -> Optional[HeadDetection]:
        h_img, w_img = frame.shape[:2]
        hb = clip_bbox(horse_bbox, w_img, h_img)
        self.last_items = []
        if bbox_area(hb) <= 0:
            return None
        cb = expand_bbox(hb, self.crop_margin, w_img, h_img)
        crop_img = np.ascontiguousarray(frame[cb[1]:cb[3], cb[0]:cb[2]])
        ch, cw = crop_img.shape[:2]
        if ch < 8 or cw < 8:
            return self._fallback(frame, horse_bbox, horse_conf)
        items = self.run(crop_img)
        ox, oy = float(cb[0]), float(cb[1])
        # shift everything to frame coordinates
        items = [(lbl, s, (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)) for lbl, s, b in items]
        self.last_items = items
        best = pick_head(items, cw, ch, self.max_head_area_fraction)
        if best is None:
            return self._fallback(frame, horse_bbox, horse_conf)
        score, hbox = best
        box = clip_bbox((int(round(hbox[0])), int(round(hbox[1])), int(round(hbox[2])), int(round(hbox[3]))),
                        w_img, h_img)
        if bbox_area(box) <= 0:
            return self._fallback(frame, horse_bbox, horse_conf)
        kps = collect_parts(items, hbox, margin=self.part_margin)
        return HeadDetection(bbox=box, confidence=float(score), method=self.name, keypoints=kps)


__all__ = ["GroundingDinoHeadDetector", "MODEL_ID", "PROMPT", "pick_head", "collect_parts"]
