"""AI-Hub "말(馬) 부위 식별 및 이상상태 진단 이미지 데이터" (dataSetSn=71707) -> YOLO segmentation.

The dataset is NOT available in this project (login + approval, Korean
nationals only); this converter is written against the schema documented on
the official dataset page and is deliberately tolerant:

* the ``horse_object`` block and its ``polygon`` list are found by nested key
  lookup (any depth);
* polygon coordinates may be ``[[x, y], ...]``, a flat ``[x1, y1, x2, y2, ...]``
  list or ``[{"x": .., "y": ..}, ...]``;
* image size from ``image_resolution`` (dict ``width``/``height``, ``[w, h]``
  or ``"WxH"``), else read from the image file;
* missing fields / images are logged and the file is skipped.

The exact polygon ``category`` strings are not published; run
:func:`discover_categories` (``scripts/convert_aihub.py --dry-run``) first and
build ``category_map`` from what it prints.

# TODO(phase2-upgrade): verify the field names against real files once access is granted.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------------------- #
# Tolerant JSON helpers
# --------------------------------------------------------------------------- #
def find_key(obj: Any, key: str) -> Any:
    """First value stored under ``key`` anywhere in a nested dict/list (DFS), else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def load_annotation(path: str | Path) -> Optional[dict[str, Any]]:
    """Load a JSON annotation (utf-8 / utf-8-sig / cp949); None on failure."""
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            with open(p, encoding=enc) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {"_root": data}
        except UnicodeDecodeError:
            continue
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Cannot read %s: %s", p, e)
            return None
    logger.warning("Cannot decode %s", p)
    return None


def parse_coords(coor: Any) -> Optional[list[tuple[float, float]]]:
    """Normalise polygon coordinates to ``[(x, y), ...]`` (>= 3 points) or None."""
    if coor is None:
        return None
    if isinstance(coor, str):
        try:
            coor = json.loads(coor)
        except json.JSONDecodeError:
            return None
    pts: list[tuple[float, float]] = []
    try:
        if isinstance(coor, list) and coor and all(isinstance(v, (int, float)) for v in coor):
            if len(coor) % 2:
                return None
            pts = [(float(coor[i]), float(coor[i + 1])) for i in range(0, len(coor), 2)]
        elif isinstance(coor, list):
            # Possibly one extra nesting level: [[[x, y], ...]]
            if len(coor) == 1 and isinstance(coor[0], list) and coor[0] and isinstance(coor[0][0], (list, dict)):
                coor = coor[0]
            for p in coor:
                if isinstance(p, dict):
                    pts.append((float(p.get("x", p.get("X"))), float(p.get("y", p.get("Y")))))
                else:
                    pts.append((float(p[0]), float(p[1])))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return pts if len(pts) >= 3 else None


def iter_polygons(ann: dict[str, Any]) -> Iterator[tuple[str, list[tuple[float, float]]]]:
    """Yield ``(category, points)`` for every polygon of the annotation's ``horse_object``."""
    ho = find_key(ann, "horse_object")
    polys = find_key(ho if ho is not None else ann, "polygon")
    if polys is None:
        return
    if isinstance(polys, dict):
        polys = [polys]
    for item in polys if isinstance(polys, list) else []:
        if not isinstance(item, dict):
            continue
        cat = item.get("category", item.get("class", item.get("label")))
        pts = parse_coords(item.get("coor", item.get("coords", item.get("points"))))
        if cat is None or pts is None:
            logger.debug("Skipping malformed polygon %r", {k: item.get(k) for k in list(item)[:3]})
            continue
        yield str(cat), pts


def image_size(ann: dict[str, Any], image_path: Optional[Path]) -> Optional[tuple[int, int]]:
    """``(width, height)`` from ``image_resolution`` or the image file."""
    res = find_key(ann, "image_resolution")
    try:
        if isinstance(res, dict):
            w = res.get("width", res.get("Width", res.get("w")))
            h = res.get("height", res.get("Height", res.get("h")))
            if w and h:
                return int(float(w)), int(float(h))
        elif isinstance(res, (list, tuple)) and len(res) >= 2:
            return int(float(res[0])), int(float(res[1]))
        elif isinstance(res, str):
            for sep in ("x", "X", "*", ",", " "):
                if sep in res:
                    a, b = res.split(sep)[:2]
                    return int(float(a)), int(float(b))
    except (TypeError, ValueError):
        pass
    if image_path is not None and image_path.exists():
        import cv2

        img = cv2.imread(str(image_path))
        if img is not None:
            return img.shape[1], img.shape[0]
    return None


# --------------------------------------------------------------------------- #
def _json_files(json_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(json_dir).rglob("*") if p.suffix.lower() == ".json")


def discover_categories(json_dir: str | Path) -> Counter:
    """Count polygon categories over all JSON files below ``json_dir``."""
    c: Counter = Counter()
    for jp in _json_files(json_dir):
        ann = load_annotation(jp)
        if ann is None:
            continue
        for cat, _ in iter_polygons(ann):
            c[cat] += 1
    return c


def _index_images(image_dir: str | Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in Path(image_dir).rglob("*"):
        if p.suffix in IMG_EXTS:
            idx.setdefault(p.name, p)
            idx.setdefault(p.stem, p)
    return idx


def _place_image(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def convert_to_yolo_seg(json_dir: str | Path, image_dir: str | Path, out_dir: str | Path,
                        category_map: dict[str, int], split: float = 0.9, seed: int = 0,
                        copy_images: bool = False) -> dict[str, Any]:
    """Write a YOLO segmentation dataset.

    Layout: ``out_dir/images/{train,val}/<img>``, ``out_dir/labels/{train,val}/<stem>.txt``
    (one line per polygon: ``class x1 y1 x2 y2 ...`` normalised to [0, 1]) and
    ``out_dir/data.yaml``. Only categories in ``category_map`` are written;
    files without any mapped polygon are skipped. Images are symlinked unless
    ``copy_images``. Returns conversion statistics.
    """
    out = Path(out_dir)
    images = _index_images(image_dir)
    files = _json_files(json_dir)
    rng = random.Random(seed)
    order = list(range(len(files)))
    rng.shuffle(order)
    n_train = int(round(split * len(files)))
    is_train = {files[i]: rank < n_train for rank, i in enumerate(order)}
    if len(files) == 1:
        is_train[files[0]] = True

    stats: Counter = Counter()
    for jp in files:
        ann = load_annotation(jp)
        if ann is None:
            stats["bad_json"] += 1
            continue
        fname = find_key(ann, "file_name")
        img_path: Optional[Path] = None
        if fname:
            fn = Path(str(fname).replace("\\", "/"))
            img_path = images.get(fn.name) or images.get(fn.stem)
        if img_path is None:
            img_path = images.get(jp.stem)
        if img_path is None:
            logger.warning("%s: image %r not found under %s; skipped", jp.name, fname, image_dir)
            stats["missing_image"] += 1
            continue
        size = image_size(ann, img_path)
        if size is None or size[0] <= 0 or size[1] <= 0:
            logger.warning("%s: unknown image size; skipped", jp.name)
            stats["missing_size"] += 1
            continue
        w, h = size
        lines = []
        for cat, pts in iter_polygons(ann):
            if cat not in category_map:
                stats["unmapped_polygons"] += 1
                continue
            coords = []
            for x, y in pts:
                coords += [min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)]
            lines.append(f"{int(category_map[cat])} " + " ".join(f"{v:.6f}" for v in coords))
        if not lines:
            stats["no_mapped_polygon"] += 1
            continue
        sub = "train" if is_train[jp] else "val"
        _place_image(img_path, out / "images" / sub / img_path.name, copy_images)
        lbl = out / "labels" / sub / f"{img_path.stem}.txt"
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("\n".join(lines) + "\n")
        stats[f"images_{sub}"] += 1
        stats["polygons"] += len(lines)

    names: dict[int, str] = {}
    for cat, cid in sorted(category_map.items(), key=lambda kv: kv[1]):
        names[int(cid)] = f"{names[int(cid)]}|{cat}" if int(cid) in names else cat
    out.mkdir(parents=True, exist_ok=True)
    yaml_lines = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {cid}: {json.dumps(n, ensure_ascii=False)}" for cid, n in sorted(names.items())]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    result = dict(stats)
    result["json_files"] = len(files)
    logger.info("AI-Hub conversion: %s", result)
    return result
