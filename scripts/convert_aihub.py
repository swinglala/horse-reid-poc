#!/usr/bin/env python
"""Convert the AI-Hub horse body-part dataset (dataSetSn=71707) to YOLO segmentation format.

The dataset is not included (login + approval required, see docs/aihub_dataset.md).

Examples:
    # 1) list polygon categories present in the label JSONs
    python scripts/convert_aihub.py --json-dir /data/aihub/labels --dry-run
    # 2) convert the white-marking polygons
    python scripts/convert_aihub.py --json-dir /data/aihub/labels --image-dir /data/aihub/images \\
        --out-dir data/aihub_yolo --category-map '{"head_whitespot":0}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import horse_reid  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

from horse_reid.marking.aihub import convert_to_yolo_seg, discover_categories  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("data/aihub_yolo"))
    p.add_argument("--category-map", default=None, help='JSON dict category -> class id, e.g. \'{"head_whitespot":0}\'')
    p.add_argument("--split", type=float, default=0.9, help="train fraction")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    p.add_argument("--dry-run", action="store_true", help="only print the polygon categories")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dry_run or args.category_map is None:
        cats = discover_categories(args.json_dir)
        print(f"{sum(cats.values())} polygons, {len(cats)} categories:")
        for c, n in cats.most_common():
            print(f"  {n:8d}  {c}")
        if not args.dry_run:
            print("No --category-map given; nothing converted.")
        return 0
    if args.image_dir is None:
        p.error("--image-dir is required for conversion")
    cmap = json.loads(args.category_map)
    stats = convert_to_yolo_seg(args.json_dir, args.image_dir, args.out_dir, cmap, split=args.split,
                                copy_images=args.copy)
    print(json.dumps(stats, indent=1))
    return 0 if stats.get("polygons", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
