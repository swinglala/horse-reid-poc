#!/usr/bin/env python
"""Draw a Phase 3 canonical marking reference onto a frontal head diagram.

Example:
    python scripts/draw_on_diagram.py --output data/output_mal2 --diagram docs/diagram/front.json
    -> <output>/reference/marking_on_diagram.png (+ .json with the affine, residual, peaks)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from horse_reid.canonical.diagram import draw_on_diagram, load_diagram, load_reference_maps  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True, help="pipeline output dir (contains reference/)")
    p.add_argument("--diagram", type=Path, default=ROOT / "docs/diagram/front.json",
                   help="JSON spec with the diagram image and its 5 landmarks")
    p.add_argument("--scale", type=float, default=4.0, help="upscale factor for the drawing")
    p.add_argument("--min-support", type=float, default=0.15, help="support below this is not tinted")
    p.add_argument("--landmarks", action="store_true", help="also mark the five alignment points")
    p.add_argument("--peaks", action="store_true", help="also draw candidate peaks (circle + frame count)")
    p.add_argument("--tint", action="store_true", help="also tint support >= --min-support (yellow..red)")
    p.add_argument("--no-label", action="store_true")
    p.add_argument("--save", type=Path, default=None, help="default: <output>/reference/marking_on_diagram.png")
    a = p.parse_args()

    ref_dir = a.output / "reference"
    img, lm = load_diagram(a.diagram)
    maps = load_reference_maps(ref_dir)
    label = None if a.no_label else f"{maps['horse_id']}  views={sum(maps['views'].values()) if maps['views'] else len(maps['frames'])}"
    out, info = draw_on_diagram(img, lm, maps, scale=a.scale, min_support=a.min_support,
                                show_landmarks=a.landmarks, show_peaks=a.peaks, show_tint=a.tint, label=label)
    save = a.save or (ref_dir / "marking_on_diagram.png")
    save.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save), out)
    info["diagram"] = str(a.diagram)
    save.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {save}  (affine rms {info['rms_px']:.1f} px, mask px {info['mask_px']}, peaks {len(info['peaks'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
