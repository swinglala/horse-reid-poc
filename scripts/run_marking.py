#!/usr/bin/env python
"""Phase 2 CLI: segment white markings on the Phase 1 selected face frames.

Example:
    python scripts/run_marking.py --output data/output --segmenter sam_refined --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import horse_reid  # noqa: F401
except ImportError:  # package not installed -> use the source tree
    sys.path.insert(0, str(ROOT / "src"))

from horse_reid.marking import available_marking_segmenters  # noqa: E402
from horse_reid.marking_pipeline import MarkingConfig, run_phase2  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("data/output"),
                   help="Phase 1 output dir (reads frame_scores.json, writes marking/)")
    p.add_argument("--input", type=Path, default=Path("data/input/1000018050.mp4"))
    p.add_argument("--segmenter", default="sam_refined", choices=available_marking_segmenters())
    p.add_argument("--sam-weights", type=Path, default=Path("models/mobile_sam.pt"))
    p.add_argument("--marking-weights", type=Path, default=None, help="weights for the yolo_seg segmenter")
    p.add_argument("--device", default="cpu", help="auto | cpu | cuda | mps")
    p.add_argument("--seg-model", type=Path, default=Path("models/yolo11s-seg.pt"))
    p.add_argument("--max-frames", type=int, default=None, help="process only the first N selected frames")
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = MarkingConfig(output=args.output, input=args.input, segmenter=args.segmenter, device=args.device,
                        sam_weights=args.sam_weights, seg_model=args.seg_model, marking_weights=args.marking_weights,
                        max_frames=args.max_frames, margin=args.margin, upscale=args.upscale)
    summary = run_phase2(cfg)
    print((Path(cfg.output) / "marking" / "summary.txt").read_text())
    return 0 if summary["frames_processed"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
