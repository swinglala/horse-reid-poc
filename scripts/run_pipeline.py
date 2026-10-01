#!/usr/bin/env python
"""Phase 1 CLI: select high-quality, diverse horse-head frames from a video.

Example:
    python scripts/run_pipeline.py --input data/input/1000018050.mp4 --num-frames 30 --visualize
    # re-run only track grouping + selection on an existing output (no detection pass):
    python scripts/run_pipeline.py --reselect --output data/output --num-frames 30 --min-score 0.35
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

from horse_reid.config import PipelineConfig, QualityParams  # noqa: E402
from horse_reid.face import available_head_detectors  # noqa: E402
from horse_reid.pipeline import run_phase1, run_reselect  # noqa: E402
from horse_reid.marking import available_marking_segmenters  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("data/input/1000018050.mp4"))
    p.add_argument("--output", type=Path, default=Path("data/output"))
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--save-all", action="store_true", help="also save every scored head crop")
    p.add_argument("--visualize", action="store_true", help="write annotated.mp4")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--min-gap", type=int, default=None, help="min frame distance between picks (default: auto)")
    p.add_argument("--min-score", type=float, default=0.35,
                   help="selection quality floor; fewer frames are returned rather than low-score ones")
    p.add_argument("--blur-ref-mode", default="adaptive", choices=["adaptive", "absolute"],
                   help="blur score reference: adaptive = max(20, median Laplacian variance of the run's "
                        "candidates); absolute = fixed 150 (tuned for close-ups)")
    p.add_argument("--reselect", action="store_true",
                   help="instead of Phase 1, re-run track grouping + frame selection from "
                        "<output>/detections.json and frame_scores.json (uses --input, --num-frames, "
                        "--min-gap, --min-score); detection is not re-run")
    p.add_argument("--head-detector", default="grounding_dino", choices=available_head_detectors())
    p.add_argument("--head-fallback", default="mask_top", choices=["mask_top", "bbox_top", "none"],
                   help="detector used by grounding_dino when it finds no head")
    p.add_argument("--head-stride", type=int, default=3,
                   help="run head detection + scoring only on frames with frame_idx %% N == 0 "
                        "(tracking still runs on every frame)")
    p.add_argument("--hf-cache-dir", type=Path, default=Path("models/hf/hub"),
                   help="Hugging Face hub cache for Grounding DINO (relative -> project root)")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--model", type=Path, default=Path("models/yolo11s.pt"))
    p.add_argument("--seg-model", type=Path, default=Path("models/yolo11s-seg.pt"))
    p.add_argument("--rotation", type=int, default=None, choices=[0, 90, 180, 270],
                   help="override container rotation metadata")
    p.add_argument("--max-frames", type=int, default=None, help="debug: process only the first N frames")
    p.add_argument("--phase", default="1", choices=["1", "2", "3", "all"],
                   help="1 = frame selection (default), 2 = white-marking segmentation on an existing "
                        "Phase 1 output, 3 = canonical marking reference on existing Phase 1+2 outputs, "
                        "all = phases 1, 2 and 3")
    p.add_argument("--mapper", default="planar", choices=["planar", "4dequine"],
                   help="Phase 3 canonical mapper")
    p.add_argument("--horse-id", default="unknown", help="Phase 3 horse id")
    p.add_argument("--marking-segmenter", default="sam_refined", choices=available_marking_segmenters())
    p.add_argument("--sam-weights", type=Path, default=Path("models/mobile_sam.pt"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig(
        input=args.input, output=args.output, num_frames=args.num_frames, device=args.device,
        save_all=args.save_all, visualize=args.visualize, frame_stride=args.frame_stride,
        min_frame_gap=args.min_gap, min_score=args.min_score, max_frames=args.max_frames, rotation=args.rotation,
        model=args.model, tracker=args.tracker, conf=args.conf,
        head_detector=args.head_detector, head_fallback=args.head_fallback, head_stride=args.head_stride,
        hf_cache_dir=args.hf_cache_dir, seg_model=args.seg_model,
        quality=QualityParams(blur_ref_mode=args.blur_ref_mode),
    )
    rc = 0
    if args.reselect or args.phase in ("1", "all"):
        summary = run_reselect(cfg) if args.reselect else run_phase1(cfg)
        print((Path(cfg.output) / "summary.txt").read_text(encoding="utf-8"))
        rc = 0 if summary["selected"] > 0 else 1
        if rc != 0:
            return rc
    if args.phase in ("2", "all"):
        from horse_reid.marking_pipeline import MarkingConfig, run_phase2

        mcfg = MarkingConfig(output=cfg.output, input=cfg.input, segmenter=args.marking_segmenter,
                             device=cfg.device, sam_weights=args.sam_weights, seg_model=cfg.seg_model,
                             seg_imgsz=cfg.seg_imgsz, rotation=cfg.rotation, margin=cfg.head_crop_margin)
        msum = run_phase2(mcfg)
        print((Path(cfg.output) / "marking" / "summary.txt").read_text())
        rc = 0 if msum["frames_processed"] > 0 else 1
        if rc != 0:
            return rc
    if args.phase in ("3", "all"):
        from horse_reid.canonical.fourdequine import FourDEquineNotAvailable
        from horse_reid.canonical.reference import build_reference

        try:
            ref = build_reference(cfg.output, cfg.input, mapper_name=args.mapper, horse_id=args.horse_id)
        except FourDEquineNotAvailable as exc:
            print(f"[4dequine] {exc}", file=sys.stderr)
            return 2
        print((Path(cfg.output) / "reference" / "canonical_marking.txt").read_text(encoding="utf-8"))
        rc = 0 if ref["reference"]["view_count"] > 0 else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
