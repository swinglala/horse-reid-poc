#!/usr/bin/env python
"""Phase 3 CLI: canonical white-marking map + identity reference.

Reads Phase 1 (frame_scores.json) and Phase 2 (marking/) outputs from --output
and writes <output>/reference/ and <output>/final_report.jpg.

Example:
    python scripts/build_reference.py --output data/output --input data/input/1000018050.mp4 --mapper planar
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
except ImportError:  # package not installed -> use the source tree
    sys.path.insert(0, str(ROOT / "src"))

from horse_reid.canonical.fourdequine import FourDEquineNotAvailable  # noqa: E402
from horse_reid.canonical.reference import build_reference  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("data/output"))
    p.add_argument("--input", type=Path, default=Path("data/input/1000018050.mp4"),
                   help="source video (face crops for the auxiliary embedding)")
    p.add_argument("--mapper", default="planar", choices=["planar", "4dequine"])
    p.add_argument("--horse-id", default="unknown")
    p.add_argument("--no-embedding", action="store_true", help="skip the auxiliary face embedding")
    p.add_argument("--selected-only", action="store_true",
                   help="use only Phase-1 selected frames (default: every frame with a marking result)")
    p.add_argument("--fourdequine-results", type=Path, default=None, help="4DEquine refined_results.pt")
    p.add_argument("--varen-model", type=Path, default=None, help="VAREN model .pkl")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    kw = {}
    if args.mapper == "4dequine":
        kw = {"results_path": args.fourdequine_results, "varen_model_path": args.varen_model}
    try:
        res = build_reference(args.output, args.input, mapper_name=args.mapper, horse_id=args.horse_id,
                              compute_embedding=not args.no_embedding, mapper_kwargs=kw,
                              selected_only=args.selected_only)
    except FourDEquineNotAvailable as exc:
        print(f"[4dequine] {exc}", file=sys.stderr)
        return 2
    r = res["reference"]
    print(json.dumps({"horse_id": res["horse_id"], "view_count": r["view_count"], "views": r["views"],
                      "mapper": r["mapper"], "face_embedding_method": r["face_embedding_method"],
                      "embedding_dim": len(r["face_embedding"]), "stats": r["stats"]}, indent=1))
    from horse_reid.canonical.reference import peaks_summary

    print(peaks_summary(r.get("canonical_peaks") or []))
    print((Path(args.output) / "reference" / "canonical_marking.txt").read_text(encoding="utf-8"))
    return 0 if r["view_count"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
