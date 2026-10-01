#!/usr/bin/env python
"""Alias of run_pipeline.py (same CLI); kept because the spec names it."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
