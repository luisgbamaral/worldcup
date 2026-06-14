#!/usr/bin/env python
"""Build the clean Parquet bases (teams + players).

    python scripts/build_bases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.clean import main  # noqa: E402

raise SystemExit(main())
