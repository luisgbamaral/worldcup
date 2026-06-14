#!/usr/bin/env python
"""Daily entry point: refresh every raw dataset in one run.

    python scripts/update_raw.py

Works without installing the package (adds ``src/`` to the path).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.update import main  # noqa: E402

raise SystemExit(main())
