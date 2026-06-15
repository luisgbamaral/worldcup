#!/usr/bin/env python
"""Build the model-ready feature parquets (match / team_match / player).

    python scripts/build_features.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldcup.features import main  # noqa: E402

raise SystemExit(main())
