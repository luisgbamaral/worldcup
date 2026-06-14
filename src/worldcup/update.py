"""Refresh the raw datasets from their canonical sources.

Run daily::

    python scripts/update_raw.py        # or: python -m worldcup.update

All sources download from GitHub raw URLs / the GitHub API (no auth). Elo is
not downloaded — it is computed from results in :mod:`worldcup.elo`.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from . import config

GH_RAW = "https://raw.githubusercontent.com"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "worldcup-updater"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _save(data: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def update_martj42() -> int:
    """martj42/international_results — 4 CSVs at the repo root."""
    base = f"{GH_RAW}/martj42/international_results/master"
    files = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]
    for f in files:
        _save(_get(f"{base}/{f}"), config.RAW / "martj42" / f)
    return len(files)


def update_statsbomb() -> int:
    """statsbomb/open-data — only the competition catalogue."""
    url = f"{GH_RAW}/statsbomb/open-data/master/data/competitions.json"
    _save(_get(url), config.RAW / "statsbomb" / "competitions.json")
    return 1


def update_openfootball() -> int:
    """openfootball/worldcup.json — every .json across all editions."""
    api = "https://api.github.com/repos/openfootball/worldcup.json/git/trees/master?recursive=1"
    tree = json.loads(_get(api))["tree"]
    n = 0
    for node in tree:
        path = node["path"]
        if node["type"] == "blob" and path.endswith(".json"):
            _save(_get(f"{GH_RAW}/openfootball/worldcup.json/master/{path}"),
                  config.WORLDCUP_JSON_DIR / path)
            n += 1
    return n


SOURCES = {
    "martj42": update_martj42,
    "statsbomb": update_statsbomb,
    "openfootball": update_openfootball,
}


def main() -> int:
    """Update every source; never let one failure abort the rest."""
    failed = 0
    for name, fn in SOURCES.items():
        try:
            print(f"[ok]   {name}: {fn()} file(s)")
        except Exception as exc:  # noqa: BLE001 — report and continue
            failed += 1
            print(f"[FAIL] {name}: {exc}")
    print("done." if not failed else f"done with {failed} failure(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
