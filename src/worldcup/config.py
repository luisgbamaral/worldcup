"""Project paths, resolved relative to the repository root.

No hard-coded absolute paths: the root is found by walking up from this
file until the ``data/`` directory is located, so the package works from
notebooks, scripts or an installed location alike.
"""
import os
from pathlib import Path


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "raw").is_dir():
            return p
    # fallback: two levels up from src/worldcup/
    return start.parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: populate os.environ without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


ROOT = _find_root(Path(__file__).resolve())
_load_dotenv(ROOT / ".env")  # e.g. TABPFN_TOKEN for the cloud foundation model

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# processed feature bases
MATCH_FEATURES = PROCESSED / "match_features.parquet"
TEAM_MATCH_FEATURES = PROCESSED / "team_match_features.parquet"
PLAYER_FEATURES = PROCESSED / "player_features.parquet"

# source-specific raw paths
RESULTS_CSV = RAW / "martj42" / "results.csv"
GOALSCORERS_CSV = RAW / "martj42" / "goalscorers.csv"
SHOOTOUTS_CSV = RAW / "martj42" / "shootouts.csv"
FORMER_NAMES_CSV = RAW / "martj42" / "former_names.csv"
WORLDCUP_JSON_DIR = RAW / "worldcup.json-master" / "worldcup.json-master"
FIXTURES_CSV = RAW / "fixtures" / "wc2026_group_stage_schedule.csv"
SQUAD_PLAYERS_CSV = RAW / "squadlist" / "wc2026_squads_players.csv"
SQUAD_COACHES_CSV = RAW / "squadlist" / "wc2026_squads_coaches.csv"

for _d in (INTERIM, PROCESSED, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)
