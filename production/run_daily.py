"""Daily production run: predict the World Cup as it unfolds and track its evolution.

Each run (after `scripts/update_raw.py` + `scripts/build_features.py` refresh the
data):
1. reads the group fixtures; played games are already folded into the data (Elo
   updates from results), unplayed ones are predicted;
2. predicts every still-unplayed group game (P(1X2), expected goals, P(over 2.5));
3. Monte-Carlo simulates the whole tournament to the champion (advance / round /
   title probabilities per team);
4. appends a **run-dated snapshot** of both outputs to append-only logs, so the
   evolution over time can be tracked, and refreshes the "latest" parquets + tables.

    python production/run_daily.py [--sims N] [--run-date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tournament as T  # noqa: E402
from worldcup import config, data, modeling as M, viz  # noqa: E402
from worldcup.clean import canon_team  # noqa: E402

TOURNEY_LOG = config.PROCESSED / "wc2026_tournament_log.parquet"
FIXTURE_LOG = config.PROCESSED / "wc2026_fixture_log.parquet"


def fixture_predictions(run_date) -> pl.DataFrame:
    """Elo–Poisson P(1X2)/goals for every still-unplayed group fixture."""
    elo, pair = T.current_elo(), T.load_pairing()
    fx = data.load_fixtures().with_columns(
        canon_team("home_team").alias("home_team"), canon_team("away_team").alias("away_team"))
    import numpy as np
    rows = []
    for r in fx.filter(pl.col("home_score").is_null()).iter_rows(named=True):
        h, a = r["home_team"], r["away_team"]
        if h not in elo or a not in elo:
            continue
        m = M.poisson_joint(pair.lam(elo[h] - elo[a]), pair.lam(elo[a] - elo[h]))
        ro = M.scoreline_readoffs(m)
        rows.append({"match_id": f'{r["match_date"]}__{h}__{a}', "run_date": run_date,
                     "date": r["match_date"], "home_team": h, "away_team": a,
                     "p_home": round(ro["p_home"], 4), "p_draw": round(ro["p_draw"], 4),
                     "p_away": round(ro["p_away"], 4),
                     "exp_home": round(ro["exp_home"], 3), "exp_away": round(ro["exp_away"], 3),
                     "p_over25": round(ro["p_over25"], 4),
                     "pred_winner": ["H", "D", "A"][int(np.argmax(
                         [ro["p_home"], ro["p_draw"], ro["p_away"]]))]})
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--run-date", type=str, default=None)
    a = ap.parse_args()
    run_date = dt.date.fromisoformat(a.run_date) if a.run_date else dt.date.today()

    tourney = T.monte_carlo(n_sims=a.sims).with_columns(run_date=pl.lit(run_date))
    fixtures = fixture_predictions(run_date)

    # latest snapshots
    tourney.write_parquet(config.PROCESSED / "wc2026_tournament.parquet")
    fixtures.write_parquet(config.PROCESSED / "wc2026_fixture_predictions.parquet")
    # append-only, run-dated history (the time series to track evolution)
    M.append_prediction_log(TOURNEY_LOG, tourney, keys=("team", "run_date"))
    M.append_prediction_log(FIXTURE_LOG, fixtures, keys=("match_id", "run_date"))

    top = tourney.head(12).select(
        "team", pl.col("elo").round(0).cast(int),
        *[pl.col(c).round(3) for c in ("p_advance", "p_QF", "p_SF", "p_final", "p_champion")])
    tex = viz.df_to_neurips_latex(
        top, label="tab:wc2026_tournament", float_format="%.3f",
        bold_best="p_champion", lower_is_better=False,
        caption=(f"2026 World Cup title-race simulation (run {run_date}, {a.sims:,} Monte-Carlo "
                 "draws, Elo–Poisson): probability of advancing, reaching each round and "
                 "winning. Re-run as results arrive to track the evolution."))
    viz.save_table(tex, "wc2026_tournament")
    print(f"run {run_date}: {fixtures.height} fixtures predicted, "
          f"{tourney.height} teams simulated ({a.sims:,} sims)")
    print(top)
    print(f"\nappended → {TOURNEY_LOG.name}, {FIXTURE_LOG.name}")


if __name__ == "__main__":
    main()
