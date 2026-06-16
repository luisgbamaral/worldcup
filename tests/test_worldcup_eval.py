"""World-Cup evaluation: 8-year rolling window + cumulative-hits monotonicity."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import worldcup_eval as W  # noqa: E402
from worldcup import data  # noqa: E402


def _rated():
    played = data.load_results(played_only=True).with_columns(
        pl.when(pl.col("home_score") > pl.col("away_score")).then(pl.lit("H"))
          .when(pl.col("home_score") == pl.col("away_score")).then(pl.lit("D"))
          .otherwise(pl.lit("A")).alias("result"))
    return played


def test_window_is_8_years_and_excludes_tournament():
    df = _rated()
    for wc in W.WORLD_CUPS:
        start = W._wc_start(df, wc)
        tr, va = W._split(df, start.replace(year=start.year - 8), start)
        assert tr["date"].min() >= start.replace(year=start.year - 8)
        assert va["date"].max() < start          # window strictly before the tournament
        assert tr.height > 0 and va.height > 0


def _toy_eval():
    rows = []
    for model in ["LogReg", "M2"]:
        for i in range(5):
            rows.append({"model": model, "wc": 2018, "date": dt.date(2018, 6, 10 + i),
                         "match_id": i, "correct": 1 if i % 2 == 0 else 0})
    return pl.DataFrame(rows)


def test_cumulative_hits_monotonic_and_total():
    cum = W.cumulative_series(_toy_eval())
    for m in ("LogReg", "M2"):
        v = cum[m].to_numpy()
        assert (np.diff(v) >= 0).all()           # non-decreasing
        assert v[-1] == 3                          # ends at total hits (i=0,2,4)
