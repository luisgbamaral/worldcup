"""Modelling machinery: temporal CV, metrics, calibration, scoreline, log."""
import datetime as dt

import numpy as np
import polars as pl

from worldcup import config, modeling as M


def test_temporal_cv_no_leakage():
    dates = np.array([dt.date(2018, 1, 1) + dt.timedelta(days=7 * i) for i in range(200)])
    for tr, va in M.walk_forward_folds(dates, n_folds=4):
        assert dates[tr].max() < dates[va].min()       # validation strictly after train


def test_phase1_excludes_wc():
    mf = pl.read_parquet(config.MATCH_FEATURES)
    data = mf.filter((~pl.col("is_2026")) & (pl.col("date") <= dt.date(2026, 6, 10)))
    assert data["is_2026"].sum() == 0


def test_rps_sanity():
    perfect = np.array([[1.0, 0.0, 0.0]])
    worst = np.array([[0.0, 0.0, 1.0]])
    y = np.array([0])                                  # true class H
    assert M.rps(perfect, y) == 0.0
    assert M.rps(worst, y) == 1.0
    mid = M.rps(np.array([[0.0, 1.0, 0.0]]), y)        # predict draw
    assert 0.0 < mid < 1.0                             # ordinal: closer than predicting away


def test_calibration_helps():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    # deliberately over-confident probabilities
    raw = np.full((400, 3), 0.1)
    raw[np.arange(400), y] = 0.8
    raw += rng.normal(0, 0.05, raw.shape)
    raw = np.clip(raw, 1e-3, None)
    raw /= raw.sum(1, keepdims=True)
    cals = M.fit_calibrators(raw, y)
    cal = M.apply_calibrators(raw, cals)
    assert M.rps(cal, y) <= M.rps(raw, y) + 1e-9


def test_arm_b_marginalizes():
    m = M.poisson_joint(1.6, 1.1)
    hda = M.hda_from_joint(m)
    assert abs(hda.sum() - 1.0) < 1e-9
    ro = M.scoreline_readoffs(m)
    assert abs(ro["p_home"] + ro["p_draw"] + ro["p_away"] - 1.0) < 1e-9
    assert 0.0 <= ro["p_over25"] <= 1.0


def test_prediction_log_append_only(tmp_path):
    path = tmp_path / "log.parquet"
    rd = dt.date(2026, 6, 20)
    df = pl.DataFrame({"match_id": [1, 2], "run_date": [rd, rd], "p_home": [0.5, 0.4]})
    M.append_prediction_log(path, df)
    # rerun with an overwriting attempt + one new row
    df2 = pl.DataFrame({"match_id": [1, 3], "run_date": [rd, rd], "p_home": [0.99, 0.3]})
    out = M.append_prediction_log(path, df2)
    assert out.height == 3                              # only match_id=3 added
    assert out.filter((pl.col("match_id") == 1))["p_home"][0] == 0.5  # never overwritten
