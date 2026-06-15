"""Three pre-match rating systems with a uniform interface (1X2 baselines).

* ``elo_classic`` — plain result-based Elo: one rating/team, constant ``K``, additive
  home advantage, **no** goal-difference multiplier and **no** tournament weighting
  (Elo 1978; Hvattum & Arntzen 2010). The deliberate "plain Elo" contrast.
* ``elo_world``   — the eloratings.net variant (:mod:`worldcup.elo`): goal-difference
  multiplier + tournament-weighted ``K`` + home_adv 100. Kept as-is, exposed here.
* ``pi_rating``   — Constantinou & Fenton (2013), JQAS 9(1):37–50: **two** ratings per
  team (home ``R^H`` and away ``R^A``), goal-difference driven with diminishing returns
  on large margins; cross-updates the opposite rating by ``γ``.

Every ``rate_*`` returns the played matches plus pre-match columns
``home_rating_pre`` / ``away_rating_pre`` (leakage-safe: the rating *before* the match).
:class:`Readout` then maps the pre-match rating difference to calibrated ``P(H/D/A)``
(multinomial logistic on the difference + isotonic calibration on a temporal slice).
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from . import data, elo, modeling as M

ELO_CLASSIC_K = 20.0
ELO_CLASSIC_HOME_ADV = 65.0
PI_LAMBDA, PI_GAMMA, PI_C = 0.06, 0.5, 3.0


def _played(played):
    return (played if played is not None else data.load_results(played_only=True)).sort("date")


def _outcome(home_score, away_score) -> np.ndarray:
    """0=H, 1=D, 2=A from scores."""
    h, a = np.asarray(home_score), np.asarray(away_score)
    return np.where(h > a, 0, np.where(h == a, 1, 2))


# --------------------------------------------------------------------------- #
# A1. classic result-based Elo
# --------------------------------------------------------------------------- #
def rate_classic(played: pl.DataFrame | None = None, k: float = ELO_CLASSIC_K,
                 home_adv: float = ELO_CLASSIC_HOME_ADV, init: float = 1500.0) -> pl.DataFrame:
    df = _played(played)
    r: dict[str, float] = {}
    hp, ap = [], []
    for m in df.iter_rows(named=True):
        h, a = m["home_team"], m["away_team"]
        rh, ra = r.get(h, init), r.get(a, init)
        hp.append(rh); ap.append(ra)
        adv = 0.0 if m["neutral"] else home_adv
        e = 1.0 / (1.0 + 10 ** (-((rh + adv) - ra) / 400))
        gd = m["home_score"] - m["away_score"]
        s = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        delta = k * (s - e)            # equal-and-opposite
        r[h], r[a] = rh + delta, ra - delta
    return df.with_columns(home_rating_pre=pl.Series(hp, dtype=pl.Float64),
                           away_rating_pre=pl.Series(ap, dtype=pl.Float64))


# --------------------------------------------------------------------------- #
# A2. eloratings.net variant (reuse worldcup.elo)
# --------------------------------------------------------------------------- #
def rate_world(played: pl.DataFrame | None = None) -> pl.DataFrame:
    return elo.rate_matches(played).rename(
        {"home_elo_pre": "home_rating_pre", "away_elo_pre": "away_rating_pre"})


# --------------------------------------------------------------------------- #
# A3. pi-ratings (Constantinou & Fenton 2013)
# --------------------------------------------------------------------------- #
def rate_pi(played: pl.DataFrame | None = None, lam: float = PI_LAMBDA,
            gamma: float = PI_GAMMA, c: float = PI_C) -> pl.DataFrame:
    df = _played(played)
    H: dict[str, float] = {}   # home ratings
    A: dict[str, float] = {}   # away ratings
    g = lambda R: math.copysign(10 ** (abs(R) / c) - 1, R)
    hp, ap = [], []
    for m in df.iter_rows(named=True):
        h, a = m["home_team"], m["away_team"]
        rh, ra = H.get(h, 0.0), A.get(a, 0.0)
        hp.append(rh); ap.append(ra)
        err = (m["home_score"] - m["away_score"]) - (g(rh) - g(ra))
        psi = c * math.log10(1 + abs(err))
        d = lam * psi * (1 if err > 0 else -1 if err < 0 else 0)
        H[h] = rh + d                       # home's home rating
        A[a] = ra - d                       # away's away rating (opposite)
        A[h] = A.get(h, 0.0) + gamma * d    # cross-updates
        H[a] = H.get(a, 0.0) - gamma * d
    return df.with_columns(home_rating_pre=pl.Series(hp, dtype=pl.Float64),
                           away_rating_pre=pl.Series(ap, dtype=pl.Float64))


RATERS = {"elo_classic": rate_classic, "elo_world": rate_world, "pi_rating": rate_pi}


# --------------------------------------------------------------------------- #
# readout: rating difference -> calibrated P(H/D/A)
# --------------------------------------------------------------------------- #
class Readout:
    """Multinomial logistic on the pre-match rating difference + isotonic calibration."""

    @staticmethod
    def _diff(df):
        return (df["home_rating_pre"] - df["away_rating_pre"]).to_numpy().reshape(-1, 1)

    def fit(self, rated_train: pl.DataFrame, rated_val: pl.DataFrame) -> "Readout":
        ytr = _outcome(rated_train["home_score"], rated_train["away_score"])
        self.clf = LogisticRegression(max_iter=1000).fit(self._diff(rated_train), ytr)
        pv = self._raw(rated_val)
        yval = _outcome(rated_val["home_score"], rated_val["away_score"])
        self.cals = M.fit_calibrators(pv, yval)
        return self

    def _raw(self, df) -> np.ndarray:
        p = self.clf.predict_proba(self._diff(df))
        return p[:, [list(self.clf.classes_).index(i) for i in range(3)]]

    def predict(self, df) -> np.ndarray:
        return M.apply_calibrators(self._raw(df), self.cals)
