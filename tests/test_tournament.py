"""2026 World Cup bracket + Monte-Carlo simulator sanity checks."""
import polars as pl
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
import tournament as T  # noqa: E402


def test_bracket_and_groups():
    bracket = T.load_bracket()
    assert len(bracket) == 31                       # 16+8+4+2+1 knockout matches
    groups = T.load_groups()
    assert len(groups) == 12 and all(len(v) == 6 for v in groups.values())


@pytest.fixture(scope="module")
def sim():
    return T.monte_carlo(n_sims=200, seed=0)


def test_probabilities_well_formed(sim):
    assert sim.height == 48
    for c in ("p_advance", "p_champion", "p_final"):
        assert sim[c].min() >= 0.0 and sim[c].max() <= 1.0


def test_exactly_32_advance(sim):
    assert abs(sim["p_advance"].sum() - 32.0) < 1e-9   # 12*2 group qualifiers + 8 best thirds


def test_one_champion(sim):
    assert abs(sim["p_champion"].sum() - 1.0) < 1e-9


def test_monotone_rounds(sim):
    # a team cannot reach a later round more often than an earlier one
    for a, b in (("p_R16", "p_advance"), ("p_QF", "p_R16"), ("p_SF", "p_QF"),
                 ("p_final", "p_SF"), ("p_champion", "p_final")):
        assert (sim[a] <= sim[b] + 1e-9).all()
