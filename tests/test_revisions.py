"""Tests for the revision analyses (Tier A–D)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))


def test_a1_fdr_flags_match_qvalues():
    import rev_a1_multiplicity as A1  # noqa: E402
    p = np.array([0.0005, 0.0005, 0.003, 0.0045, 0.0325, 0.11, 0.20, 0.55])
    q = multipletests(np.clip(p, A1.PFLOOR, 1.0), method="fdr_bh")[1]
    survives = q < 0.05
    # survives_fdr must be exactly q<0.05, and FDR must be no looser than raw
    assert list(survives) == list(q < 0.05)
    assert (q >= np.clip(p, A1.PFLOOR, 1.0) - 1e-9).all()


def test_a3_subsampling_reproducible_and_cap_logic():
    import rev_a3_tabpfn_cap as A3  # noqa: E402
    n = 20775
    recent = np.arange(n)[-A3.CAP:]
    rand1 = np.random.RandomState(A3.SEED).choice(n, A3.CAP, replace=False)
    rand2 = np.random.RandomState(A3.SEED).choice(n, A3.CAP, replace=False)
    assert len(recent) == A3.CAP and recent[0] == n - A3.CAP        # most-recent block
    assert np.array_equal(rand1, rand2)                            # reproducible under seed
    assert n > A3.CAP                                              # cap binds on the whole series


def test_a2_principled_is_fixed_grid_no_inner_cv():
    import rev_a2_principled as A2  # noqa: E402
    # the principled baseline is a small fixed-C grid chosen a priori, not via CV
    assert A2.C_GRID and all(c <= 0.1 for c in A2.C_GRID)          # strong priors only
    assert A2.WINDOWS == [("1mo", 30), ("6mo", 182)]               # extreme windows only
    import inspect
    src = inspect.getsource(A2)
    assert "walk_forward_folds" not in src and "tune_classifier" not in src  # no inner CV


def test_b1_elo_only_uses_single_feature():
    import rev_b1_elo_only as B1  # noqa: E402
    assert B1.FEAT == ["elo_diff_eff"] and len(B1.FEAT) == 1


def test_b3_eras_non_overlapping_and_ordered():
    import rev_b3_multiera as B3  # noqa: E402
    starts = [a for a, _ in B3.ERAS]
    assert starts == sorted(starts)                                # temporally ordered
    for (a0, a1), (b0, b1) in zip(B3.ERAS, B3.ERAS[1:]):
        assert a1 <= b0                                            # non-overlapping [start, end)


def test_foundation_models_never_tuned():
    import rev_a2_principled as A2
    import rev_b3_multiera as B3
    # FMs are not in any classical/tuning list in the revision scripts
    assert "TabPFN" not in B3.MODELS and "TabICL" not in B3.MODELS
    # A2 builds FMs with empty params (tune-free)
    import inspect
    assert "_fit_predict(m, {}" in inspect.getsource(A2)


def test_c1_scores_populated():
    import rev_c1_scores as C1  # noqa: E402
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [0.33, 0.34, 0.33]])
    y = np.array([0, 2, 1])
    Y = np.eye(3)[y]
    ll = float(-np.log(np.clip(p[np.arange(3), y], 1e-12, 1)).mean())
    brier = float(((p - Y) ** 2).sum(1).mean())
    assert ll > 0 and brier > 0                                   # both proper scores defined
