# Experiments

Modelling on top of the consolidated feature base
([`match_features` / `team_match_features`](../data/processed)). The reusable
machinery lives in [`../src/worldcup/modeling.py`](../src/worldcup/modeling.py);
these scripts only compose it (temporal protocol, never shuffled CV).

## Forecasting — `forecasting.py` (three phases)

```bash
python experiments/forecasting.py --phase 1 [--trials N]   # academic benchmark
python experiments/forecasting.py --phase 2                # test on played WC games
python experiments/forecasting.py --phase 3                # daily production
```

| Phase | Trains on | Predicts | Output |
|-------|-----------|----------|--------|
| **1 — Benchmark** | pre-WC only (`date <= 2026-06-10`), walk-forward CV + held-out test | held-out pre-WC slice | `benchmark_results.parquet` (per fold), `benchmark_summary.parquet`; table `benchmark.tex`, fig `10_benchmark_rps` |
| **2 — Tournament test** | pre-WC, as of each game's date | the 2026 games already played (out-of-sample) | `wc2026_scorecard.parquet`; table `wc2026_scorecard.tex` |
| **3 — Production** | everything known now (incl. played WC games) | scheduled fixtures | `wc2026_predictions.parquet`, append-only `wc2026_prediction_log.parquet`; table `wc2026_predictions.tex` |

**Models.** Arm A (direct 1X2): LogReg, RandomForest, ExtraTrees, XGBoost,
CatBoost, and the **TabPFN foundation model — our main model** (cloud, free tier,
`TABPFN_TOKEN` in `.env`). Arm B (goals → independent-Poisson scoreline): Poisson
GLM, XGBoost/CatBoost Poisson. **Elo is a covariate of the feature base, not a
model.**

**Protocol.** Optuna HPO runs only inside walk-forward CV (`--trials`,
classification GBDTs). Calibration (isotonic) is fit on a temporal validation
slice. Metric: **RPS** (primary), plus log-loss / Brier / accuracy / ECE.
Everything is seeded and deterministic.

> Phase 1 excludes every 2026 World Cup row — it is the clean historical
> comparison. The tournament rows appear only in Phases 2 and 3.

## World-Cup hit-count evaluation — `worldcup_eval.py`

```bash
python experiments/worldcup_eval.py
```

Rolling **8-year train per World Cup** (`WORLD_CUPS = [2010, 2014, 2018, 2022]`),
predict every match of that tournament. Compares LogReg / CatBoost / XGBoost and the
tune-free foundation models (TabPFN / TabICL / TabDPT when available), **all on the
single-rating feature table** (Elo is a covariate, not a baseline; `ratings.py`
generates it) with CFS selection. Metrics: **RPS** (primary), hits, accuracy, ECE.
Significance vs the simplest feature model **LogReg**: bootstrap clustered by World
Cup + Diebold-Mariano (RPS) and McNemar (correctness), Holm-adjusted. Outputs:
`reports/tables/worldcup_eval.tex`, `reports/figures/12_cumulative_hits.{pdf,png}`
+ `cumulative_hits.csv`, `data/processed/worldcup_eval.parquet`.

**Finding.** On World-Cup matches all feature models tie on RPS (~0.207; none beats
LogReg significantly, Holm p = 1.0) — the **tune-free foundation models (TabICL/TabPFN)
match the tuned GBDTs and even plain logistic regression**; TabICL/XGBoost lead on raw
hits (143/256, 55.9%).
