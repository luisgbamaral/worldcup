# Pre-World-Cup model (frozen — paper baseline)

This is the **academic artifact**: models trained, validated and tested on
**pre-World-Cup data only** (`date <= 2026-06-10`, every 2026 World Cup row
excluded). It is the clean historical comparison the paper reports, and it never
sees tournament results — so it stays fixed regardless of how the Cup unfolds.

## Reproduce

```bash
python experiments/forecasting.py --phase 1 --trials 25
```

Deterministic (fixed seeds, temporal walk-forward CV — never shuffled). Outputs:

- `benchmark_metrics.csv` (here) — the frozen headline numbers.
- `reports/tables/benchmark.tex`, `reports/figures/10_benchmark_rps.{pdf,png}`.
- `data/processed/benchmark_results.parquet` (per fold), `benchmark_summary.parquet`.

## What it reports

1X2 forecasting (RPS primary; log-loss / Brier / accuracy / ECE) with isotonic
calibration, comparing direct 1X2 learners, goal (Poisson) models and the
**TabPFN foundation model — the main model**. Elo enters only as a covariate of
the feature base.

> The production tracker (`../../production/`) is a *separate* layer: it ingests
> tournament results as they arrive and simulates the Cup to the champion. It does
> **not** feed back into this frozen pre-Cup model.
