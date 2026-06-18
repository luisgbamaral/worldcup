# CLAUDE.md — worldcup

Project memory for Claude Code. Versioned and shared with the team: clone the repo and you
have full context, **zero verbal briefing**. (Personal preferences live in `~/.claude/CLAUDE.md`;
on conflict, this project file wins.)

## Persona / how to respond
- **Conversation in Portuguese; everything in the codebase in English** (code, comments,
  docstrings, log messages, commit messages).
- Professional, **short and concise** code with good abstractions — no over-engineering.
- **Report faithfully.** If a result is null, a test fails, or a step was skipped, say so. Never
  tune or cherry-pick to make a finding look better.
- Act when you have enough information; give a recommendation, not an exhaustive menu of options.

## Project goal
International football forecasting, with the **2026 World Cup** as the headline application. Three
purposes: (1) win a prediction pool; (2) a short paper for a journal; (3) a senior-ML portfolio.

## Structure — folders and what they are for
```
src/worldcup/     installable package = the ENGINE (keep stable, reused everywhere)
  config.py  data.py  clean.py  lookups.py   — paths, loading, canonical team names
  elo.py  ratings.py                         — self-computed Elo + rating systems
  features.py                                — 107 leakage-safe match features (+ cache)
  modeling.py                                — classifiers, calibration, RPS, temporal CV, HPO
  update.py                                  — refresh raw data from public sources
experiments/      research scripts (ablations, evaluations) — the PAPER lives off these
production/       standalone 2026 forecaster (depends only on src/, NOT on the paper)
paper/            short_paper.md — the write-up
reports/          GENERATED output: figures/ and tables/ (tracked)
scripts/          one-shot build/update entry points
tests/            pytest suite
data/             see data policy below
notebooks/  models/
```

### Files you must NOT edit
- **`data/raw/**` — raw source data. Never hand-edit.** It is refreshed *only* via
  `python -m worldcup.update` (downloads from the canonical GitHub sources). Treat it as
  read-only input; the permission rules deny edits to it.
- **`.env`** — holds `TABPFN_TOKEN`. Never read into context, never commit (gitignored).

### Where generated output goes
- `reports/figures/`, `reports/tables/` — committed artifacts (charts, LaTeX/CSV tables).
- `data/processed/`, `data/interim/` — **gitignored** generated layers (feature caches,
  result parquets). Safe to delete and regenerate.
- `production/outputs/` — gitignored forecast CSVs.

## Code style
- **`snake_case`** for functions/variables; `UPPER_SNAKE` for module constants; `PascalCase` for
  classes. Module names short and lowercase.
- **Polars first** for dataframes (not pandas) unless a dep forces otherwise.
- Every module starts with a one-line-plus docstring saying what it does and how to run it;
  comments explain *why*, not *what*. Match the density of the surrounding code.
- Logs/prints in English, terse, append-only (e.g. `print(..., flush=True)` in long runs).

## Config / stack
- **Python ≥ 3.10** (this machine runs 3.13). **No R or Julia** — Python only.
- Core libs: **Polars**, NumPy, scikit-learn, **XGBoost**, **CatBoost**, **Optuna**,
  Matplotlib, PyArrow; foundation models **TabPFN** (cloud, needs `TABPFN_TOKEN`) and **TabICL**
  (CPU, local). Full list in `requirements.txt` / `pyproject.toml`.
- Secrets: `.env` (gitignored), template in `.env.example`.

### How to run
```bash
pytest                                   # test suite (pythonpath=src configured in pyproject)
python -m worldcup.update                # refresh data/raw/ from public sources
python scripts/build_features.py         # rebuild the feature cache
python experiments/<name>.py             # an experiment / ablation
python production/predict_wc2026.py      # the live 2026 champion forecast
```
Heavy experiments (foundation models on CPU, Optuna) can run for minutes to hours — prefer
background runs with append-only logs.

## Team & responsibilities
- **Luís Guilherme Brandão Amaral** (LEME) — lead/author; owns modeling, features, evaluation.
- **Claude (Opus)** — pair contributor; commits end with `Co-Authored-By: Claude ...`.
- Solo repo today; conventions here keep it onboarding-ready for new members.

## Architectural decisions already made (do not silently revert)
- **Elo is a COVARIATE, never a baseline.** No "Elo" row in any benchmark; the significance
  reference is the simplest feature model (**LogReg on all features**).
- **Leakage-safe + strictly temporal.** Every feature uses only data prior to the match; CV is
  walk-forward, **never** shuffled k-fold; calibration on a temporal slice.
- **RPS is the primary metric** (ordered 1X2); log-loss/Brier/ECE are secondary.
- **Foundation models stay tune-free** (TabPFN/TabICL) — that configuration-free property is the
  research claim under test; only *classical* baselines may be tuned.
- The feature build is **cached** (`features.build_match_features_cached`); the row-level build is
  expensive, so reuse the cache across scripts.

## Scope — what NOT to do
- Don't edit `data/raw/**` or commit/read `.env`.
- Don't reintroduce Elo (or any rating) as a standalone model/baseline row.
- Don't tune the foundation models, and don't use shuffled CV or any future-peeking feature.
- Don't add pandas where Polars fits; don't add heavy deps without need.
- Don't push to `main` or run `worldcup.update` without confirmation (see `.claude/settings.json`).

## Example of a well-formed task (Context → Example → Plan → Scope)
> **Context:** `experiments/worldcup_eval.py` already does one-step 1X2 eval with RPS + Holm tests.
> **Want:** add multiclass **log-loss** next to RPS in the series and World-Cup tables, computed
> from the saved per-match probabilities — same table style as `reports/tables/onestep_series.tex`.
> **Plan:** read `data/processed/onestep_*.parquet`; compute log-loss per model; append a column;
> re-emit the LaTeX via `viz.df_to_neurips_latex`.
> **Scope:** don't retrain models, don't touch `data/raw/`, keep RPS as the primary (first) metric.
