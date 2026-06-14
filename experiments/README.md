# Experiments

Modeling experiments built on top of the EDA in [`../notebooks`](../notebooks)
and the reusable code in [`../src/worldcup`](../src/worldcup).

Each experiment is self-contained and numbered. The goal is to **forecast the
2026 World Cup** from historical results and Elo ratings.

| # | Experiment | Idea | Status |
|---|------------|------|--------|
| 01 | Elo baseline | Win/draw/loss probabilities from current Elo difference | planned |
| 02 | Poisson goals model | Team attack/defense strengths → scoreline distribution | planned |
| 03 | Elo–Poisson hybrid | Elo as a covariate in the Poisson rates | planned |
| 04 | Tournament simulation | Monte-Carlo the 2026 bracket → title probabilities | planned |

> Raw outputs (metrics, simulated brackets) go to `data/processed`;
> figures to `reports/figures`.
