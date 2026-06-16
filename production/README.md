# production/ — 2026 World Cup live forecast

Standalone forecaster for the **2026 World Cup champion**. Depends only on the
`worldcup` engine in `src/` (features, Elo, modeling) — **not** on the paper/experiments,
which can be deleted after the Cup.

## Run

```bash
python production/predict_wc2026.py            # default model: TabPFN
WC2026_MODEL=CatBoost python production/predict_wc2026.py
```

Outputs `production/outputs/wc2026_{pretournament,conditional}.csv` with each team's
`P_champion`, `P_final`, `P_podium`, plus a printed top-12.

## What it does

1. Reads `data/raw/fixtures/wc2026_group_stage_schedule.csv` (12 groups of 4, 72 matches).
2. **Host advantage** applies only when a host plays in its **own** country
   (USA in the USA, Mexico in Mexico, Canada in Canada) — venue→country from the venue
   string; otherwise the match is neutral.
3. Snapshots each team's latest pre-tournament feature vector and trains the chosen
   classifier (default **TabPFN**, the data-efficient pick from the training-size ablation)
   on the 8 years of internationals before kickoff (isotonic-calibrated on a temporal slice).
4. Simulates the tournament **10,000×**: group stage → 12 winners + 12 runners-up + 8 best
   thirds → a **strength-reseeded 32-team knockout bracket** → champion / runner-up / third.

## Two modes

- **pretournament** — ignores results so far; clean prior forecast.
- **conditional** — uses the group games already played and simulates the rest (the live call).

## Caveats

- The knockout bracket is **strength-reseeded**, not the exact official FIFA slotting (the
  best-thirds combination table); champion probabilities are robust to this, exact paths are not.
- Group ranking tie-breaks use points then Elo (goals are out of scope).
- It is a **probabilistic** forecast — it reports each team's championship probability, not a
  single guaranteed winner; tournament variance is large.
