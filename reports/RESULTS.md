# World Cup 1X2 Forecasting — Results Summary

Pipeline: `feature_forward.py` → `worldcup_eval.py` → `worldcup_multistep.py`.
Target: match outcome (1X2 = Home / Draw / Away). Primary metric: **RPS** (lower is better).
Elo is a **covariate**, never a baseline. Significance reference = **LogReg** (simplest feature model).
Models: LogReg, CatBoost, XGBoost (tuned GBDTs) + tune-free foundation models **TabPFN** (cloud) and **TabICL** (CPU).

---

## 1. Feature engineering & forward selection

- **107 leakage-safe features**: Elo (level/diff/momentum/win-prob), rolling form (5/10-game ppg, goals for/against, win/clean-sheet/over-2.5 rates), fatigue (rest days, matches in 30/90d), head-to-head, squad (age/height/abroad share/top-5 league/star caps), venue/altitude, host context, and phase flags `is_world_cup` / `is_knockout`.
- **Greedy forward selection** (LogReg-guided, from empty; subsampled 10k rows of the full pre-2022 history, all eras):

  | # | feature | # | feature |
  |---|---|---|---|
  | 1 | `elo_diff_eff` | 7 | **`is_knockout`** ✅ |
  | 2 | `ga_pg_5_diff` (goals against, last 5) | 8 | `cs_rate_10_away` |
  | 3 | `elo_momentum_diff` | 9 | `win_rate_5_home` |
  | 4 | `rest_days_diff` | 10 | `days_since_last_meeting` |
  | 5 | `matches_90d_diff` (fatigue) | 11 | `gd_pg_10_diff` |
  | 6 | `away_elo_pre` | 12 | **`is_world_cup`** ✅ |

- **Optimal set `k* = 107` (full set).** The RPS-vs-#features curves are **very flat**: Elo alone lifts every model from RPS ≈ 0.207 (wins-only) to ≈ 0.169 with 1-2 features; beyond that, extra features help only marginally, and the pooled optimum lands on the complete set.
- Both phase flags entered the order — the greedy finds real signal in "is it a knockout?" (7th) and "is it a World Cup?" (12th).

Artifacts: `13_feature_forward.{pdf,png}`, `feature_forward.csv`, `optimal_features.json`.

---

## 2. One-step (per-match) evaluation

### 2a. Whole historical series — train < 2022, test 2023+ (3597 matches)

| model | RPS ↓ | Hits | Accuracy | ECE | vs LogReg (Holm) |
|---|---|---|---|---|---|
| **LogReg** | **0.1685** | 2170 | 60.3% | 0.026 | — (reference) |
| TabPFN | 0.1691 | 2173 | 60.4% | 0.025 | n.s. (1.0) |
| TabICL | 0.1692 | **2177** | **60.5%** | 0.027 | n.s. (1.0) |
| CatBoost | 0.1704 | 2160 | 60.0% | 0.030 | n.s. (1.0) |
| XGBoost | 0.1736 | 2143 | 59.6% | 0.041 | n.s. (1.0) |

**On the general series, LogReg has the best RPS and nothing beats it significantly.** Foundation models tie (slight edge in hits/accuracy); GBDTs trail slightly. After Elo + form, the signal is essentially linear.

### 2b. World Cups only — rolling 8-year train per Cup (256 matches, 2010–2022)

| model | RPS ↓ | Hits | Accuracy | ECE |
|---|---|---|---|---|
| **TabPFN** | **0.2063** | 140 | 54.7% | 0.060 |
| TabICL | 0.2079 | 140 | 54.7% | 0.051 |
| LogReg | 0.2083 | 136 | 53.1% | 0.049 |
| CatBoost | 0.2098 | 140 | 54.7% | 0.041 |
| XGBoost | 0.2141 | 138 | 53.9% | 0.057 |

**The ranking flips: on harder World-Cup matches the foundation models lead** (TabPFN best, TabICL 2nd). With only 256 matches no difference is significant (`p_holm = 1.0`), but it matches the thesis — tune-free FMs shine in the small/hard-data regime. RPS is much higher than the series (~0.207 vs 0.169): Cup matches are genuinely harder to call.

Artifacts: `onestep_{series,worldcup}.tex`, `12_cumhits_{series,worldcup}.{pdf,png}`, parquets.

---

## 3. Multi-step (tournament simulation) — last 5 World Cups

Protocol: for each Cup, train on **the 5 years immediately before it** (all international matches, frozen — no in-tournament update), snapshot each team's last pre-Cup feature vector, then **simulate the full bracket forward** (group stage → standings → knockout → champion) over **10,000 Monte-Carlo runs**. Podium points: **+1 per correctly-placed top-3 position (max 3/Cup)**.

### Predicted vs actual champion

| Cup | actual | LogReg | CatBoost | XGBoost | TabPFN | TabICL |
|---|---|---|---|---|---|---|
| 2006 | 🇮🇹 Italy | Brazil | Netherlands¹ | Netherlands¹ | Brazil | Brazil |
| 2010 | 🇪🇸 Spain | Brazil | Brazil | Brazil | Brazil | **🇪🇸 Spain** ✅ |
| 2014 | 🇩🇪 Germany | Brazil | Brazil¹ | Brazil | Brazil | Brazil |
| 2018 | 🇫🇷 France | Brazil | Brazil | Spain | Brazil | Brazil |
| 2022 | 🇦🇷 Argentina | Brazil | Brazil¹ | Brazil | Brazil¹ | Brazil¹ |

¹ hit one podium position (not the champion).

### Totals (max 15 podium points)

| model | podium points | champions correct |
|---|---|---|
| **CatBoost** | **3** | 0 |
| **TabICL** | **2** | **1** (Spain 2010) |
| TabPFN | 1 | 0 |
| XGBoost | 1 | 0 |
| LogReg | 0 | 0 |

**Honest takeaway:** with frozen pre-Cup features, every model bets on **Brazil** (highest Elo) almost every time — and Brazil won **none** of these five Cups (all upsets relative to the Elo favorite). Only **TabICL broke from the favorite and nailed Spain 2010**. CatBoost accumulated the most podium points by getting 2nd/3rd places right without ever calling the champion. Tournament simulation from a frozen favorite is intrinsically hard; podium points reward partial correctness.

Artifacts: `worldcup_multistep.{parquet,tex}`, `14_multistep_cumpoints.{pdf,png}`, `multistep_points.csv`.

---

## 4. Headline conclusions

1. **Elo is the load-bearing covariate** — it alone takes RPS from 0.207 to ~0.169; everything else is marginal.
2. **General series:** a plain LogReg on the full feature table is statistically as good as TabPFN/TabICL/GBDTs — sophistication buys no significant gain.
3. **World-Cup regime:** foundation models (TabPFN/TabICL) edge ahead on the harder, scarcer Cup matches — directionally consistent, not yet significant (256 matches).
4. **Tournament simulation:** predicting champions from frozen pre-Cup state is dominated by the Elo favorite (Brazil), which rarely wins — TabICL was the only model to pick a non-Brazil champion correctly (Spain 2010).
5. `is_knockout` and `is_world_cup` carry genuine signal (entered the greedy order at 7 and 12).
