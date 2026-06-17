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

## 3b. Training-size ablation — does the FM advantage survive a *tuned* classical baseline?

For growing train windows {1mo, 6mo, 1y, 2y, 5y, 8y} and two feature sets (`few` = top-8
greedy, `best` = 107) we re-fit every model and measure RPS. **Crucially, the classical
comparators (LogReg/CatBoost/XGBoost) are re-tuned at every cell** with an inner *temporal*
CV inside the training window (LogReg: `C`×{l1,l2} grid; GBDTs: Optuna, identical budget per
window); the foundation models stay **tune-free**. One-step uses a fixed test (2023+) and
fixed calibration (2022) with ≥5-seed CIs; multi-step uses the per-Cup pre-kickoff window
with CIs across the five Cups. This replaces an earlier un-tuned run (fixed `C=1`) whose
dramatic logistic "collapse" (Cup-RPS 0.347 at 1mo) was partly a regularisation artefact.

### One-step — test RPS (2023+), 107 features, **classical tuned per window**

| model | 1mo | 6mo | 1y | 2y | 5y | 8y |
|---|---|---|---|---|---|---|
| **TabPFN** | **0.1708** | **0.1684** | 0.1691 | 0.1691 | **0.1675** | 0.1679 |
| **TabICL** | 0.1821 | 0.1689 | **0.1688** | **0.1690** | 0.1680 | **0.1678** |
| LogReg | 0.1819 | 0.1697 | 0.1710 | 0.1760 | 0.1688 | 0.1682 |
| CatBoost | 0.1825 | 0.1721 | 0.1710 | 0.1711 | 0.1692 | 0.1681 |
| XGBoost | 0.1849 | 0.1727 | 0.1728 | 0.1715 | 0.1690 | 0.1689 |

### Survival test — best FM vs best **tuned** classical (paired per-match RPS, clustered bootstrap)

Δ = classical − FM (positive ⇒ FM better); **bold p ⇒ FM significantly better** (p<0.05).

| scope / features | 1mo | 6mo | 1y | 2y | 5y | 8y |
|---|---|---|---|---|---|---|
| one-step / **best** | Δ.011 **p0** † | Δ.0012 **p.005** | Δ.0018 **p0** | Δ.0017 **p.003** | Δ.0012 **p0** | Δ−.0001 p.70 |
| one-step / few | Δ.008 **p0** † | Δ.001 **p.03** | Δ.0008 p.19 | Δ.0004 p.20 | Δ.0002 p.30 | Δ.0001 p.26 |
| multi-step / **best** | Δ.004 **p0** † | Δ.009 **p.01** † | Δ.003 p.11 | Δ.002 p.13 | Δ.0015 p.19 | Δ−.0 p.55 |
| multi-step / few | Δ.006 p.07 † | Δ.002 p.17 † | Δ.0008 p.20 | Δ.0003 p.41 | Δ−.0004 p.65 | Δ−.002 p.76 |

† classical models were `hpo_degenerate` (≤57–~370 samples ⇒ inner CV invalid ⇒ strong-reg fallback;
they **cannot be tuned at all** in this corner, which is itself a point for tune-free FMs).

**Findings (the FM advantage survives — conditionally):**
- **High-dimensional + small/medium data is where FMs win significantly.** In one-step `best`
  (107 features) the FM beats the **tuned** classical with **p<0.05 from 1 month through 5 years**,
  converging to a tie only at **8 years** (Δ≈0, p=0.70).
- **It ties when features are few or data is abundant.** With `few` (8 features) the gap is
  significant only at 1–6 months, then ties (the classical models stop over-fitting); and at
  8 years everyone converges regardless of feature count.
- **Per-window tuning closes the gap but does not erase it.** Tuning fixed the earlier artefact
  (the 0.347 blow-up is gone), yet the FM edge persists and is significant precisely in the
  high-dim/low-sample regime that tournament forecasting inhabits.
- **Tuning-cost asymmetry.** The classical models paid 30–180 inner-CV fits per cell; the FMs
  paid **zero**. Even where RPS ties (8y), the FM reaches it tune-free — a practical win.

Artifacts: `ablation_{onestep,multistep,survival}.{parquet,csv,tex}`, `15_trainsize_onestep`,
`17_trainsize_multistep_rps` (curves with 95% CI bands).

## 4. Headline conclusions

1. **Elo is the load-bearing covariate** — it alone takes RPS from 0.207 to ~0.169; everything else is marginal.
2. **General series:** a plain LogReg on the full feature table is statistically as good as TabPFN/TabICL/GBDTs — sophistication buys no significant gain.
3. **World-Cup regime:** foundation models (TabPFN/TabICL) edge ahead on the harder, scarcer Cup matches — directionally consistent, not yet significant (256 matches).
4. **Tournament simulation:** predicting champions from frozen pre-Cup state is dominated by the Elo favorite (Brazil), which rarely wins — TabICL was the only model to pick a non-Brazil champion correctly (Spain 2010).
5. `is_knockout` and `is_world_cup` carry genuine signal (entered the greedy order at 7 and 12).
6. **Training-size ablation is the strongest result — and it survives per-window tuning.**
   With the classical comparators re-tuned at every window (so the comparison is FM vs a
   *properly regularised* baseline, not a straw man), the tune-free foundation models still
   beat them **with statistical significance from 1 month through 5 years of data in the
   high-dimensional (107-feature) setting**, converging to a tie only at 8 years or when
   features are few. The earlier dramatic "collapse" (LogReg 0.347) was partly a fixed-`C`
   artefact and disappears under tuning — but the core finding holds: tune-free tabular FMs
   are the robust choice in the high-dimensional, small-sample regime that World-Cup
   forecasting inhabits, and they get there at **zero tuning cost** (vs 30–180 inner-CV fits).
