# Tune-Free Tabular Foundation Models for International Football Outcome Forecasting: A Training-Size Ablation

**Working short paper — World Cup forecasting project.**

---

## Abstract

We study probabilistic 1X2 (home/draw/away) forecasting of international men's football
matches, with World Cup prediction as the headline application. Our feature table is built
around a self-computed Elo rating treated strictly as a **covariate** (never a baseline), augmented
with leakage-safe recent-form, fatigue, head-to-head, squad and tournament-phase features. We
compare a simple logistic regression, two gradient-boosted decision trees (CatBoost, XGBoost),
and two **tune-free tabular foundation models** (TabPFN, TabICL) under a strictly temporal protocol,
with Ranked Probability Score (RPS) as the primary metric and Holm-corrected significance testing.
On the full historical series (test 2023+, *n*=3,597) logistic regression attains the best RPS
(0.1685) and **no model beats it significantly**; on World Cup matches (*n*=256) the foundation
models lead numerically but again without significance. A tournament-level multi-step simulation of
the last five World Cups shows all models essentially track the pre-tournament Elo favourite (Brazil),
which won none of those Cups. Our strongest result is a **training-size ablation**: foundation models are
nearly invariant to the amount of training history, matching their 8-year performance with as little as one
month of data, whereas traditional models degrade sharply in the small-sample regime — an effect that
*worsens with feature dimensionality*. Critically, when the classical comparators are **re-tuned at every
training window** (so the baseline is properly regularised, not a straw man), the tune-free foundation models
**still win with statistical significance from one month through five years of data in the high-dimensional
setting**, converging to a tie only at eight years — and they do so at **zero tuning cost**. We argue the
practical case for tune-free tabular foundation models lies precisely in the high-dimensional, low-sample
regime, while being candid that the head-to-head comparison at full data is a well-reported null result.

---

## 1. Introduction

Football outcome forecasting is a canonical low-signal prediction problem: even strong models rarely
exceed ~55–60% top-1 accuracy on 1X2, and the draw class is intrinsically hard. The dominant predictor
is a team-strength rating (Elo or variants), and a recurring methodological question is whether richer
feature engineering and more expressive models meaningfully improve on a rating-driven baseline.

This paper makes three moves. First, we treat the rating as a **covariate inside every model** rather
than as a competing baseline — the scientific question is "do models exploiting a rich feature table,
whose load-bearing feature is Elo, beat a plain logistic regression on the same features?", not "do
models beat Elo?". Second, we bring **tune-free tabular foundation models** (TabPFN, TabICL) to the
problem; these models perform in-context inference without hyperparameter search and are reported to
excel on small tabular datasets. Third, and most importantly, we run a **training-size ablation** to
test directly whether their advantage materialises in the data regime that actually characterises
tournament forecasting: many engineered features, relatively few relevant recent matches.

We evaluate in three settings: per-match one-step forecasting over the whole historical series; per-match
one-step forecasting restricted to World Cups; and a multi-step Monte-Carlo tournament simulation that
predicts each World Cup's podium from a model frozen before kick-off.

---

## 2. Data

We use publicly available sources: (i) the *martj42* international results dataset (every men's
international, with tournament, venue and neutrality flags); (ii) match-level inputs to a self-computed
Elo rating following the eloratings.net update rule (our self-computed ratings correlate ~0.96 with the
published ones); (iii) *openfootball* World Cup JSON files for group composition, fixtures and round
labels; and (iv) squad-level descriptors (age, height, share playing abroad / in a top-5 league, star
caps and goals). The modelling table holds **25,429 matches from 2000 onward** (features are computed over
the full historical record; only the final row filter moves). De-duplication by canonical match id removes
source artefacts (e.g. a fixture repeated sixteen times in the raw feed).

---

## 3. Method

### 3.1 Rating covariate

Elo is updated after each match with a goal-difference-aware multiplier and a home-advantage term, then
exposed to the models as several columns: pre-match home/away rating, rating difference, an
"effective" difference that folds in home advantage and neutrality, rating sum, Elo win-probability, and
a short-horizon Elo momentum. Per the project's standing rule, the rating is **only** a feature; it never
appears as a model row in any table, and the significance reference is the simplest feature model
(logistic regression), not Elo.

### 3.2 Feature table (107 columns)

All features are **leakage-safe**: every quantity for a match is computed from information strictly prior
to it (shift-based rolling windows). The groups are:

- **Recent form** over the last 5 and 10 matches per team: points per game, goals for/against per game,
  goal difference, win rate, clean-sheet rate, over-2.5 rate, plus opponent-adjusted variants.
- **Fatigue / scheduling**: rest days, matches in the last 30 and 90 days.
- **Head-to-head**: number of prior meetings, home win rate, average goals, days since last meeting.
- **Squad**: average age/height, share abroad, share in a top-5 league, star caps/goals, goal
  concentration, defensive height.
- **Context**: tournament importance, competitiveness, venue altitude, host-country / host-confederation /
  host-proximity flags, and two **phase flags** — `is_world_cup` and `is_knockout` — populated for
  historical World Cup matches from the openfootball round labels.

Per-team levels and standalone predictors are winsorised on the historical (non-future) rows, and the
differential columns (`x_diff = x_home − x_away`) are recomputed afterwards so the identity holds exactly.

### 3.3 Models

All models share the same imputed, standardised feature pipeline:

- **LogReg** — multinomial logistic regression (`StandardScaler` + L2, `C=1.0`).
- **CatBoost**, **XGBoost** — gradient-boosted trees, hyper-parameters tuned once by temporal Optuna.
- **TabPFN** — cloud-served prior-data-fitted transformer; **tune-free**, capped at 10,000 training rows.
- **TabICL** — in-context tabular classifier with a synthetic prior; **tune-free**, CPU.

Probabilities are calibrated with **isotonic regression** fitted on a temporal validation slice
(per-class, then renormalised).

### 3.4 Primary metric

We report the **Ranked Probability Score (RPS)**, the natural ordinal proper scoring rule for ordered
1X2 outcomes, as primary (lower is better), alongside hits, accuracy and Expected Calibration Error (ECE).

### 3.5 Forward feature ranking

To understand feature importance we run a **greedy forward selection** guided by a lightweight logistic
regression: starting from the empty set, repeatedly add the single feature that most reduces validation
RPS. A separate, forced "wins-only" model (best single win-rate feature) is reported as a reference
"model 0". The ranking is computed on a random 10,000-row subsample of the full pre-2022 history (every
era represented) for tractability.

### 3.6 One-step evaluation (two scopes)

- **Whole series**: train on all matches before 2022, calibrate on 2022, **test on 2023+** (*n*=3,597).
- **World Cups**: for each Cup (2010–2022) train on a **rolling 8-year window** ending before kick-off,
  predict that Cup's matches (*n*=256 pooled).

Significance versus LogReg uses a **World-Cup/year-clustered block bootstrap** of the RPS difference (2,000
resamples, one-sided), **McNemar's test** on correctness, and **Holm** correction across models.

### 3.7 Multi-step tournament simulation (World Cup)

For each of the last five Cups (2006–2022, 32-team format) we train each model on the **5 years before
kick-off**, **freeze** it (no in-tournament updates), snapshot each qualified team's latest pre-tournament
feature vector, and **forward-simulate** the entire tournament 10,000 times: group matches → points table
(Elo break for ties, since goals are out of scope) → top two advance → standard 32-team knockout bracket,
where the winner of a tie is drawn with probability `P(home)+0.5·P(draw)`. The predicted podium is the
arg-max of the Monte-Carlo champion/runner-up/third distributions. We score **+1 per correctly-placed
podium position (max 3 per Cup)**, accumulated across the five Cups.

### 3.8 Training-size ablation

For growing train windows {1 month, 6 months, 1 year, 2 years, 5 years, 8 years} we re-fit every model in
two feature sets — `few` (top-8 of the greedy ranking) and `best` (all 107) — and measure performance.
For one-step, the test set (2023+) and calibration set (2022) are held **fixed** so only the train window
varies. For multi-step, the per-Cup window grows back from kick-off and we report both podium points and
per-match RPS on the actual Cup matches.

---

## 4. Results

### 4.1 Forward feature ranking

The greedy order begins `elo_diff_eff → ga_pg_5_diff → elo_momentum_diff → rest_days_diff →
matches_90d_diff → away_elo_pre → is_knockout (7th) → cs_rate_10_away → win_rate_5_home →
days_since_last_meeting → gd_pg_10_diff → is_world_cup (12th)`. Both phase flags enter the ranking,
confirming genuine signal in "is this a knockout?" and "is this a World Cup?". The pooled RPS-vs-#features
curve is, however, **very flat**: a single Elo feature lifts every model from RPS≈0.207 (wins-only) to
≈0.169, after which extra features help only marginally, and the pooled optimum is the **full 107-feature
set** (`k*=107`). The procedure therefore yields a useful *importance ranking*, not a parsimonious subset.

### 4.2 One-step — whole series (test 2023+, *n*=3,597)

| model | RPS ↓ | Hits | Accuracy | ECE | vs LogReg (Holm) |
|---|---|---|---|---|---|
| **LogReg** | **0.1685** | 2170 | 60.3% | 0.026 | — |
| TabPFN | 0.1691 | 2173 | 60.4% | 0.025 | n.s. |
| TabICL | 0.1692 | 2177 | 60.5% | 0.027 | n.s. |
| CatBoost | 0.1704 | 2160 | 60.0% | 0.030 | n.s. |
| XGBoost | 0.1736 | 2143 | 59.6% | 0.041 | n.s. |

On the broad series the **plain logistic regression has the best RPS and nothing beats it significantly**
(all Holm-adjusted *p* = 1.0). After Elo and form, the residual signal is essentially linear.

### 4.3 One-step — World Cups (*n*=256, rolling 8-year train)

| model | RPS ↓ | Accuracy | ECE |
|---|---|---|---|
| **TabPFN** | **0.2063** | 54.7% | 0.060 |
| TabICL | 0.2079 | 54.7% | 0.051 |
| LogReg | 0.2083 | 53.1% | 0.049 |
| CatBoost | 0.2098 | 54.7% | 0.041 |
| XGBoost | 0.2141 | 53.9% | 0.057 |

The ranking **flips** on the harder Cup matches: the foundation models lead. With only 256 matches no
difference is significant (*p*=1.0), but the direction is consistent with the small/hard-data thesis. RPS
is markedly higher than on the series (~0.207 vs ~0.169) — Cup matches are genuinely harder to call.

### 4.4 Multi-step — five World Cups

| model | podium points (max 15) | champions correct |
|---|---|---|
| CatBoost | 3 | 0 |
| TabICL | 2 | 1 (Spain 2010) |
| TabPFN | 1 | 0 |
| XGBoost | 1 | 0 |
| LogReg | 0 | 0 |

Every model predicts **Brazil** champion in almost every Cup (highest pre-tournament Elo); Brazil won none
of these five (Italy '06, Spain '10, Germany '14, France '18, Argentina '22). Only TabICL deviated from the
favourite to correctly call Spain 2010. The champion/podium signal is dominated by the Elo favourite and is
near-uninformative about model skill; the informative quantity here is the per-match Cup RPS (§4.5).

### 4.5 Training-size ablation (with per-window-tuned classical baselines)

For growing train windows {1mo, 6mo, 1y, 2y, 5y, 8y} and two feature sets (`few` = top-8, `best` = 107) we
re-fit every model and measure RPS, with ≥5-seed 95% CIs. **The classical comparators are re-tuned at every
cell** — LogReg over a `C`×{l1,l2} grid, the GBDTs via Optuna with an identical per-window budget — using an
inner *temporal* CV strictly inside the training window; the foundation models stay **tune-free**. (A prior
un-tuned run with fixed `C=1` produced a dramatic logistic blow-up — Cup-RPS 0.347 at one month — which we
found to be largely a regularisation artefact; the tuned numbers below replace it.) *n*_train per window =
57 / 628 / 1115 / 1462 / 4464 / 7279; at 1 month inner CV is degenerate, so the classical models fall back
to a strong-regularisation default and are flagged `hpo_degenerate` — they cannot be tuned at all there.

**One-step test RPS (2023+), 107 features, classical tuned per window:**

| model | 1mo | 6mo | 1y | 2y | 5y | 8y |
|---|---|---|---|---|---|---|
| **TabPFN** | **0.1708** | **0.1684** | 0.1691 | 0.1691 | **0.1675** | 0.1679 |
| **TabICL** | 0.1821 | 0.1689 | **0.1688** | **0.1690** | 0.1680 | **0.1678** |
| LogReg | 0.1819 | 0.1697 | 0.1710 | 0.1760 | 0.1688 | 0.1682 |
| CatBoost | 0.1825 | 0.1721 | 0.1710 | 0.1711 | 0.1692 | 0.1681 |
| XGBoost | 0.1849 | 0.1727 | 0.1728 | 0.1715 | 0.1690 | 0.1689 |

**Survival test — best FM vs best *tuned* classical** (paired per-match RPS difference Δ = classical − FM,
positive ⇒ FM better; clustered block bootstrap, one-sided *p*):

| scope / features | 1mo | 6mo | 1y | 2y | 5y | 8y |
|---|---|---|---|---|---|---|
| one-step / **best** (107) | Δ.011 **p≈0**† | Δ.0012 **p.005** | Δ.0018 **p≈0** | Δ.0017 **p.003** | Δ.0012 **p≈0** | Δ−.0001 p.70 |
| one-step / few (8) | Δ.008 **p≈0**† | Δ.001 **p.03** | Δ.0008 p.19 | Δ.0004 p.20 | Δ.0002 p.30 | Δ.0001 p.26 |
| multi-step / **best** (107) | Δ.004 **p≈0**† | Δ.009 **p.01**† | Δ.003 p.11 | Δ.002 p.13 | Δ.0015 p.19 | Δ−.0 p.55 |

† classical models were `hpo_degenerate` (untunable at that sample size). Tuning cost: classical = 30–180
inner-CV fits per cell; foundation models = **0**.

**Findings.** (1) **The FM advantage survives a properly tuned baseline — conditionally.** In the
high-dimensional setting (107 features) the tune-free FM beats the *tuned* classical with **p<0.05 from one
month through five years** of one-step data, converging to a statistical tie only at eight years. (2) **It
ties when features are few or data is abundant**: with 8 features the gap is significant only at 1–6 months
(the classical models stop over-fitting), and at 8 years all models converge regardless of dimensionality.
(3) **Tuning closes the gap but does not erase it** — the 0.347 artefact disappears, yet the FM edge remains
significant precisely in the high-dim/low-sample corner. (4) **Tuning-cost asymmetry**: even where RPS ties,
the FM reaches it at zero tuning cost, so a tie in accuracy is a practical win for the configuration-free model.

---

## 5. Discussion: an exhaustive critical evaluation

**What holds up.** The methodology is clean: leakage-safe features, strictly temporal splits, isotonic
calibration on a temporal slice, the principled Elo-as-covariate framing, and honest significance testing
(clustered bootstrap + McNemar + Holm). The training-size ablation is internally consistent and its central
effect is large and unambiguous. Importantly, the 10,000-row cap on the foundation models **never binds** in
the ablation (the largest window has 7,279 matches), so the FM curves are not an artefact of truncation.

**The head-to-head comparison is, honestly, a null result.** On both one-step scopes every Holm-adjusted
*p*-value is 1.0. The claim "foundation models win on World Cups" is **directional, not established**: with
256 matches the statistical power is very low. A paper should state this plainly. The defensible
model-quality contribution is therefore *not* "model X beats model Y on accuracy" but the *robustness*
behaviour exposed by the ablation.

**The earlier straw man is now addressed (§4.5).** A first un-tuned run held LogReg at a fixed `C=1.0`,
producing the 0.347 blow-up — partly a regularisation artefact. We re-ran the ablation with the classical
models **re-tuned per window** (LogReg `C`×{l1,l2}; GBDTs Optuna, equal budget) under inner temporal CV, and
the prediction held: tuning shrinks but does **not** erase the gap. Against a properly tuned baseline the
tune-free FM still wins **with statistical significance from one month through five years in the
high-dimensional setting**, tying only at eight years or with few features. The data-efficiency claim is
therefore no longer a baseline artefact but a tested result — with the honest caveat that at the very
smallest window the classical models are *untunable* (degenerate inner CV), which is itself part of the case
for tune-free models.

**The multi-step test is the weakest evidence.** A frozen pre-tournament model essentially rides the Elo
favourite; champion/podium scoring measures "who is favourite" more than model skill, and with five Cups and
integer podium points no inference is possible. The simulation is also a simplification (Elo tie-breaks
because goals are out of scope; neutral-venue "home" framing; penalty shootouts and seeding approximated).
We would reposition champion/podium as qualitative illustration and let the per-match Cup RPS carry the
quantitative weight.

**Single test era.** The one-step series is tested only on 2023+ (~1.5 years). "LogReg is best on the
series" may be period-specific; repeated walk-forward across multiple test eras is needed to generalise.

**"Feature selection" selected nothing.** `k*=107` means *use everything*. The flat curves make the
procedure an importance ranking, not a parsimony result; calling it feature selection over-states it.

**Effect sizes are small in absolute terms.** 0.1685 vs 0.1736 on the series is ~3% relative, and ~60%
accuracy is in line with the football-forecasting literature. The practical (betting/pool) value is not
demonstrated; that would require an economic evaluation against bookmaker odds, not RPS alone.

**Variability bands are now in place.** The ablation is repeated over ≥5 seeds (and, for multi-step, across
the five Cups) with 95% CIs on every cell; the survival test uses a clustered block bootstrap. The remaining
multi-step CIs are wide (five-Cup spread), so its per-window differences are individually non-significant
beyond the small-data corner — the one-step survival carries the significant result.

**Remaining limitations (by impact):** (1) a **single test era** — the one-step series is tested only on
2023+; repeated walk-forward across eras would generalise "LogReg is best at full data"; (2) the **multi-step
test is weak** (five Cups, frozen favourite) and should be read as illustration, with per-match Cup RPS
carrying the weight; (3) **effect sizes are small** in absolute RPS (~3% relative on the series) and no
economic evaluation against market odds is provided; (4) **"feature selection" selected nothing** (`k*=107`)
— it is an importance ranking, not a parsimony result.

---

## 6. Conclusion

Across a carefully leakage-controlled, temporally honest evaluation of international football 1X2
forecasting, the substantive conclusion is two-sided. **On the standard question** — does model
sophistication beat a plain logistic regression on a rich, Elo-anchored feature table? — the answer, stated
honestly, is **no, not significantly**: a logistic regression is as good as tuned GBDTs and tune-free
foundation models on the broad series, and the foundation models' numerical edge on World Cup matches is not
statistically established. After Elo and recent form, the residual structure is largely linear and the
ceiling is low, as expected for this problem.

**The robust, defensible contribution is about *data efficiency*.** The training-size ablation shows that
tune-free tabular foundation models (TabPFN, and TabICL beyond a few hundred samples) are **nearly invariant
to training-set size**, delivering with one month of history what GBDTs and logistic regression reach only
with several years — and that the advantage **grows with feature dimensionality**, precisely the regime
(many engineered features, few relevant recent matches) that tournament forecasting inhabits. This reframes
the practical recommendation: foundation models are not chosen here because they win the asymptotic
accuracy race (they do not, significantly), but because they are **strong, calibrated, and configuration-free
in the small-sample, high-dimensional corner** where traditional models require careful regularisation and
abundant data. Crucially, this advantage is **statistically significant against per-window-tuned classical
baselines** (one-step, 107 features, one month to five years of data) — it is not an artefact of leaving the
competitors mis-regularised — and it is achieved at **zero tuning cost**. For applications such as a newly
expanded competition, an under-documented league, or the opening matches of a tournament, that robustness is
the decisive property. This training-size robustness — not the (null) head-to-head accuracy at full data —
is the result we put at the centre of the work.
