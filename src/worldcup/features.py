"""Consolidated, model-ready feature base for 2026 World Cup forecasting.

Builds three Parquet outputs in ``data/processed/`` (base only — no modelling):

* ``match_features.parquet``      — one row per match: 2018+ historical played
  matches (training) **plus** every 2026 fixture (scored directly).
* ``team_match_features.parquet`` — long, two rows per match (per-team goals model).
* ``player_features.parquet``     — per-(team, player) historical scorer priors.

Leakage rule (enforced everywhere and tested): every rolling / form / rest /
head-to-head feature for a match uses only that team's matches *strictly before*
the match date — implemented with ``shift``. Elo ``*_pre`` are pre-match by
construction. History (Elo, form windows) is computed over the **full** record
1872→2026; the 2018 cut is applied only to the final supervised training rows.
The builder is stateless: a daily rebuild on freshly pulled data is current.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from . import config, data, elo, lookups
from .clean import _name_key, canon_team

TRAIN_CUT = dt.date(2018, 1, 1)
WINDOWS = (5, 10)
MOM_K = 10          # window for Elo momentum
TOP5 = {"ENG", "ESP", "ITA", "GER", "FRA"}
GOAL_RATE_ALPHA, GOAL_RATE_BETA = 1.0, 5.0  # smoothing for player goal rate

# per-team feature columns that expand into _home / _away / _diff
_FORM = [f"{m}_{k}" for k in WINDOWS
         for m in ("ppg", "gf_pg", "ga_pg", "gd_pg", "win_rate", "cs_rate", "over25_rate")]
TEAM_FEATS = _FORM + ["adj_ppg_10", "adj_gd_10", "elo_momentum",
                      "rest_days", "matches_30d", "matches_90d"]
SQUAD_FEATS = ["squad_avg_age", "squad_avg_height", "squad_def_height", "share_abroad",
               "share_top5_league", "star_caps", "star_goals", "goal_concentration"]


# --------------------------------------------------------------------------- #
# spine + long log
# --------------------------------------------------------------------------- #
def _match_id() -> pl.Expr:
    return pl.concat_str([pl.col("date").cast(pl.Utf8),
                          pl.col("home_team"), pl.col("away_team")], separator="__").hash()


def _spine() -> pl.DataFrame:
    """Full played history with Elo, canonical names, targets and match_id."""
    s = elo.rate_matches().with_columns(
        canon_team("home_team").alias("home_team"),
        canon_team("away_team").alias("away_team"),
        canon_team("country").alias("country"))
    return s.with_columns(
        _match_id().alias("match_id"),
        (pl.col("home_score") + pl.col("away_score")).alias("total_goals"),
        (pl.col("home_score") - pl.col("away_score")).alias("goal_diff"),
        pl.when(pl.col("home_score") > pl.col("away_score")).then(pl.lit("H"))
          .when(pl.col("home_score") == pl.col("away_score")).then(pl.lit("D"))
          .otherwise(pl.lit("A")).alias("result"),
        pl.lit(True).alias("played"))


def _long(spine: pl.DataFrame) -> pl.DataFrame:
    """Two rows per match (one per team), with own/opponent Elo."""
    home = spine.select("match_id", "date", "tournament", "country",
                        team="home_team", opponent="away_team",
                        is_home=pl.lit(True), neutral="neutral",
                        gf="home_score", ga="away_score",
                        elo_pre="home_elo_pre", elo_post="home_elo_post",
                        opp_elo_pre="away_elo_pre")
    away = spine.select("match_id", "date", "tournament", "country",
                        team="away_team", opponent="home_team",
                        is_home=pl.lit(False), neutral="neutral",
                        gf="away_score", ga="home_score",
                        elo_pre="away_elo_pre", elo_post="away_elo_post",
                        opp_elo_pre="home_elo_pre")
    return pl.concat([home, away]).sort("team", "date")


def _form_exprs(shift_n: int) -> list[pl.Expr]:
    """Rolling form + Elo momentum, shifted by ``shift_n`` (1=exclude current row)."""
    def roll(col, k):
        return pl.col(col).shift(shift_n).rolling_mean(k, min_samples=1).over("team")

    out = []
    for k in WINDOWS:
        out += [roll("points", k).alias(f"ppg_{k}"),
                roll("gf", k).alias(f"gf_pg_{k}"),
                roll("ga", k).alias(f"ga_pg_{k}"),
                roll("gd", k).alias(f"gd_pg_{k}"),
                roll("win", k).alias(f"win_rate_{k}"),
                roll("cs", k).alias(f"cs_rate_{k}"),
                roll("over25", k).alias(f"over25_rate_{k}")]
    # strength-adjusted (opponent-Elo weighted) form over last 10
    wsum = pl.col("opp_elo_pre").shift(shift_n).rolling_sum(10, min_samples=1).over("team")
    out += [
        (pl.col("points_w").shift(shift_n).rolling_sum(10, min_samples=1).over("team")
         / wsum).alias("adj_ppg_10"),
        (pl.col("gd_w").shift(shift_n).rolling_sum(10, min_samples=1).over("team")
         / wsum).alias("adj_gd_10"),
        (pl.col("elo_post").shift(shift_n)
         - pl.col("elo_post").shift(shift_n + MOM_K)).over("team").alias("elo_momentum"),
    ]
    return out


def _add_form(long: pl.DataFrame, shift_n: int) -> pl.DataFrame:
    long = long.with_columns(
        (3 * (pl.col("gf") > pl.col("ga")).cast(pl.Int32)
         + (pl.col("gf") == pl.col("ga")).cast(pl.Int32)).alias("points"),
        (pl.col("gf") - pl.col("ga")).alias("gd"),
        (pl.col("gf") > pl.col("ga")).cast(pl.Int32).alias("win"),
        (pl.col("ga") == 0).cast(pl.Int32).alias("cs"),
        ((pl.col("gf") + pl.col("ga")) > 2.5).cast(pl.Int32).alias("over25"))
    long = long.with_columns(
        (pl.col("points") * pl.col("opp_elo_pre")).alias("points_w"),
        (pl.col("gd") * pl.col("opp_elo_pre")).alias("gd_w"))
    return long.with_columns(_form_exprs(shift_n))


def _rest_counts(long: pl.DataFrame) -> pl.DataFrame:
    """rest_days, matches_30d, matches_90d per team-match (strictly earlier only)."""
    base = long.with_row_index("row_id").select("row_id", "team", "date")
    j = (base.join(base.select("team", d2="date"), on="team")
         .filter(pl.col("d2") < pl.col("date")))
    agg = j.group_by("row_id").agg(
        matches_30d=(pl.col("d2") >= pl.col("date").dt.offset_by("-30d")).sum(),
        matches_90d=(pl.col("d2") >= pl.col("date").dt.offset_by("-90d")).sum(),
        _last=pl.col("d2").max())
    out = base.join(agg, on="row_id", how="left").with_columns(
        (pl.col("date") - pl.col("_last")).dt.total_days().alias("rest_days"),
        pl.col("matches_30d").fill_null(0), pl.col("matches_90d").fill_null(0))
    return long.with_columns(
        out["rest_days"], out["matches_30d"], out["matches_90d"])


def _h2h(spine: pl.DataFrame) -> pl.DataFrame:
    """Head-to-head priors per match (strictly earlier meetings of the pair)."""
    lo = pl.col("home_team") < pl.col("away_team")
    s = spine.with_columns(
        pl.when(lo).then(pl.col("home_team")).otherwise(pl.col("away_team")).alias("t1"),
        pl.when(lo).then(pl.col("away_team")).otherwise(pl.col("home_team")).alias("t2"))
    s = s.with_columns(
        pl.concat_str("t1", "t2", separator="|").alias("pair"),
        pl.when(((pl.col("home_team") == pl.col("t1")) & (pl.col("home_score") > pl.col("away_score")))
                | ((pl.col("away_team") == pl.col("t1")) & (pl.col("away_score") > pl.col("home_score"))))
          .then(1).otherwise(0).alias("t1_win"),
        (pl.col("home_score") == pl.col("away_score")).cast(pl.Int32).alias("is_draw"),
    ).sort("pair", "date")
    s = s.with_columns(
        pl.int_range(0, pl.len()).over("pair").alias("h2h_n"),
        pl.col("t1_win").cum_sum().shift(1).over("pair").fill_null(0).alias("_t1w"),
        pl.col("is_draw").cum_sum().shift(1).over("pair").fill_null(0).alias("_dr"),
        pl.col("total_goals").cum_sum().shift(1).over("pair").alias("_gsum"),
        pl.col("date").shift(1).over("pair").alias("_last"))
    return s.select(
        "match_id", "h2h_n",
        pl.when(pl.col("h2h_n") > 0).then(
            pl.when(pl.col("home_team") == pl.col("t1")).then(pl.col("_t1w"))
              .otherwise(pl.col("h2h_n") - pl.col("_t1w") - pl.col("_dr"))
            / pl.col("h2h_n")).alias("h2h_home_winrate"),
        (pl.col("_gsum") / pl.col("h2h_n")).alias("h2h_avg_total_goals"),
        (pl.col("date") - pl.col("_last")).dt.total_days().alias("days_since_last_meeting"))


# --------------------------------------------------------------------------- #
# expansion helpers
# --------------------------------------------------------------------------- #
def _expand(spine: pl.DataFrame, team_feat: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Join per-team feature cols onto the match grain as _home/_away (+ _diff)."""
    home = team_feat.filter(pl.col("is_home")).select(
        "match_id", *[pl.col(c).alias(f"{c}_home") for c in cols])
    away = team_feat.filter(~pl.col("is_home")).select(
        "match_id", *[pl.col(c).alias(f"{c}_away") for c in cols])
    out = spine.join(home, on="match_id", how="left").join(away, on="match_id", how="left")
    return out.with_columns(
        [(pl.col(f"{c}_home") - pl.col(f"{c}_away")).alias(f"{c}_diff") for c in cols])


def _elo_context(df: pl.DataFrame) -> pl.DataFrame:
    """Elo diff/sum + tournament context (neutral-independent parts)."""
    k0 = {t: elo._k0(t) for t in df["tournament"].unique().to_list()}
    return df.with_columns(
        (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("elo_diff"),
        (pl.col("home_elo_pre") + pl.col("away_elo_pre")).alias("elo_sum"),
        pl.col("tournament").replace_strict(k0, default=30).alias("tournament_importance"),
        (pl.col("tournament") != "Friendly").alias("is_competitive"),
        (pl.col("tournament") == "FIFA World Cup").alias("is_world_cup"))


def _elo_effective(df: pl.DataFrame) -> pl.DataFrame:
    """Home-advantage-adjusted Elo diff + win prob (needs finalized ``neutral``)."""
    return df.with_columns(
        (pl.col("elo_diff") + pl.when(pl.col("neutral")).then(0).otherwise(100)).alias("elo_diff_eff"),
    ).with_columns(
        (1 / (1 + 10 ** (-pl.col("elo_diff_eff") / 400))).alias("elo_winprob"))


# --------------------------------------------------------------------------- #
# squad features (2026 snapshot only)
# --------------------------------------------------------------------------- #
def _squad_features() -> pl.DataFrame:
    teams = data.load_squad_players().with_columns(canon_team("team").alias("team"))
    defs = (teams.filter(pl.col("position") == "DF")
            .group_by("team").agg(squad_def_height=pl.col("height_cm").mean()))
    agg = teams.group_by("team").agg(
        squad_avg_age=pl.col("age_at_tournament").mean(),
        squad_avg_height=pl.col("height_cm").mean(),
        share_abroad=(pl.col("club_country") != pl.col("team_code")).mean(),
        share_top5_league=pl.col("club_country").is_in(TOP5).mean(),
        star_caps=pl.col("caps").max(),
        star_goals=pl.col("goals").max(),
        goal_concentration=pl.col("goals").max() / pl.col("goals").sum())
    return agg.join(defs, on="team", how="left")


# --------------------------------------------------------------------------- #
# winsorization (fit on train only, never targets)
# --------------------------------------------------------------------------- #
def _winsorize(df: pl.DataFrame, cols: list[str], train_mask: pl.Series) -> pl.DataFrame:
    train = df.filter(train_mask)
    exprs = []
    for c in cols:
        if c not in df.columns or train[c].null_count() == train.height:
            continue
        lo, hi = train[c].quantile(0.01), train[c].quantile(0.99)
        if lo is None or hi is None:
            continue
        exprs.append(pl.col(c).clip(lo, hi).alias(c))
    return df.with_columns(exprs)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def _fixtures() -> pl.DataFrame:
    return (data.load_fixtures()
            .with_columns(canon_team("home_team").alias("home_team"),
                          canon_team("away_team").alias("away_team"),
                          pl.col("match_date").alias("date"))
            .with_columns(_match_id().alias("match_id"), pl.lit("group").alias("stage"),
                          pl.col("home_score").is_not_null().alias("played")))


def _pending_team_long(fixtures: pl.DataFrame, long_base: pl.DataFrame) -> pl.DataFrame:
    """Per-(fixture, team) features for pending 2026 games (current snapshot)."""
    snap = _add_form(long_base, 0).group_by("team", maintain_order=True).last()
    form_cols = [c for c in TEAM_FEATS if c not in ("rest_days", "matches_30d", "matches_90d")]
    plong = pl.concat([
        fixtures.select("match_id", "date", team="home_team", is_home=pl.lit(True)),
        fixtures.select("match_id", "date", team="away_team", is_home=pl.lit(False))])
    plong = plong.join(snap.select("team", *form_cols), on="team", how="left")
    pdates = long_base.select("team", d2="date")
    rc = (plong.select("match_id", "is_home", "team", "date")
          .join(pdates, on="team", how="left").filter(pl.col("d2") < pl.col("date"))
          .group_by("match_id", "is_home").agg(
              _date=pl.col("date").first(), _last=pl.col("d2").max(),
              matches_30d=(pl.col("d2") >= pl.col("date").first().dt.offset_by("-30d")).sum(),
              matches_90d=(pl.col("d2") >= pl.col("date").first().dt.offset_by("-90d")).sum())
          .with_columns((pl.col("_date") - pl.col("_last")).dt.total_days().alias("rest_days")))
    return plong.join(rc.select("match_id", "is_home", "rest_days", "matches_30d", "matches_90d"),
                      on=["match_id", "is_home"], how="left")


def _pending_spine(fixtures: pl.DataFrame, pending_ids: list) -> pl.DataFrame:
    lr = (elo.latest_ratings().with_columns(canon_team("team").alias("team"))
          .select("team", "rating"))
    pend = fixtures.filter(pl.col("match_id").is_in(pending_ids)).with_columns(
        pl.lit("FIFA World Cup").alias("tournament"), pl.lit(False).alias("played"),
        pl.lit(None, pl.Int64).alias("home_score"), pl.lit(None, pl.Int64).alias("away_score"),
        pl.lit(None, pl.Int64).alias("total_goals"), pl.lit(None, pl.Int64).alias("goal_diff"),
        pl.lit(None, pl.Utf8).alias("result"), pl.lit(None, pl.Utf8).alias("country"))
    return (pend
            .join(lr.rename({"rating": "home_elo_pre"}), left_on="home_team", right_on="team", how="left")
            .join(lr.rename({"rating": "away_elo_pre"}), left_on="away_team", right_on="team", how="left"))


def _venue_home(df: pl.DataFrame) -> pl.DataFrame:
    """is_home_country (all rows) + quasi-home gradient (2026 rows only)."""
    cf = lambda c: pl.col(c).replace_strict(lookups.CONFEDERATION, default=None)
    df = df.with_columns(
        pl.col("venue").map_elements(lookups.venue_country, return_dtype=pl.Utf8).alias("_vcountry"),
        pl.col("venue").map_elements(lookups.venue_altitude, return_dtype=pl.Float64).alias("venue_altitude_m"))
    df = df.with_columns(
        pl.when(pl.col("is_2026")).then(pl.col("home_team") == pl.col("_vcountry"))
          .otherwise(pl.col("home_team") == pl.col("country")).alias("is_home_country_home"),
        pl.when(pl.col("is_2026")).then(pl.col("away_team") == pl.col("_vcountry"))
          .otherwise(pl.col("away_team") == pl.col("country")).alias("is_home_country_away"))
    q = lambda e: pl.when(pl.col("is_2026")).then(e).otherwise(None)
    return df.with_columns(
        pl.when(pl.col("is_2026"))
          .then(~(pl.col("is_home_country_home") | pl.col("is_home_country_away")))
          .otherwise(pl.col("neutral")).alias("neutral"),
        q(cf("home_team") == lookups.HOST_CONFED_2026).alias("host_confed_match_home"),
        q(cf("away_team") == lookups.HOST_CONFED_2026).alias("host_confed_match_away"),
        q(pl.col("home_team").map_elements(lookups.host_proximity, return_dtype=pl.Float64)).alias("host_proximity_home"),
        q(pl.col("away_team").map_elements(lookups.host_proximity, return_dtype=pl.Float64)).alias("host_proximity_away"),
    ).drop("_vcountry")


def _add_squad(df: pl.DataFrame) -> pl.DataFrame:
    sf = _squad_features()
    sh = sf.rename({c: f"{c}_home" for c in SQUAD_FEATS} | {"team": "home_team"})
    sa = sf.rename({c: f"{c}_away" for c in SQUAD_FEATS} | {"team": "away_team"})
    df = df.join(sh, on="home_team", how="left").join(sa, on="away_team", how="left")
    df = df.with_columns([(pl.col(f"{c}_home") - pl.col(f"{c}_away")).alias(f"{c}_diff")
                          for c in SQUAD_FEATS])
    sides = [f"{c}_{s}" for c in SQUAD_FEATS for s in ("home", "away", "diff")]
    return df.with_columns([pl.when(pl.col("is_2026")).then(pl.col(c)).otherwise(None).alias(c)
                            for c in sides])


def build_match_features(train_cut: dt.date = TRAIN_CUT) -> pl.DataFrame:
    """``train_cut`` keeps historical rows on/after this date (features are always
    computed over the full record; only the final filter moves). The World-Cup
    evaluation lowers it to reach 8-year windows before 2018."""
    spine = _spine()
    long_base = _long(spine)
    hist = _elo_context(_expand(spine, _rest_counts(_add_form(long_base, 1)), TEAM_FEATS))

    fixtures = _fixtures()
    fix_ids = fixtures["match_id"].to_list()
    hist_ids = set(hist["match_id"].to_list())
    pend_ids = [i for i in fix_ids if i not in hist_ids]

    hist = (hist.join(fixtures.select("match_id", "stage", "venue"), on="match_id", how="left")
            .with_columns(pl.col("match_id").is_in(fix_ids).alias("is_2026")))

    pend_spine = _pending_spine(fixtures, pend_ids)
    plong = _pending_team_long(fixtures.filter(pl.col("match_id").is_in(pend_ids)), long_base)
    pend = _elo_context(_expand(pend_spine, plong, TEAM_FEATS)).with_columns(
        pl.lit(True).alias("is_2026"))

    keep_hist = hist.filter(((~pl.col("is_2026")) & (pl.col("date") >= train_cut))
                            | pl.col("is_2026"))
    mf = pl.concat([keep_hist, pend], how="diagonal_relaxed")

    combined = pl.concat([
        spine.select("match_id", "date", "home_team", "away_team",
                     "home_score", "away_score", "total_goals"),
        pend_spine.select("match_id", "date", "home_team", "away_team",
                          "home_score", "away_score", "total_goals")])
    mf = mf.join(_h2h(combined), on="match_id", how="left")

    mf = _elo_effective(_add_squad(_venue_home(mf)))

    # winsorize the per-team levels (+ standalone predictors); recompute the
    # differentials afterwards so `X_diff == X_home - X_away` holds exactly.
    levels = ([f"{c}_{s}" for c in TEAM_FEATS + SQUAD_FEATS for s in ("home", "away")]
              + ["elo_sum", "h2h_avg_total_goals", "days_since_last_meeting"])
    mf = _winsorize(mf, levels, ~mf["is_2026"])
    mf = mf.with_columns([(pl.col(f"{c}_home") - pl.col(f"{c}_away")).alias(f"{c}_diff")
                          for c in TEAM_FEATS + SQUAD_FEATS])
    mf = mf.with_columns((pl.col("stage") == "knockout").fill_null(False).alias("is_knockout"))
    return mf.sort("date", "match_id")


def build_team_match_features(mf: pl.DataFrame) -> pl.DataFrame:
    """Long: two rows per match (per team) for the goals model."""
    shared = ["match_id", "date", "tournament", "neutral", "is_2026", "played",
              "tournament_importance", "is_competitive", "is_world_cup", "is_knockout",
              "venue_altitude_m"]
    team_side = TEAM_FEATS + ["is_home_country", "host_confed_match", "host_proximity"]
    rows = []
    for home in (True, False):
        s, o = ("home", "away") if home else ("away", "home")
        rows.append(mf.select(
            *shared,
            pl.col(f"{s}_team").alias("team"),
            pl.col(f"{o}_team").alias("opponent"),
            pl.lit(home).alias("is_home"),
            pl.col(f"{s}_score").alias("gf"),
            pl.col(f"{s}_elo_pre").alias("team_elo_pre"),
            pl.col(f"{o}_elo_pre").alias("opp_elo_pre"),
            *[pl.col(f"{c}_{s}").alias(c) for c in team_side],
        ).with_columns((pl.col("team_elo_pre") - pl.col("opp_elo_pre")).alias("elo_diff")))
    return pl.concat(rows).sort("date", "match_id", "is_home")


def build_player_features() -> pl.DataFrame:
    """Per-(team, player) scorer priors for the 2026 squads.

    Lightweight by design: no historical rosters exist, so this is a priors table
    keyed on the *current* 48 squads, not a per-match player-supervised table.
    """
    sq = data.load_squad_players().with_columns(
        canon_team("team").alias("team"),
        (pl.col("first_names").str.split(" ").list.first() + " "
         + pl.col("last_names")).alias("_rk"))
    sq = sq.with_columns(_name_key("_rk").alias("name_key"))

    g = data.load_goalscorers().with_columns(
        canon_team("team").alias("team"), _name_key("scorer").alias("name_key"))
    pens = (g.filter(pl.col("penalty")).select("team", "name_key").unique()
            .with_columns(pl.lit(True).alias("is_penalty_taker")))

    played = data.load_results(played_only=True).with_columns(
        canon_team("home_team").alias("home_team"), canon_team("away_team").alias("away_team"))
    tdates = pl.concat([played.select(team="home_team", date="date"),
                        played.select(team="away_team", date="date")]).unique()
    last10 = tdates.sort("date", descending=True).group_by("team").head(10)
    recent = (g.join(last10, on=["team", "date"], how="inner")
              .group_by("team", "name_key").agg(recent_goal_form=pl.len()))

    return (sq.select("team", "name_key", "player_name", "position", "caps",
                      career_goals=pl.col("goals"))
            .join(pens, on=["team", "name_key"], how="left")
            .join(recent, on=["team", "name_key"], how="left")
            .with_columns(
                pl.col("is_penalty_taker").fill_null(False),
                pl.col("recent_goal_form").fill_null(0),
                ((pl.col("career_goals") + GOAL_RATE_ALPHA)
                 / (pl.col("caps") + GOAL_RATE_BETA)).alias("goal_rate"))
            .sort("team", "career_goals", descending=[False, True]))


def main() -> int:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    mf = build_match_features()
    tmf = build_team_match_features(mf)
    pf = build_player_features()
    mf.write_parquet(config.MATCH_FEATURES)
    tmf.write_parquet(config.TEAM_MATCH_FEATURES)
    pf.write_parquet(config.PLAYER_FEATURES)
    print(f"match_features.parquet:      {mf.height} rows x {mf.width} cols "
          f"({mf['is_2026'].sum()} are 2026)")
    print(f"team_match_features.parquet: {tmf.height} rows x {tmf.width} cols")
    print(f"player_features.parquet:     {pf.height} rows x {pf.width} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
