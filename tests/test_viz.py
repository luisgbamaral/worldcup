"""The viz layer builds figures and NeurIPS-compliant LaTeX tables."""
import polars as pl

from worldcup import viz
from worldcup.viz import plots


def test_all_figures_build(played, goals, editions, elo, latest):
    viz.set_style()
    figs = [
        plots.matches_per_year(played),
        plots.goals_per_match_trend(played),
        plots.goals_distribution(played),
        plots.home_advantage_by_decade(played),
        plots.goal_minute_distribution(goals),
        plots.worldcup_goals_per_match(editions),
        plots.top_elo(latest),
        plots.elo_history(elo, ["Brazil", "Spain"]),
    ]
    assert all(f is not None for f in figs)


def test_save_fig_writes_pdf_and_png(played, tmp_path):
    viz.set_style()
    viz.save_fig(plots.matches_per_year(played), "fig", outdir=tmp_path)
    assert (tmp_path / "fig.pdf").exists()
    assert (tmp_path / "fig.png").exists()


def test_table_is_neurips_compliant(editions):
    tex = viz.df_to_neurips_latex(editions.head(3), caption="c", label="t", float_format="%.2f")
    assert r"\toprule" in tex and r"\bottomrule" in tex
    colspec = tex.split("tabular}{")[1].split("}")[0]
    assert "|" not in colspec               # no vertical rules
    assert "1930 &" in tex                   # integers stay integers


def test_table_bolds_best():
    df = pl.DataFrame({"model": ["a", "b"], "rps": [0.20, 0.18]})
    tex = viz.df_to_neurips_latex(df, caption="c", label="t", float_format="%.2f",
                                  bold_best="rps", lower_is_better=True)
    assert r"\textbf{0.18}" in tex
