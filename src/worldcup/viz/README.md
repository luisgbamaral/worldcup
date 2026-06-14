# `worldcup.viz` — visual standard

The **only** place figures and tables are produced, so the whole project keeps a
single NeurIPS-ready look. Never call `plt.savefig` or `df.to_latex` elsewhere.

## Figures

```python
from worldcup import viz
from worldcup.viz import plots

viz.set_style()                                  # NeurIPS rcParams (once)
fig = plots.goals_per_match_trend(played)        # any plot → Figure at final size
viz.save_fig(fig, "02_goals_per_match")          # → reports/figures/<name>.pdf + .png
```

- Text width **5.5 in**; figures drawn at final size (never shrunk).
- Vector **PDF** (primary) + 300-dpi **PNG** (preview), fonts embedded (`pdf.fonttype=42`).
- Colorblind-safe palette **plus** linestyle/marker (`style_for(i)`) → legible in B&W.
- Figure captions go *below* the figure in the paper and must end with a key takeaway.

## Tables

```python
tex = viz.df_to_neurips_latex(
    df, caption="... (lower is better).", label="tab:rps",
    float_format="%.3f", bold_best="rps", lower_is_better=True)
viz.save_table(tex, "rps")                        # → reports/tables/rps.tex
```

- `booktabs` rules, **no vertical lines**, caption *above*, consistent decimals.
- Integer columns render as integers; `bold_best` highlights the optimum per the
  metric direction. Report `mean ± std` for repeated runs/folds.
