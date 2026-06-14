"""NeurIPS-ready LaTeX tables — the single source of table output.

Takes a Polars DataFrame. Tables use ``booktabs`` rules, no vertical lines,
caption *above* the tabular, and consistent decimals.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import config


def _esc(s) -> str:
    return str(s).replace("\\", r"\textbackslash ").replace("_", r"\_").replace("%", r"\%")


def _fmt(series: pl.Series, float_format: str) -> list[str]:
    vals = series.to_list()
    if series.dtype.is_integer():
        return ["" if v is None else f"{int(v):d}" for v in vals]
    if series.dtype.is_numeric():
        return ["" if v is None else float_format % v for v in vals]
    return [_esc(v) for v in vals]


def df_to_neurips_latex(df: pl.DataFrame, caption: str, label: str,
                        float_format: str = "%.3f", bold_best: str | None = None,
                        lower_is_better: bool = True) -> str:
    """Render ``df`` as a NeurIPS ``table`` block (booktabs, caption above).

    ``bold_best`` bolds the optimal value of that column; ``lower_is_better``
    sets the direction (e.g. RPS / log-loss → True). State it in the caption.
    """
    cols = df.columns
    body = {c: _fmt(df[c], float_format) for c in cols}

    if bold_best and bold_best in cols:
        vals = df[bold_best].to_list()
        finite = [v for v in vals if v is not None]
        best = (min if lower_is_better else max)(finite) if finite else None
        body[bold_best] = [r"\textbf{%s}" % t if v == best else t
                           for t, v in zip(body[bold_best], vals)]

    align = "l" + "r" * (len(cols) - 1)
    header = " & ".join(_esc(c) for c in cols) + r" \\"
    rows = [" & ".join(body[c][i] for c in cols) + r" \\" for i in range(df.height)]
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{%s}" % caption, r"  \label{%s}" % label,
             r"  \begin{tabular}{%s}" % align, r"  \toprule",
             "  " + header, r"  \midrule",
             *["  " + r for r in rows],
             r"  \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def save_table(latex: str, name: str, outdir: Path | None = None) -> Path:
    """Single table-export point: write ``<name>.tex``."""
    outdir = Path(outdir) if outdir else config.TABLES
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.tex"
    path.write_text(latex, encoding="utf-8")
    return path
