"""NeurIPS-ready LaTeX tables — the single source of table output.

No module should call ``df.to_latex`` directly. Tables use ``booktabs`` rules,
no vertical lines, caption *above* the tabular, and consistent decimals.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config


def _esc(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash ").replace("_", r"\_").replace("%", r"\%")


def _fmt(col: pd.Series, float_format: str) -> list[str]:
    if pd.api.types.is_integer_dtype(col):
        return ["" if pd.isna(v) else f"{int(v):d}" for v in col]
    if pd.api.types.is_numeric_dtype(col):
        return ["" if pd.isna(v) else float_format % v for v in col]
    return [_esc(v) for v in col]


def df_to_neurips_latex(df: pd.DataFrame, caption: str, label: str,
                        float_format: str = "%.3f", bold_best: str | None = None,
                        lower_is_better: bool = True, index: bool = False) -> str:
    """Render ``df`` as a NeurIPS ``table`` block (booktabs, caption above).

    ``bold_best`` bolds the optimal value of that column; ``lower_is_better``
    sets the direction (e.g. RPS / log-loss → True). State the metric direction
    in the caption.
    """
    d = df.reset_index() if index else df.copy()
    cols = list(d.columns)
    body = {c: _fmt(d[c], float_format) for c in cols}

    if bold_best and bold_best in d.columns:
        vals = pd.to_numeric(d[bold_best], errors="coerce")
        best = vals.min() if lower_is_better else vals.max()
        body[bold_best] = [r"\textbf{%s}" % t if v == best else t
                           for t, v in zip(body[bold_best], vals)]

    align = "l" + "r" * (len(cols) - 1)
    header = " & ".join(_esc(c) for c in cols) + r" \\"
    rows = [" & ".join(body[c][i] for c in cols) + r" \\" for i in range(len(d))]
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
