"""Visualization layer: the only place figures and tables are produced.

Style (`style`), figures (`plots`) and LaTeX tables (`tables`) all live here so
the project keeps a single, NeurIPS-consistent visual standard.
"""
from . import plots, tables
from .style import (CB_PALETTE, HALFWIDTH_IN, TEXTWIDTH_IN,
                    save_fig, set_style, style_for)
from .tables import df_to_neurips_latex, save_table

__all__ = [
    "plots", "tables", "set_style", "save_fig", "style_for",
    "df_to_neurips_latex", "save_table",
    "CB_PALETTE", "TEXTWIDTH_IN", "HALFWIDTH_IN",
]
