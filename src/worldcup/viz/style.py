"""NeurIPS-ready Matplotlib style — the single source of figure styling.

Every figure in the project goes through :func:`set_style` and :func:`save_fig`;
no module should ever call ``plt.savefig`` directly. Figures are drawn at their
final size (text width = 5.5 in) and exported as vector PDF + 300-dpi PNG preview
with embedded fonts.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from .. import config

TEXTWIDTH_IN = 5.5   # NeurIPS single-column text width
HALFWIDTH_IN = 2.7

# Okabe–Ito colorblind-safe palette + secondary encodings for B&W legibility
CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
              "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v"]

NEURIPS_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.figsize": (TEXTWIDTH_IN, TEXTWIDTH_IN * 0.62),
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.prop_cycle": cycler(color=CB_PALETTE),
}


def set_style() -> None:
    """Apply the NeurIPS rcParams globally."""
    mpl.rcParams.update(NEURIPS_RC)


def style_for(i: int) -> dict:
    """Color + linestyle + marker for the i-th series (legible in grayscale)."""
    return {"color": CB_PALETTE[i % len(CB_PALETTE)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)],
            "marker": MARKERS[i % len(MARKERS)]}


def save_fig(fig: plt.Figure, name: str, outdir: Path | None = None) -> Path:
    """Single figure-export point: write ``<name>.pdf`` (vector) + ``.png`` (preview)."""
    outdir = Path(outdir) if outdir else config.FIGURES
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=300)
    plt.close(fig)
    return outdir / f"{name}.pdf"
