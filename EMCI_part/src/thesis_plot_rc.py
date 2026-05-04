"""
Thesis figure defaults: vector-friendly PDF + one consistent font scale.
Call apply_thesis_style() before building figures; avoid per-element fontsize= overrides.
"""

from __future__ import annotations

import matplotlib as mpl

THESIS_RC: dict = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "axes.titleweight": "normal",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
    "figure.titleweight": "normal",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_thesis_style() -> None:
    mpl.rcParams.update(THESIS_RC)
    try:
        import seaborn as sns
    except ImportError:
        return
    # Default sns.set_theme() uses darkgrid (grey axes); keep white like matplotlib-only plots
    sns.set_theme(style="whitegrid", rc=dict(THESIS_RC))


def as_pdf_filename(filename: str) -> str:
    """Force .pdf extension for saved figures."""
    if not filename:
        return "figure.pdf"
    p = filename.strip()
    lower = p.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".svg")):
        return p.rsplit(".", 1)[0] + ".pdf"
    if "." not in p:
        return p + ".pdf"
    return p
