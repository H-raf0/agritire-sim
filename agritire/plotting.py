"""Style graphique commun aux notebooks (palette adaptée aux daltoniens)."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Palette Okabe–Ito (lisible par les personnes daltoniennes).
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
    })


def save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
