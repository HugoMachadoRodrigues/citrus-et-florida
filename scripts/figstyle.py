"""Shared visual system for the manuscript figures.

One typeface, one ink scale, and fixed colour roles so that an entity keeps its colour in every
figure. Palettes were checked with the dataviz palette validator (lightness band, chroma,
colour-vision-deficiency separation for all pairs, normal-vision floor):

  regions    blue / orange / aqua (all-pairs CVD dE >= 9.2). Aqua is below 3:1 contrast on
             white, so region series are always labelled directly.
  ET0        two hue families: violet = gridMET (raw dark, bias-corrected light), amber = FAWN
             (wind-standardized dark = central scenario, measured wind light); all-pairs CVD
             dE >= 20. Light members are always labelled directly.
  ordinal    periods and soil terciles use an ink ramp (light grey -> near black), so they never
             borrow a region hue.

Every figure is saved as a 600-dpi PNG and a vector PDF with embedded TrueType fonts.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "figures"

# Ink and chrome.
INK, INK_2, MUTED = "#1a1a19", "#52514e", "#898781"
GRID, AXIS, WASH = "#e6e5df", "#b9b8b0", "#f1f0ec"
# Width of a full-page figure in the Agriculture template (text block ~16.5 cm).
FULL, HALF = 6.85, 3.35

REGION = {"Southwest Flatwoods": "#2a78d6", "Indian River": "#eb6834", "Ridge": "#1baf7a"}
REGION_SHORT = {"Southwest Flatwoods": "SW Flatwoods", "Indian River": "Indian River",
                "Ridge": "Ridge", "All": "All cells"}
ETO = {  # label -> (colour, line style, marker)
    "gridMET, raw": ("#4a3aa7", "-", "o"),
    "gridMET, OpenET bias-corrected": ("#9085e9", "-", "D"),
    "FAWN": ("#eda100", "-", "s"),
    "FAWN, wind-standardized": ("#a86400", "-", "^"),
}
ETO_SHORT = {"gridMET, raw": "gridMET raw", "gridMET, OpenET bias-corrected": "gridMET bias-corr.",
             "FAWN": "FAWN", "FAWN, wind-standardized": "FAWN wind-std."}
ORDINAL = ["#bdbbb3", "#7d7b75", "#1a1a19"]  # light -> dark
ACCENT = "#e34948"   # reference curves and highlighted events
HURRICANE_YEARS = {2004: "Charley, Frances, Jeanne", 2005: "Wilma", 2017: "Irma", 2022: "Ian",
                   2024: "Helene, Milton"}
HLB_YEAR = 2005.6


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial Unicode MS", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titleweight": "regular", "axes.titlelocation": "left",
        "axes.titlepad": 6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.major.size": 3, "ytick.major.size": 0,
        "xtick.major.width": 0.7, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.default": "regular", "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica Neue", "mathtext.it": "Helvetica Neue:italic",
        "mathtext.bf": "Helvetica Neue:bold", "mathtext.sf": "Helvetica Neue",
        "mathtext.cal": "Helvetica Neue", "mathtext.tt": "Helvetica Neue",
    })


def panel(ax, letter, title=None, sub=None):
    """Panel letter in bold, title in regular weight, optional muted subtitle below it."""
    ax.set_title("")
    up = 11 if sub else 3
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, up),
                textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color=INK,
                weight="bold", annotation_clip=False)
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(9, up),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8,
                    color=INK, annotation_clip=False)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2.5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=6.3,
                    color=MUTED, annotation_clip=False)


def label_ends(ax, items, x, min_gap, dx=0.4, fontsize=7, weight="regular"):
    """Direct labels at the right end of lines, nudged apart vertically.

    items: list of (y, text, colour). min_gap in data units."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = (np.mean([t[0] for t in items]) - np.mean(ys))
    ys = [y + shift for y in ys]
    for (y0, text, color), y in zip(items, ys):
        ax.annotate(text, xy=(x, y0), xytext=(x + dx, y), textcoords="data",
                    va="center", ha="left", fontsize=fontsize, color=INK, weight=weight,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                                    shrinkA=0, shrinkB=1.5) if abs(y - y0) > 1e-9 else None)
        ax.plot([x + dx * 0.15], [y], "o", ms=0)  # keeps autoscale aware of the label


def shade_events(ax, years=HURRICANE_YEARS, label=True, ytext=None, hlb=True):
    """Light bands for hurricane years and a hairline for the HLB confirmation."""
    for y, name in years.items():
        ax.axvspan(y - 0.5, y + 0.5, color=WASH, lw=0, zorder=0)
    if hlb:
        ax.axvline(HLB_YEAR, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)


def zero_line(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=AXIS, lw=0.8, zorder=1)


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    out = FIGS / name
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {out}.png and .pdf")


def block_ids(cells, km=20):
    return ((cells["x_albers"] // (km * 1000)).astype(int).astype(str) + "_"
            + (cells["y_albers"] // (km * 1000)).astype(int).astype(str))
