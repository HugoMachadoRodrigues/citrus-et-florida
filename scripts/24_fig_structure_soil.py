"""Figure 8: grove canopy (NAIP index), soil water storage and the recent ET change.

(a) NAIP canopy index by region at the three winter acquisitions (citrus-area weighted), with the
    flight months.
(b) Change in ET (2015-2019 -> 2021-2025) against the change in the canopy index (2017/2019 ->
    2023): cells (faint), binned means with 95% bootstrap intervals (bins with >= 15 cells), and
    the cell regression of Equation (6) at average soil water storage (AWS). Bin means are
    citrus-area weighted; their intervals resample 20-km blocks of cells within the bin.
(c) ET change not explained by the canopy index (intercept of Equation (6)) by ET window and
    threshold rule, Ridge and Southwest Flatwoods (Table 8).
(d) ET change in cells with a small change in the canopy index (<= 5 pp) by tercile of AWS
    (terciles defined across all cells and applied within each region).
Intervals in (b)-(d): 95% spatial block bootstrap (20-km blocks; (c) and (d) from script 23).

Reads  data/processed/naip_canopy_cells.csv, structure_soil_cells_naip.csv,
       structure_soil_regression_naip.csv, structure_soil_strata_naip.csv, citrus_cells_1km.csv
Writes figures/fig_structure_soil.png and .pdf
"""

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs

RNG = np.random.default_rng(3)
TWO = ["Ridge", "Southwest Flatwoods"]
WINDOWS = [("recent", "2015–19 to 2021–25\n(main)"), ("aligned", "2017–19 to 2023"),
           ("to2023", "2015–19 to 2021–23")]
RULES = [("cell", "o", "per cell-year"), ("year", "s", "per year"),
         ("year_region", "D", "per year × region")]


def block_mean(b, nboot=1000):
    """Citrus-area-weighted mean ET change of a bin, with a 20-km block bootstrap interval."""
    g = b.assign(wy=b["citrus_ha"] * b["dln_et"]).groupby("block")[["wy", "citrus_ha"]].sum()
    wy, w = g["wy"].values, g["citrus_ha"].values
    cnt = np.stack([np.bincount(RNG.integers(0, len(w), len(w)), minlength=len(w))
                    for _ in range(nboot)])
    boots = (cnt @ wy) / (cnt @ w)
    return wy.sum() / w.sum(), *np.percentile(boots, [2.5, 97.5])


def main():
    fs.setup()
    spec = importlib.util.spec_from_file_location("s03", fs.ROOT / "scripts" / "03_summarize_series.py")
    s03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s03)
    cl = pd.read_csv(fs.PROC / "citrus_cells_1km.csv")
    cl["region"] = cl["county"].map(s03.region_of)
    cl["block"] = fs.block_ids(cl)
    nc = pd.read_csv(fs.PROC / "naip_canopy_cells.csv").merge(cl[["cell_id", "region", "citrus_ha"]],
                                                             on="cell_id")
    c = pd.read_csv(fs.PROC / "structure_soil_cells_naip.csv")
    c = c.drop(columns=[k for k in ("block", "citrus_ha") if k in c]).merge(
        cl[["cell_id", "block", "citrus_ha"]], on="cell_id")
    reg = pd.read_csv(fs.PROC / "structure_soil_regression_naip.csv")
    reg = reg[(reg["aws"] == "aws0_100") & (reg["response"] == "dln_et")]
    st = pd.read_csv(fs.PROC / "structure_soil_strata_naip.csv").query("period == 'recent'")

    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL, 5.4),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.32, "width_ratios": [1, 1.15]})
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (a) Canopy index trajectories.
    ends = []
    for r in ["Indian River", "Southwest Flatwoods", "Ridge"]:
        g = nc[nc["region"] == r].groupby("year").apply(
            lambda x: 100 * np.average(x["canopy_frac"], weights=x["citrus_ha"]), include_groups=False)
        ax1.plot(g.index, g.values, color=fs.REGION[r], lw=1.7, marker="o", ms=4, mec="white",
                 mew=0.6, zorder=3)
        ends.append((g.iloc[-1], fs.REGION_SHORT[r], fs.REGION[r]))
    fs.label_ends(ax1, ends, 2023.15, min_gap=2.2, dx=0.35, fontsize=6.8)
    ax1.set_xlim(2016.5, 2025.6)
    ax1.spines["bottom"].set_bounds(2016.5, 2023.5)
    ax1.set_xticks([2017, 2019, 2023])
    ax1.set_xticklabels(["2017\nOct", "2019\nNov–Dec", "2023\nJan–Mar"])
    ax1.set_ylabel("Canopy index (% of grove area)")
    fs.panel(ax1, "a", "NAIP canopy index", "winter acquisitions; per-image Otsu NDVI threshold")

    # (b) ET change vs canopy change.
    bins = np.arange(-35, 16, 5)
    for r in TWO:
        g = c[c["region"] == r]
        col = fs.REGION[r]
        ax2.scatter(g["dnaip"], g["dln_et"], s=2.2, color=col, alpha=0.18, lw=0, zorder=1)
        g = g.assign(bin=pd.cut(g["dnaip"], bins))
        for _, b in g.groupby("bin", observed=True):
            if len(b) < 15:
                continue
            est, lo, hi = block_mean(b)
            x = np.average(b["dnaip"], weights=b["citrus_ha"])
            ax2.plot([x, x], [lo, hi], color=col, lw=1.0, zorder=3)
            ax2.scatter(x, est, s=18, color=col, edgecolor="white", lw=0.5, zorder=4)
        q = reg[(reg["period"] == "recent") & (reg["rule"] == "cell") & (reg["group"] == r)].iloc[0]
        xx = np.array([g["dnaip"].quantile(0.03), g["dnaip"].quantile(0.97)])
        ax2.plot(xx, q["intercept"] + q["per_pp_tcc"] * xx, color=col, lw=1.0, ls=(0, (4, 2)), zorder=2)
    fs.zero_line(ax2)
    ax2.axvline(0, color=fs.AXIS, lw=0.8, zorder=1)
    ax2.set_xlim(-40, 18)
    ax2.set_ylim(-22, 12)
    ax2.grid(axis="x", color=fs.GRID, lw=0.6)
    ax2.set_xlabel("Change in canopy index, 2017/19 to 2023 (pp)")
    ax2.set_ylabel("Change in ET, 2015–19 to\n2021–25 (100 × Δln)")
    ax2.annotate("Ridge: ET lower\nat zero canopy change", xy=(0, -4.5), xytext=(3, -15),
                 fontsize=6, color=fs.INK_2,
                 arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6))
    h = [Line2D([], [], marker="o", ls="", color=fs.REGION[r], mec="white", ms=5,
                label=fs.REGION_SHORT[r]) for r in TWO]
    h.append(Line2D([], [], color=fs.MUTED, ls=(0, (4, 2)), lw=1, label="Eq. (6) fit"))
    ax2.legend(handles=h, loc="upper left", fontsize=6.3, handletextpad=0.3)
    fs.panel(ax2, "b", "ET follows the canopy index", "dots: cells; markers: 5-pp bins, 95% block-bootstrap interval")

    # (c) Residual ET change: window x threshold rule.
    yticks, ylabels = [], []
    y = 0
    for w, wl in WINDOWS:
        for ru, mk, _ in RULES:
            for k, r in enumerate(TWO):
                q = reg[(reg["period"] == w) & (reg["rule"] == ru) & (reg["group"] == r)].iloc[0]
                yy = y + (k - 0.5) * 0.28
                col = fs.REGION[r]
                ax3.plot([q["int_lo"], q["int_hi"]], [yy, yy], color=col, lw=1.0, alpha=0.6,
                         solid_capstyle="butt")
                ax3.plot(q["intercept"], yy, mk, ms=4, color=col, mec="white", mew=0.5)
            y += 1
        yticks.append(y - 2)
        ylabels.append(wl)
        y += 0.6
    for yb in (2.8, 6.4):
        ax3.axhline(yb, color=fs.GRID, lw=0.8)
    fs.zero_line(ax3, "x")
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(ylabels, fontsize=6.6)
    ax3.tick_params(axis="y", length=0)
    ax3.set_ylim(y - 0.4, -0.7)
    ax3.grid(axis="y", visible=False)
    ax3.grid(axis="x", color=fs.GRID, lw=0.6)
    ax3.set_xlabel("ET change at zero canopy change (100 × Δln)")
    h = [Line2D([], [], marker=mk, ls="", color=fs.INK_2, mec="white", ms=4.5, label=lab)
         for _, mk, lab in RULES]
    ax3.legend(handles=h, title="Otsu threshold:", title_fontsize=6.2, fontsize=6.2, ncol=4,
               loc="upper center", bbox_to_anchor=(0.42, -0.2), handletextpad=0.15,
               columnspacing=0.8, alignment="left")
    fs.panel(ax3, "c", "Robustness by ET window and threshold", "Ridge (green) and SW Flatwoods (blue)")

    # (d) Small canopy change, by AWS tercile.
    groups = ["All", "Ridge", "Southwest Flatwoods"]
    terc = ["low AWS", "mid AWS", "high AWS"]
    for i, gname in enumerate(groups):
        g = st[st["group"] == gname]
        for k, t in enumerate(terc):
            r = g[g["aws_tercile"] == t]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (k - 1) * 0.24
            ax4.plot([x, x], [r["et_lo"], r["et_hi"]], color=fs.ORDINAL[k], lw=1.3)
            ax4.scatter(x, r["dln_et"], s=26, color=fs.ORDINAL[k], edgecolor="white", lw=0.6, zorder=3)
            ax4.text(x, r["et_hi"] + 0.4, f"n={int(r['cells'])}", fontsize=5.3, color=fs.MUTED,
                     ha="center", va="bottom")
    fs.zero_line(ax4)
    ax4.set_xticks(range(len(groups)))
    ax4.set_xticklabels([fs.REGION_SHORT[gn] for gn in groups])
    ax4.tick_params(axis="x", length=0)
    ax4.set_ylabel("Change in ET (100 × Δln)")
    ax4.set_ylim(-9, 7)
    h = [Line2D([], [], marker="o", ls="", color=fs.ORDINAL[k], mec="white", ms=5,
                label=t.replace(" AWS", "")) for k, t in enumerate(terc)]
    ax4.legend(handles=h, title="AWS 0–100 cm tercile", title_fontsize=6.2, fontsize=6.2, ncol=3,
               loc="upper left", handletextpad=0.1, columnspacing=0.8)
    ax4.set_xlim(-0.55, 2.45)
    fs.panel(ax4, "d", "Small canopy change, by soil storage", "cells with |Δ canopy index| ≤ 5 pp")
    fs.save(fig, "fig_structure_soil")


if __name__ == "__main__":
    main()
