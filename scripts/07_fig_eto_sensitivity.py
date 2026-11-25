"""Figure 2: the four reference-ET scenarios and why they diverge.

(a) Annual ET0 over the citrus cells under each scenario.
(b) All-cell K_c,act = ET/ET0 with each scenario (same ET in every case).
(c) Trends of the ET0 drivers at the ten long-record FAWN stations (% per decade): FAWN
    measurements, gridMET and ERA5-Land; dots are station medians, bars the range across
    stations.

Reads  data/processed/annual_region[_<eto>].csv (script 03), eto_drivers_trend.csv (script 06)
Writes figures/fig_eto_sensitivity.png and .pdf
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

FILES = {"gridMET, raw": "annual_region.csv",
         "gridMET, OpenET bias-corrected": "annual_region_gridmet_bc.csv",
         "FAWN": "annual_region_fawn.csv",
         "FAWN, wind-standardized": "annual_region_fawn_windclim.csv"}
SOURCES = [("fawn", "FAWN stations", fs.ETO["FAWN, wind-standardized"][0]),
           ("gm", "gridMET", fs.ETO["gridMET, raw"][0]),
           ("era", "ERA5-Land", fs.MUTED)]
DRIVERS = [("wind10", "10-m wind speed"), ("srad", "Solar radiation"), ("vpd", "Vapour pressure\ndeficit")]


def main():
    fs.setup()
    fig = plt.figure(figsize=(fs.FULL, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1], wspace=0.42, hspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    ends1, ends2 = [], []
    for label, fname in FILES.items():
        color, ls, mk = fs.ETO[label]
        g = pd.read_csv(fs.PROC / fname).query("region == 'All'").sort_values("year")
        lw = 2.0 if label == "FAWN, wind-standardized" else 1.4
        ax1.plot(g["year"], g["eto_mm"], color=color, ls=ls, lw=lw, zorder=3)
        ax2.plot(g["year"], g["kc"], color=color, ls=ls, lw=lw, zorder=3)
        ends1.append((g["eto_mm"].iloc[-1], fs.ETO_SHORT[label], color))
        ends2.append((g["kc"].iloc[-1], fs.ETO_SHORT[label], color))
    for ax in (ax1, ax2):
        fs.shade_events(ax)
        ax.set_xlim(1999.5, 2029.8)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
        ax.spines["bottom"].set_bounds(1999.5, 2025.5)
    ax1.text(fs.HLB_YEAR + 0.3, 1755, "HLB confirmed", fontsize=6, color=fs.MUTED, va="top")
    ax1.text(2004, 1755, "2004–05\nstorms", fontsize=6, color=fs.MUTED, va="top", ha="right")
    plt.setp(ax1.get_xticklabels(), visible=False)
    fs.label_ends(ax1, ends1, 2025.3, min_gap=48, dx=0.5, fontsize=6.5)
    fs.label_ends(ax2, ends2, 2025.3, min_gap=0.03, dx=0.5, fontsize=6.5)
    ax1.set_ylim(1000, 1760)
    ax1.set_ylabel("Reference ET, ET$_0$ (mm yr$^{-1}$)")
    ax2.set_ylim(0.58, 1.05)
    ax2.set_ylabel("K$_{c,act}$ = ET / ET$_0$")
    fs.panel(ax1, "a", "Reference ET over the citrus cells")
    fs.panel(ax2, "b", "K$_{c,act}$: the same ET over four ET$_0$")

    # (c) Driver trends.
    d = pd.read_csv(fs.PROC / "eto_drivers_trend.csv")
    for i, (var, vlabel) in enumerate(DRIVERS):
        for j, (src, slabel, color) in enumerate(SOURCES):
            col = f"{src}_{var}"
            if col not in d:
                continue
            v = d[col].dropna()
            y = i + (j - 1) * 0.22
            ax3.plot([v.min(), v.max()], [y, y], color=color, lw=1.0, alpha=0.55,
                     solid_capstyle="butt")
            ax3.plot(v.median(), y, "o", ms=4.6, color=color, mec="white", mew=0.6, zorder=3,
                     label=slabel if i == 0 else None)
    fs.zero_line(ax3, "x")
    ax3.set_yticks([])
    for i, (_, vlabel) in enumerate(DRIVERS):
        ax3.text(-31.5, i - 0.36, vlabel.replace("\n", " "), fontsize=7, color=fs.INK,
                 weight="bold", va="bottom")
    ax3.grid(axis="y", visible=False)
    ax3.grid(axis="x", color=fs.GRID, lw=0.6)
    ax3.tick_params(axis="y", length=0)
    ax3.set_xlabel("Trend, 2000–2025 (% per decade)")
    ax3.legend(loc="lower left", fontsize=6.5, handletextpad=0.2, borderaxespad=0.3,
               bbox_to_anchor=(0.0, 0.0))
    ax3.annotate("FAWN wind: −22% per decade\nERA5-Land, same points: −3%", xy=(-22, -0.22), xytext=(-31, 0.42),
                 fontsize=6.2, color=fs.INK_2, arrowprops=dict(arrowstyle="-", color=fs.MUTED, lw=0.6))
    ax3.set_xlim(-32, 9)
    ax3.set_ylim(len(DRIVERS) - 0.6, -0.55)
    ax3.spines["left"].set_visible(False)
    fs.panel(ax3, "c", "Trends of ET$_0$ drivers", "10 long-record FAWN stations; dot = median, line = range")
    fs.save(fig, "fig_eto_sensitivity")


if __name__ == "__main__":
    main()
