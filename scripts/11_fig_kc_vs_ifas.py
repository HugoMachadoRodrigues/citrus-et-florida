"""Figure 9: monthly K_c,act by period against the historical UF/IFAS coefficients, by region.

Observed K_c,act (OpenET ensemble / wind-standardized FAWN ET0; ratio of area-weighted sums per
year-month, mean over the years of each period) for 2000-2005, 2006-2020 and 2021-2025 (ink ramp,
light to dark), the band is +/- 1 standard deviation across the years of 2021-2025, and the
historical UF/IFAS curve for mature trees is dashed. Each panel reports the annual difference
between crop ET from the historical curve and estimated ET (Equation (4)).

Reads  data/processed/kc_monthly_vs_ifas.csv, kc_gap_vs_ifas.csv (script 10)
Writes figures/fig_kc_vs_ifas.png and .pdf
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs

PERIODS = ["2000-2005 (pre-HLB)", "2006-2020", "2021-2025"]
PLABEL = ["2000–2005", "2006–2020", "2021–2025"]
REGIONS = ["Ridge", "Southwest Flatwoods", "Indian River"]
MONTHS = list("JFMAMJJASOND")


def main():
    fs.setup()
    m = pd.read_csv(fs.PROC / "kc_monthly_vs_ifas.csv").query("eto_source == 'fawn_windclim'")
    gap = pd.read_csv(fs.PROC / "kc_gap_vs_ifas.csv").query("eto_source == 'fawn_windclim'")
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL, 2.55), sharey=True)
    fig.subplots_adjust(wspace=0.12, bottom=0.17)
    for ax, reg, letter in zip(axes, REGIONS, "abc"):
        ax.axvspan(2.5, 5.5, color=fs.WASH, lw=0, zorder=0)
        g = m[m["region"] == reg]
        for k, per in enumerate(PERIODS):
            p = g[g["period"] == per].sort_values("month")
            if per == PERIODS[-1]:
                ax.fill_between(p["month"], p["kc_obs"] - p["kc_sd"], p["kc_obs"] + p["kc_sd"],
                                color=fs.ORDINAL[k], alpha=0.12, lw=0, zorder=1)
            ax.plot(p["month"], p["kc_obs"], color=fs.ORDINAL[k], lw=1.9 if k == 2 else 1.3,
                    marker="o", ms=2.6 if k == 2 else 0, zorder=3)
        p = g[g["period"] == PERIODS[0]].sort_values("month")
        ax.plot(p["month"], p["kc_ifas"], color=fs.ACCENT, lw=1.5, ls=(0, (4, 2)), zorder=4)
        gp = gap[(gap["region"] == reg) & (gap["period"] == "2021-2025")].iloc[0]
        sub = (f"UF/IFAS − ET, 2021–25: {gp['gap_pct']:+.0f}% "
               f"({gp['gap_mm']:+.0f} mm)").replace("-", "−")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTHS)
        ax.set_xlim(0.6, 12.4)
        ax.tick_params(axis="x", length=0)
        fs.panel(ax, letter, fs.REGION_SHORT[reg], sub)
    axes[0].text(4, 1.235, "dry season", fontsize=6, color=fs.MUTED, ha="center", va="top")
    axes[0].set_ylim(0.55, 1.25)
    axes[0].set_ylabel("K$_{c,act}$ (ET / ET$_0$, FAWN wind-std.)")
    h = [Line2D([], [], color=fs.ORDINAL[k], lw=1.9 if k == 2 else 1.3, label=PLABEL[k])
         for k in range(3)]
    h.append(Line2D([], [], color=fs.ACCENT, lw=1.5, ls=(0, (4, 2)),
                    label="Historical UF/IFAS K$_c$ (mature trees)"))
    h.append(plt.Rectangle((0, 0), 1, 1, color=fs.ORDINAL[2], alpha=0.12, lw=0,
                           label="±1 SD across 2021–2025"))
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handletextpad=0.4)
    fs.save(fig, "fig_kc_vs_ifas")


if __name__ == "__main__":
    main()
