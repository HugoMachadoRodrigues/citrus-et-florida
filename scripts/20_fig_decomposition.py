"""Figure 4: decomposition of the change in K_c,act into the change in ET and in ET0.

dln Kc = dln ET - dln ET0. For each region, the black marker is the change in ET (the same
for every scenario); each arrow starts at that value and ends at the change in K_c,act under
one ET0 scenario, so the arrow is -dln ET0 of that scenario. Lines are 95% moving-block
bootstrap intervals over years (script 19). Two contrasts: 2000-2005 -> 2019-2025 and
2015-2019 -> 2021-2025.

Reads  data/processed/trend_robustness.csv
Writes figures/fig_decomposition.png and .pdf
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figstyle as fs
import pandas as pd

REGIONS = ["All", "Ridge", "Southwest Flatwoods", "Indian River"]
SCEN = list(fs.ETO)
CONTRASTS = [("2000-05 vs 2019-25", "Long-term: 2000–2005 to 2019–2025"),
             ("2015-19 vs 2021-25 (no 2020)", "Recent: 2015–2019 to 2021–2025")]
ROW = 0.17  # vertical step between rows inside a region block


def main():
    fs.setup()
    t = pd.read_csv(fs.PROC / "trend_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 4.0), sharey=True)
    fig.subplots_adjust(wspace=0.08, bottom=0.2)
    for ax, (key, title), letter in zip(axes, CONTRASTS, "ab"):
        d = t[t["contrast"] == key]
        for i, reg in enumerate(REGIONS):
            y0 = -i
            if i % 2 == 0:
                ax.axhspan(y0 + 0.5 * ROW + 0.08, y0 - 4.5 * ROW - 0.08, color="#f6f5f2", lw=0,
                           zorder=0)
            r = d[d["region"] == reg]
            et, elo, ehi = r[["dln_et", "et_lo", "et_hi"]].iloc[0]
            ax.plot([elo, ehi], [y0, y0], color=fs.INK, lw=0.7, alpha=0.45,
                    solid_capstyle="butt", zorder=3)
            ax.plot(et, y0, "D", ms=4.6, color=fs.INK, mec="white", mew=0.6, zorder=4)
            for k, sc in enumerate(SCEN):
                rr = r[r["scenario"] == sc]
                if rr.empty:
                    continue
                color, _, mk = fs.ETO[sc]
                y = y0 - (k + 1) * ROW
                kc, lo, hi = rr[["dln_kc", "kc_lo", "kc_hi"]].iloc[0]
                ax.plot([lo, hi], [y, y], color=color, lw=0.7, alpha=0.4,
                        solid_capstyle="butt", zorder=2)
                ax.plot([et, et], [y0, y], color=fs.AXIS, lw=0.5, zorder=1)
                ax.annotate("", xy=(kc, y), xytext=(et, y),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                            mutation_scale=7, shrinkA=0, shrinkB=2.5), zorder=3)
                ax.plot(kc, y, mk, ms=4.2, color=color, mec="white", mew=0.5, zorder=4)
        fs.zero_line(ax, "x")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=fs.GRID, lw=0.6)
        ax.set_xlabel("Change, 100 × Δln (≈ %)")
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        fs.panel(ax, letter, title)
    axes[0].set_yticks([-i for i in range(len(REGIONS))])
    axes[0].set_yticklabels([fs.REGION_SHORT[r] for r in REGIONS], weight="bold")
    axes[0].tick_params(axis="y", length=0, pad=4)
    axes[0].set_ylim(-len(REGIONS) + 1 - 4.5 * ROW - 0.12, 0.5 * ROW + 0.12)
    axes[0].text(0.01, 0.005, "", transform=axes[0].transAxes)
    handles = [Line2D([], [], marker="D", ls="", color=fs.INK, mec="white", ms=5,
                      label="ET (same for every ET$_0$)")]
    handles += [Line2D([], [], marker=fs.ETO[s][2], color=fs.ETO[s][0], mec="white", ms=5,
                       lw=1.1, label=f"K$_{{c,act}}$ with {fs.ETO_SHORT[s]}") for s in SCEN]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.6,
               bbox_to_anchor=(0.53, -0.01), columnspacing=1.4, handletextpad=0.4)
    fs.save(fig, "fig_decomposition")


if __name__ == "__main__":
    main()
