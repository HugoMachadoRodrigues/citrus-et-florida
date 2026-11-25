"""Figure 3: OpenET against the USGS citrus ET tower near Arcadia (May 2004 - April 2005).

(a) Monthly ET: tower, OpenET ensemble, each of the six models (thin) and their range.
(b) Monthly OpenET against tower ET with the 1:1 line (ensemble and models).
(c) Each product's bias of the 12-month total against its Kling-Gupta efficiency.

Reads  data/processed/tower_monthly.csv, tower_metrics.csv (script 12)
Writes figures/fig_tower_validation.png and .pdf
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

NAMES = {"ensemble": "Ensemble", "disalexi": "DisALEXI", "eemetric": "eeMETRIC",
         "geesebal": "geeSEBAL", "ptjpl": "PT-JPL", "sims": "SIMS", "ssebop": "SSEBop"}
OPENET, BAND, MODEL = "#2a78d6", "#dbe8f8", "#9ec5f4"


def main():
    fs.setup()
    d = pd.read_csv(fs.PROC / "tower_monthly.csv")
    met = pd.read_csv(fs.PROC / "tower_metrics.csv").set_index("model")
    d["t"] = pd.to_datetime(dict(year=d["year"], month=d["month"], day=15))
    ens = d[d["model"] == "ensemble"].sort_values("t")
    mods = d[d["model"] != "ensemble"]
    rng = mods.groupby("t")["et_mm"].agg(["min", "max"])

    fig = plt.figure(figsize=(fs.FULL, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.42)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) Time series.
    ax1.axvspan(pd.Timestamp("2004-08-13"), pd.Timestamp("2004-09-27"), color=fs.WASH, lw=0)
    ax1.text(pd.Timestamp("2004-09-04"), 223, "Charley,\nFrances,\nJeanne", fontsize=5.8,
             color=fs.MUTED, ha="center", va="top")
    ax1.axvspan(pd.Timestamp("2005-02-01"), pd.Timestamp("2005-04-30"), color=fs.WASH, lw=0)
    ax1.text(pd.Timestamp("2005-03-16"), 223, "dry\nspring", fontsize=5.8, color=fs.MUTED,
             ha="center", va="top")
    ax1.fill_between(rng.index, rng["min"], rng["max"], color=BAND, lw=0, zorder=1)
    for m, g in mods.groupby("model"):
        g = g.sort_values("t")
        ax1.plot(g["t"], g["et_mm"], color=MODEL, lw=0.6, zorder=2)
    ax1.plot(ens["t"], ens["et_mm"], color=OPENET, lw=1.8, marker="o", ms=3.2, mec="white",
             mew=0.5, zorder=4)
    ax1.plot(ens["t"], ens["tower_mm"], color=fs.INK, lw=1.8, marker="s", ms=3.2, mec="white",
             mew=0.5, zorder=5)
    last = ens.iloc[-1]
    ax1.text(last["t"] + pd.Timedelta(days=12), last["et_mm"], "OpenET\nensemble", fontsize=6.3,
             color=fs.INK, va="center")
    ax1.text(last["t"] + pd.Timedelta(days=12), last["tower_mm"] - 4, "Tower", fontsize=6.3,
             color=fs.INK, va="center")
    ax1.text(pd.Timestamp("2004-12-20"), 25, "band: six models", fontsize=6, color=fs.MUTED,
             ha="center")
    ax1.set_xlim(pd.Timestamp("2004-04-25"), pd.Timestamp("2005-06-25"))
    ax1.spines["bottom"].set_bounds(mdates.date2num(pd.Timestamp("2004-04-25")),
                                    mdates.date2num(pd.Timestamp("2005-05-05")))
    ax1.set_ylim(20, 225)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[5, 8, 11, 2]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.set_ylabel("ET (mm month$^{-1}$)")
    fs.panel(ax1, "a", "Monthly ET, flatwoods orange grove")

    # (b) Scatter against the 1:1 line.
    lim = (20, 225)
    ax2.plot(lim, lim, color=fs.AXIS, lw=0.9, zorder=1)
    ax2.text(212, 200, "1:1", fontsize=6, color=fs.MUTED, ha="center", va="top")
    ax2.scatter(mods["tower_mm"], mods["et_mm"], s=7, color=MODEL, lw=0, zorder=2)
    ax2.scatter(ens["tower_mm"], ens["et_mm"], s=18, color=OPENET, ec="white", lw=0.5, zorder=3)
    e = met.loc["ensemble"]
    ax2.text(0.04, 0.96, f"Ensemble\nbias {e.bias_pct:+.0f}%   r = {e.r:.2f}\n"
             f"RMSE {e.rmse_mm_month:.0f} mm month$^{{-1}}$", transform=ax2.transAxes,
             fontsize=6.3, color=fs.INK, va="top")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.grid(axis="x", color=fs.GRID, lw=0.6)
    ax2.set_xlabel("Tower ET (mm month$^{-1}$)")
    ax2.set_ylabel("OpenET ET (mm month$^{-1}$)")
    fs.panel(ax2, "b", "Agreement by month")

    # (c) Bias vs KGE.
    ax3.axvline(0, color=fs.AXIS, lw=0.8)
    offsets = {"ensemble": (6, -7), "disalexi": (-5, -4), "eemetric": (6, -2), "geesebal": (-6, 3),
               "ptjpl": (6, 2), "sims": (6, 3), "ssebop": (6, -6)}
    for m, r in met.iterrows():
        is_e = m == "ensemble"
        ax3.scatter(r.bias_pct, r.kge, s=34 if is_e else 22, color=OPENET if is_e else MODEL,
                    ec=fs.INK if is_e else "white", lw=0.6, zorder=3)
        dx, dy = offsets.get(m, (5, 0))
        ax3.annotate(NAMES[m], (r.bias_pct, r.kge), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.3, color=fs.INK, ha="left" if dx > 0 else "right", va="center",
                     weight="bold" if is_e else "regular")
    ax3.set_xlim(-3, 40)
    ax3.set_ylim(0.45, 0.95)
    ax3.grid(axis="x", color=fs.GRID, lw=0.6)
    ax3.set_xlabel("Bias of 12-month total (%)")
    ax3.set_ylabel("Kling–Gupta efficiency")
    ax3.text(39, 0.94, "better: low bias, high KGE (top left)", fontsize=5.8, color=fs.MUTED,
             ha="right", va="center")
    fs.panel(ax3, "c", "Bias and skill by product")
    fs.save(fig, "fig_tower_validation")


if __name__ == "__main__":
    main()
