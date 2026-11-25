"""Figure 5: annual ET and K_c,act by region, 2000-2025, with spatial uncertainty.

(a) Annual ET (mm) and (b) K_c,act (ET / ET0) by region; lines are the citrus-area-weighted mean
ET in (a) and the ratio of weighted mean ET to weighted mean ET0 in (b) (script 03), bands are 95% spatial block-bootstrap intervals (20-km blocks of cells
resampled, 1000 replicates). Horizontal ticks mark the means of the two windows of the main
contrast (2015-2019 and 2021-2025).

--eto picks the ET0 scenario for panel (b) (default: wind-standardized FAWN, the central one);
other scenarios are written as fig_kc_trend_<eto>.

Reads  data/processed/annual_cells[_<eto>].csv, citrus_cells_1km.csv
Writes figures/fig_kc_trend[_<eto>].png and .pdf
"""

import argparse
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

ETO = {"fawn_windclim": ("_fawn_windclim", "FAWN, wind-standardized"),
       "fawn": ("_fawn", "FAWN"), "gridmet": ("", "gridMET, raw"),
       "gridmet_bc": ("_gridmet_bc", "gridMET, OpenET bias-corrected")}
ORDER = ["Indian River", "Southwest Flatwoods", "Ridge"]
RNG = np.random.default_rng(5)


def regional_bands(a, cells, nboot=1000):
    """Point estimate and 95% spatial-block interval of ET and Kc per region and year."""
    a = a.merge(cells[["cell_id", "block"]], on="cell_id")
    out = []
    for reg, g in a.groupby("region"):
        g = g.assign(wet=g["citrus_ha"] * g["et_year"], weto=g["citrus_ha"] * g["eto_year"])
        piv = {k: g.pivot_table(index="block", columns="year", values=k, aggfunc="sum").fillna(0)
               for k in ("wet", "weto", "citrus_ha")}
        years = piv["wet"].columns.values
        W, E, O = (piv[k].values for k in ("citrus_ha", "wet", "weto"))
        nb = len(W)
        cnt = np.stack([np.bincount(RNG.integers(0, nb, nb), minlength=nb) for _ in range(nboot)])
        w, e, o = cnt @ W, cnt @ E, cnt @ O
        et_b, kc_b = e / w, e / o
        et0, kc0 = E.sum(0) / W.sum(0), E.sum(0) / O.sum(0)
        for j, y in enumerate(years):
            out.append({"region": reg, "year": y, "et": et0[j], "kc": kc0[j],
                        "et_lo": np.percentile(et_b[:, j], 2.5), "et_hi": np.percentile(et_b[:, j], 97.5),
                        "kc_lo": np.percentile(kc_b[:, j], 2.5), "kc_hi": np.percentile(kc_b[:, j], 97.5)})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eto", choices=list(ETO), default="fawn_windclim")
    args = p.parse_args()
    sfx, eto_label = ETO[args.eto]
    spec = importlib.util.spec_from_file_location("s03", fs.ROOT / "scripts" / "03_summarize_series.py")
    s03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s03)
    fs.setup()
    cells = pd.read_csv(fs.PROC / "citrus_cells_1km.csv")
    cells["block"] = fs.block_ids(cells)
    a = pd.read_csv(fs.PROC / f"annual_cells{sfx}.csv")
    b = regional_bands(a, cells)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fs.FULL, 4.6), sharex=True,
                                   gridspec_kw={"hspace": 0.18})
    for ax, var, ylabel, gap in ((ax1, "et", "Annual ET (mm)", 32),
                                 (ax2, "kc", f"K$_{{c,act}}$ = ET / ET$_0$", 0.028)):
        fs.shade_events(ax)
        ends = []
        for reg in ORDER:
            g = b[b["region"] == reg].sort_values("year")
            c = fs.REGION[reg]
            ax.fill_between(g["year"], g[f"{var}_lo"], g[f"{var}_hi"], color=c, alpha=0.16, lw=0)
            ax.plot(g["year"], g[var], color=c, lw=1.7, zorder=3)
            for (y0, y1) in ((2015, 2019), (2021, 2025)):
                m = g[g["year"].between(y0, y1)][var].mean()
                ax.plot([y0 - 0.3, y1 + 0.3], [m, m], color=c, lw=0.9, ls=(0, (1, 1.2)), zorder=2)
            ends.append((g[var].iloc[-1], fs.REGION_SHORT[reg], c))
        fs.label_ends(ax, ends, 2025.3, min_gap=gap, dx=0.5, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xlim(1999.5, 2028.6)
        ax.spines["bottom"].set_bounds(1999.5, 2025.5)
    ax2.set_xticks(range(2000, 2026, 5))
    ylim = ax1.get_ylim()
    for y, name in ((2004.5, "2004–05\nstorms"), (2017, "Irma"), (2022, "Ian"),
                    (2024, "Helene,\nMilton")):
        ax1.text(y, ylim[1], name, fontsize=5.6, color=fs.MUTED, ha="center", va="top")
    ax1.text(fs.HLB_YEAR + 0.25, ylim[0] + 0.03 * (ylim[1] - ylim[0]), "HLB confirmed",
             fontsize=5.8, color=fs.MUTED, va="bottom")
    ax2.annotate("dotted: 2015–2019 and\n2021–2025 means", xy=(2017, 0.0), xycoords=("data", "axes fraction"),
                 xytext=(2010.2, 0.05), textcoords=("data", "axes fraction"), fontsize=6,
                 color=fs.MUTED, va="bottom")
    fs.panel(ax1, "a", "Annual ET by region",
             "OpenET ensemble; bands: 95% spatial block-bootstrap intervals")
    fs.panel(ax2, "b", f"K$_{{c,act}}$ with {eto_label} ET$_0$")
    name = "fig_kc_trend" if args.eto == "fawn_windclim" else f"fig_kc_trend_{args.eto}"
    fs.save(fig, name)


if __name__ == "__main__":
    main()
