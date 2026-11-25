"""Do the main results hold across the OpenET ensemble and its six models?

All with the central ETo (FAWN, wind held at climatology), on the cell-years common to all seven
products (2001-2025): Kc contrasts (100 x dln, script 19 construction: 2001-05 vs 2019-25, 2015-19 vs 2021-25, 2020 vs 2023-25), gap to the
UF/IFAS Kc (script 10) and the Ian wind gradient vs. placebo (script 08), next to each
model's bias at the Arcadia citrus tower (script 12).

Reads  annual_region_fawn_windclim[_<model>].csv, kc_gap_vs_ifas[_<model>].csv,
       hurricane_wind_gradient[_<model>].csv, tower_metrics.csv
Writes data/processed/model_comparison.csv, figures/fig_model_comparison.png and .pdf
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"]
NAMES = {"ensemble": "Ensemble", "disalexi": "DisALEXI", "eemetric": "eeMETRIC",
         "geesebal": "geeSEBAL", "ptjpl": "PT-JPL", "sims": "SIMS", "ssebop": "SSEBop"}
INK, INK_2, GRID = "#1a1a19", "#5f5e5a", "#e4e3df"
ALL_C, RIDGE_C = "#2a78d6", "#1baf7a"  # categorical slots 1 and 3


def sfx(model):
    return "" if model == "ensemble" else f"_{model}"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_year(g):
    return int(g.loc[g["kc"].notna(), "year"].min())


def main():
    s03 = load("s03", "03_summarize_series.py")
    s19 = load("s19", "19_trend_robustness.py")
    tower = pd.read_csv(PROC / "tower_metrics.csv").set_index("model")
    rows = []
    # Common support: the cell-years present for all seven products (2001-2025, since DisALEXI
    # starts in January 2001), so every product is compared on the same cells, years and weights.
    cells = {m: pd.read_csv(PROC / f"annual_cells_fawn_windclim{sfx(m)}.csv") for m in MODELS}
    keys = None
    for a in cells.values():
        k = set(zip(a["cell_id"], a["year"]))
        keys = k if keys is None else keys & k
    keys = pd.DataFrame(sorted(keys), columns=["cell_id", "year"])
    print(f"Common support: {len(keys):,} cell-years, {keys['year'].min()}-{keys['year'].max()}")
    for model in MODELS:
        reg = s03.aggregate(cells[model].merge(keys, on=["cell_id", "year"]))
        gap = pd.read_csv(PROC / f"kc_gap_vs_ifas{sfx(model)}.csv")
        gap = gap[gap["eto_source"] == "fawn_windclim"]
        grad = pd.read_csv(PROC / f"hurricane_wind_gradient{sfx(model)}.csv")
        for region in ("All", "Ridge", "Southwest Flatwoods", "Indian River"):
            g = reg[reg["region"] == region].sort_values("year").dropna(subset=["kc"])
            long = s19.contrast(g, (2001, 2005), (2019, 2025))
            recent = s19.contrast(g, (2015, 2019), (2021, 2025))
            drop = s19.contrast(g, (2020, 2020), (2023, 2025))
            _, p, sen = s03.mann_kendall(g["kc"])
            gp = gap[(gap["region"] == region) & (gap["period"] == "2021-2025")].iloc[0]
            gpre = gap[(gap["region"] == region) & (gap["period"] == "2000-2005 (pre-HLB)")].iloc[0]
            row = {"model": model, "region": region, "first_year": first_year(g),
                   "tower_bias_pct": tower.loc[model, "bias_pct"],
                   "tower_kge": tower.loc[model, "kge"],
                   "change_pct": long["dln_kc"], "long_lo": long.get("kc_lo"),
                   "long_hi": long.get("kc_hi"), "long_dln_et": long["dln_et"],
                   "recent_pct": recent["dln_kc"], "recent_lo": recent["kc_lo"],
                   "recent_hi": recent["kc_hi"], "recent_dln_et": recent["dln_et"],
                   "recent_et_lo": recent["et_lo"], "recent_et_hi": recent["et_hi"],
                   "sen_per_yr": sen, "mk_p": p,
                   "drop_2020_to_2023_25_pct": drop["dln_kc"],
                   # observed ET relative to UF/IFAS-based ETc, pre-HLB (negative = below)
                   "kc_vs_ifas_pre_pct": 100 * (gpre["et_obs_mm"] / gpre["etc_ifas_mm"] - 1),
                   "gap_ifas_2021_25_pct": gp["gap_pct"],
                   "gap_ifas_2021_25_hm3": gp["gap_hm3"]}
            if region == "All":
                ian = grad[grad["event"] == "2022 (Ian)"]["slope_13-24_per10kt"].iat[0]
                plac = grad[grad["event"].str.startswith("placebo")
                            & grad["event"].str.endswith("with 2022 (Ian) wind")]["slope_13-24_per10kt"]
                row.update({"ian_slope_pct_per10kt": 100 * ian,
                            "ian_placebo_min": 100 * plac.min(),
                            "ian_placebo_max": 100 * plac.max(),
                            "ian_beyond_placebo": ian < plac.min()})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "model_comparison.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    cols = ["model", "first_year", "tower_bias_pct", "change_pct", "long_lo", "long_hi",
            "recent_pct", "recent_lo", "recent_hi", "recent_dln_et", "recent_et_lo", "recent_et_hi",
            "drop_2020_to_2023_25_pct", "gap_ifas_2021_25_pct"]
    print("All cells:")
    print(out[out["region"] == "All"][cols].round(3).to_string(index=False))
    print("\nRidge:")
    print(out[out["region"] == "Ridge"][cols].round(3).to_string(index=False))

    plot(out, tower)


def plot(out, tower):
    """Figure 6: dot plot per product, ordered by bias at the Arcadia tower."""
    import figstyle as fs
    from matplotlib.lines import Line2D
    fs.setup()
    order = out[out["region"] == "All"].sort_values("tower_bias_pct")["model"].tolist()
    ridge = fs.REGION["Ridge"]
    panels = [("change_pct", "long_lo", "long_hi", "Long-term K$_{c,act}$", "2001–2005 to 2019–2025",
               "Change, 100 × Δln"),
              ("recent_pct", "recent_lo", "recent_hi", "Recent K$_{c,act}$", "2015–2019 to 2021–2025",
               "Change, 100 × Δln"),
              ("gap_ifas_2021_25_pct", None, None, "UF/IFAS crop ET above ET",
               "2021–2025, historical coefficients", "Difference (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL, 2.9), sharey=True)
    fig.subplots_adjust(wspace=0.12, bottom=0.26, left=0.17)
    for ax, (col, lo, hi, title, sub, xl), letter in zip(axes, panels, "abc"):
        ax.axhspan(order.index("ensemble") - 0.42, order.index("ensemble") + 0.42,
                   color="#f3f2ee", lw=0, zorder=0)
        fs.zero_line(ax, "x")
        for region, color, face, dy in (("All", fs.INK, fs.INK, -0.13), ("Ridge", ridge, "white", 0.13)):
            sub_ = out[out["region"] == region].set_index("model").loc[order]
            yy = [i + dy for i in range(len(order))]
            if lo:
                for y, l, h in zip(yy, sub_[lo], sub_[hi]):
                    ax.plot([l, h], [y, y], color=color, lw=0.9, alpha=0.45, solid_capstyle="butt")
            ax.scatter(sub_[col], yy, s=24, color=face, edgecolors=color, lw=1.1, zorder=3)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=fs.GRID, lw=0.6)
        ax.set_xlabel(xl)
        fs.panel(ax, letter, title, sub)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{NAMES[m]}  {tower.loc[m, 'bias_pct']:+.0f}%" for m in order])
    for lab, m in zip(axes[0].get_yticklabels(), order):
        lab.set_fontweight("bold" if m == "ensemble" else "regular")
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(len(order) - 0.5, -0.9)
    axes[0].text(-0.02, 1.0, "Product  tower bias", transform=axes[0].transAxes, ha="right",
                 va="bottom", fontsize=6.3, color=fs.MUTED)
    handles = [Line2D([], [], marker="o", ls="", color=fs.INK, ms=5, label="All cells"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=ridge, mew=1.1, ms=5, label="Ridge"),
               Line2D([], [], color=fs.MUTED, lw=0.9, label="95% interval")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.58, 0.0))
    fs.save(fig, "fig_model_comparison")

if __name__ == "__main__":
    main()
