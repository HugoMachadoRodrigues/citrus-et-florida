"""Figure 7: hurricane exposure gradients.

(a-e) For each event, the K_c,act anomaly of the most exposed peak-wind tercile minus the least
      exposed tercile (citrus-area weighted, pre-event offset removed, 3-month running mean),
      with a 95% spatial block-bootstrap band (20-km blocks, 1000 replicates); the shaded window
      is months 7-12, used for the event slopes. Curves stop before the next event or at the
      end of the record (Table 2).
(f)   Cell-level change in K_c,act 7-12 months after each event per +10 kt of peak wind (95%
      interval), against the nine placebo Septembers given the same wind field.

Reads  data/processed/series/openet_cells_*.csv and fawn_eto_cells_windclim.csv (through script
       08), hurricane_exposure.csv, hurricane_wind_gradient.csv
Writes figures/fig_hurricane_events.png and .pdf
"""

import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle as fs

RNG = np.random.default_rng(11)
NBOOT = 1000
SHORT = {"2004 (Charley, Frances, Jeanne)": "2004: Charley, Frances, Jeanne",
         "2005 (Wilma)": "2005: Wilma", "2017 (Irma)": "2017: Irma", "2022 (Ian)": "2022: Ian",
         "2024 (Helene, Milton)": "2024: Helene, Milton"}


def load_s08():
    spec = importlib.util.spec_from_file_location("s08", fs.ROOT / "scripts" / "08_hurricane_events.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run3(a):
    """Centred 3-month running mean along the last axis (edges use 2 months)."""
    out = np.full_like(a, np.nan)
    n = a.shape[-1]
    for k in range(n):
        out[..., k] = np.nanmean(a[..., max(0, k - 1):min(n, k + 2)], axis=-1)
    return out


def tercile_contrast(w, blocks):
    """High-minus-low weighted anomaly by month, with a block bootstrap of the curve."""
    w = w.assign(block=w["cell_id"].map(blocks), wr=w["citrus_ha"] * w["rel"])
    ks = np.sort(w["k"].unique())
    ub = np.sort(w["block"].unique())
    mats = {}
    for terc in ("high", "low"):
        g = w[w["tercile"] == terc]
        mats[terc] = [g.pivot_table(index="block", columns="k", values=v, aggfunc="sum")
                      .reindex(index=ub, columns=ks).fillna(0).values for v in ("wr", "citrus_ha")]
    nb = len(ub)
    cnt = np.vstack([np.ones(nb)] + [np.bincount(RNG.integers(0, nb, nb), minlength=nb)
                                      for _ in range(NBOOT)])
    with np.errstate(invalid="ignore", divide="ignore"):
        hi = (cnt @ mats["high"][0]) / (cnt @ mats["high"][1])
        lo = (cnt @ mats["low"][0]) / (cnt @ mats["low"][1])
    d = 100 * (hi - lo)
    d = d - np.nanmean(d[:, ks < 0], axis=1, keepdims=True)
    d = run3(d)
    return ks, d[0], np.nanpercentile(d[1:], 2.5, axis=0), np.nanpercentile(d[1:], 97.5, axis=0)


def main():
    fs.setup()
    s08 = load_s08()
    expo = pd.read_csv(fs.PROC / "hurricane_exposure.csv").set_index("cell_id")
    cells = pd.read_csv(fs.PROC / "citrus_cells_1km.csv").set_index("cell_id")
    blocks = fs.block_ids(cells)
    m = s08.monthly_kc("ensemble")
    grad = pd.read_csv(fs.PROC / "hurricane_wind_gradient.csv")

    fig, axes = plt.subplots(2, 3, figsize=(fs.FULL, 4.5), gridspec_kw={"hspace": 0.48,
                                                                       "wspace": 0.3})
    flat = axes.ravel()
    names_all = list(s08.EVENTS)
    for ax, (name, ev), letter in zip(flat[:5], s08.EVENTS.items(), "abcde"):
        wind = expo[[name, "citrus_ha"]].rename(columns={name: "peak_kt"})
        wind["tercile"] = pd.qcut(wind["peak_kt"], 3, labels=["low", "mid", "high"])
        w = s08.anomalies(m, ev["month"], ev["baseline"], ev["last"]).join(wind, on="cell_id")
        ks, est, lo, hi = tercile_contrast(w, blocks)
        is_ian = name.startswith("2022")
        col = fs.ACCENT if is_ian else fs.INK
        ax.axvspan(7, 12, color=fs.WASH, lw=0, zorder=0)
        ax.axvline(0, color=fs.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        fs.zero_line(ax)
        ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(ks, est, color=col, lw=1.5, zorder=3)
        med = wind.groupby("tercile", observed=True)["peak_kt"].median()
        ax.text(0.98, 0.04, f"{med['high']:.0f} vs {med['low']:.0f} kt", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.2, color=fs.INK_2)
        if ev["last"] < 36:
            end = "end of\nrecord" if name == names_all[-1] else "next\nevent"
            ax.text(ks.max() + 0.8, est[-1], end, fontsize=5.6, color=fs.MUTED, va="center")
        ax.set_xlim(-12.5, 37)
        ax.set_xticks([-12, 0, 12, 24, 36])
        ax.set_ylim(-14, 11)
        if letter in "ad":
            ax.set_ylabel("High − low exposure\nK$_{c,act}$ anomaly (%)")
        if letter in "de":
            ax.set_xlabel("Months since landfall")
        fs.panel(ax, letter, SHORT[name])
    flat[0].text(9.5, 10.5, "months\n7–12", fontsize=5.6, color=fs.MUTED, ha="center", va="top")

    # (f) Event slopes against placebo slopes.
    ax = flat[5]
    names = list(s08.EVENTS)
    col = "slope_7-12_per10kt"
    for i, name in enumerate(names):
        pl = grad[grad["event"].str.startswith("placebo") & grad["event"].str.endswith(f"with {name} wind")]
        ax.scatter(100 * pl[col], np.full(len(pl), i) + RNG.uniform(-0.12, 0.12, len(pl)), s=9,
                   facecolor="white", edgecolor=fs.MUTED, lw=0.7, zorder=2)
        r = grad[grad["event"] == name].iloc[0]
        c = fs.ACCENT if name.startswith("2022") else fs.INK
        ax.plot([100 * r["ci_lo_7-12"], 100 * r["ci_hi_7-12"]], [i, i], color=c, lw=1.3, zorder=3)
        ax.scatter(100 * r[col], i, s=30, color=c, edgecolor="white", lw=0.6, zorder=4)
    fs.zero_line(ax, "x")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(["2004 storms" if n.startswith("2004")
                        else n.split(" (")[1].rstrip(")").replace(", ", "/") for n in names])
    ax.set_ylim(len(names) + 0.75, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=fs.GRID, lw=0.6)
    ax.set_xlabel("ΔK$_{c,act}$ per +10 kt, months 7–12 (%)")
    ax.scatter([], [], s=9, facecolor="white", edgecolor=fs.MUTED, lw=0.7, label="9 placebo Septembers")
    ax.scatter([], [], s=30, color=fs.INK, label="event, 95% interval")
    ax.legend(loc="lower left", fontsize=6, bbox_to_anchor=(-0.02, -0.03), handletextpad=0.2,
              ncol=1, labelspacing=0.3)
    fs.panel(ax, "f", "Gradient vs. placebos")
    fs.save(fig, "fig_hurricane_events")


if __name__ == "__main__":
    main()
