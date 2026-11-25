"""Does the ET decline persist after grove structure, and does it depend on soil? (Tier C)

Inputs: annual_cells_fawn_windclim.csv (script 03: cell-year ET and Kc_act with the
wind-standardized FAWN ET0), canopy_tcc_cells.csv and soils_cells.csv (script 22).

1. Canopy trajectories: citrus-area-weighted tree canopy cover (TCC, %) by region and year.
2. Two-way fixed-effects panel (cell and year effects, weights = citrus area):
       ln y_it = a_i + d_t + b * TCC_it + e_it,   y = ET or Kc_act
   The year effects d_t with and without TCC show how much of the common trajectory
   follows canopy; the recent change is mean(d_2021..2025) - mean(d_2015..2019).
3. Cell-level change, 2015-2019 -> 2021-2025 (and 2000-2005 -> 2019-2025):
       dln ET_i = c + g * dTCC_i + h * (AWS_i - mean) + e_i
   so c is the change expected for a cell of average soil whose canopy did not change.
4. Stable-canopy cells (|dTCC| <= 2 percentage points): dln ET and dln Kc by tercile of
   available water storage (gNATSGO aws0_100), all cells and within regions.
Intervals: spatial block bootstrap (20-km blocks), 1000 replicates.

Writes data/processed/structure_soil_*.csv
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RNG = np.random.default_rng(7)
NBOOT = 1000
BLOCK_KM = 20
STABLE_PP = 2.0
PERIODS = {"recent": ((2015, 2019), (2021, 2025)), "long": ((2000, 2005), (2019, 2025)),
           # aligned with the NAIP acquisitions (2017, 2019 -> January-March 2023)
           "aligned": ((2017, 2019), (2023, 2023)), "to2023": ((2015, 2019), (2021, 2023))}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def data():
    s03 = load("s03", ROOT / "scripts" / "03_summarize_series.py")
    a = pd.read_csv(PROC / "annual_cells_fawn_windclim.csv")
    tcc = pd.read_csv(PROC / "canopy_tcc_cells.csv")
    soil = pd.read_csv(PROC / "soils_cells.csv")
    soil.loc[soil["pwsl1pomu"] > 100, "pwsl1pomu"] = np.nan  # 999 = not rated
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    cells["region"] = cells["county"].map(s03.region_of)
    cells["block"] = ((cells["x_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str) + "_"
                      + (cells["y_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str))
    d = (a[["cell_id", "year", "et_year", "eto_year", "kc"]]
         .merge(tcc, on=["cell_id", "year"], how="inner")
         .merge(cells[["cell_id", "region", "citrus_ha", "block"]], on="cell_id"))
    d = d.dropna(subset=["et_year", "kc", "tcc_pct"])
    d = d[(d["et_year"] > 0) & (d["kc"] > 0)]
    return d, soil, cells


def twoway_resid(df, cols, w, iters=50):
    """Remove weighted cell and year means from each column (alternating projections)."""
    out = df[cols].copy()
    for _ in range(iters):
        for key in ("cell_id", "year"):
            g = (out.mul(w, axis=0)).groupby(df[key]).transform("sum").div(
                w.groupby(df[key]).transform("sum"), axis=0)
            out = out - g
    return out


def year_effects(df, y, w):
    """Year effects of a weighted two-way FE model (cell effects absorbed), relative scale."""
    v = df[y].copy()
    for _ in range(50):
        ce = (v * w).groupby(df["cell_id"]).transform("sum") / w.groupby(df["cell_id"]).transform("sum")
        r = df[y] - ce
        ye = (r * w).groupby(df["year"]).sum() / w.groupby(df["year"]).sum()
        v = df[y] - df["year"].map(ye)
    return ye


def panel(d, label):
    d = d.copy()
    d["lnet"], d["lnkc"] = np.log(d["et_year"]), np.log(d["kc"])
    w = d["citrus_ha"]
    res = {}
    for y in ("lnet", "lnkc"):
        r = twoway_resid(d, [y, "tcc_pct"], w)
        b = float((w * r["tcc_pct"] * r[y]).sum() / (w * r["tcc_pct"] ** 2).sum())
        ye0 = year_effects(d, y, w)
        d["_adj"] = d[y] - b * d["tcc_pct"]
        ye1 = year_effects(d, "_adj", w)
        ch = lambda ye: 100 * (ye.loc[2021:2025].mean() - ye.loc[2015:2019].mean())
        chl = lambda ye: 100 * (ye.loc[2019:2025].mean() - ye.loc[2000:2005].mean())
        res[y] = {"b_per_pp": 100 * b, "recent_without_tcc": ch(ye0), "recent_with_tcc": ch(ye1),
                  "long_without_tcc": chl(ye0), "long_with_tcc": chl(ye1)}
    return {"group": label, **{f"{y}_{k}": v for y, dd in res.items() for k, v in dd.items()}}


def cell_changes(d, soil, period):
    """Cell change between periods, 100 x dln of mean ET and of mean ET0; dln Kc = dln ET -
    dln ET0 (ratio of period means, as in script 19)."""
    (a0, a1), (b0, b1) = PERIODS[period]
    cols = ["et_year", "eto_year", "tcc_pct"]
    m0 = d[d["year"].between(a0, a1)].groupby("cell_id")[cols].mean()
    m1 = d[d["year"].between(b0, b1)].groupby("cell_id")[cols].mean()
    c = pd.DataFrame({"dln_et": 100 * np.log(m1["et_year"] / m0["et_year"]),
                      "dln_eto": 100 * np.log(m1["eto_year"] / m0["eto_year"]),
                      "dtcc": m1["tcc_pct"] - m0["tcc_pct"],
                      "tcc0": m0["tcc_pct"]}).dropna()
    c["dln_kc"] = c["dln_et"] - c["dln_eto"]
    meta = d.drop_duplicates("cell_id").set_index("cell_id")[["region", "citrus_ha", "block"]]
    return c.join(meta).join(soil.set_index("cell_id")).dropna(subset=["aws0_100"])


def wls(X, y, w):
    XtW = X.T * w
    return np.linalg.solve(XtW @ X, XtW @ y)


def boot(c, fn):
    blocks = c["block"].values
    ub = np.unique(blocks)
    idx = {u: np.where(blocks == u)[0] for u in ub}
    vals = []
    for _ in range(NBOOT):
        ii = np.concatenate([idx[u] for u in RNG.choice(ub, len(ub))])
        vals.append(fn(c.iloc[ii]))
    return np.percentile(np.array(vals), [2.5, 97.5], axis=0)


def regression(c, y, aws="aws0_100"):
    def fit(cc):
        X = np.column_stack([np.ones(len(cc)), cc["dtcc"], cc[aws] - c[aws].mean()])
        return wls(X, cc[y].values, cc["citrus_ha"].values)
    est = fit(c)
    lo, hi = boot(c, fit)
    return {"intercept": est[0], "int_lo": lo[0], "int_hi": hi[0],
            "per_pp_tcc": est[1], "tcc_lo": lo[1], "tcc_hi": hi[1],
            "per_10mm_aws": 10 * est[2], "aws_lo": 10 * lo[2], "aws_hi": 10 * hi[2], "n": len(c)}


def wmean(cc, col):
    return np.average(cc[col], weights=cc["citrus_ha"])


def main():
    d, soil, cells = data()
    # 1. Canopy trajectories.
    traj = (pd.concat([d, d.assign(region="All")])
            .groupby(["region", "year"])
            .apply(lambda g: np.average(g["tcc_pct"], weights=g["citrus_ha"]), include_groups=False)
            .rename("tcc_pct").reset_index())
    traj.to_csv(PROC / "structure_soil_tcc_by_region.csv", index=False, float_format="%.2f")
    print("Tree canopy cover (%), area-weighted:")
    print(traj.pivot(index="year", columns="region", values="tcc_pct").round(1).T.to_string())

    # 2. Panel with and without canopy.
    pans = [panel(d, "All")] + [panel(d[d["region"] == r], r) for r in sorted(d["region"].unique())]
    pans = pd.DataFrame(pans)
    pans.to_csv(PROC / "structure_soil_panel.csv", index=False, float_format="%.3f")
    print("\nTwo-way FE panel: change in year effects (x100 ln-units), without / with TCC:")
    print(pans.round(2).to_string(index=False))

    # 3-4. Cell-level changes.
    regs, strata = [], []
    for period in PERIODS:
        c = cell_changes(d, soil, period)
        c["aws_tercile"] = pd.qcut(c["aws0_100"], 3, labels=["low AWS", "mid AWS", "high AWS"])
        for y in ("dln_et", "dln_kc"):
            for reg_name, cc in [("All", c)] + [(r, c[c["region"] == r]) for r in sorted(c["region"].unique())]:
                regs.append({"period": period, "response": y, "group": reg_name, **regression(cc, y)})
        st = c[c["dtcc"].abs() <= STABLE_PP]
        for reg_name, cc in [("All", st)] + [(r, st[st["region"] == r]) for r in sorted(st["region"].unique())]:
            for terc, g in cc.groupby("aws_tercile", observed=True):
                if len(g) < 15:
                    continue
                lo, hi = boot(g, lambda x: np.array([wmean(x, "dln_et"), wmean(x, "dln_kc")]))
                strata.append({"period": period, "group": reg_name, "aws_tercile": terc,
                               "cells": len(g), "aws_median": g["aws0_100"].median(),
                               "dtcc_mean": wmean(g, "dtcc"),
                               "dln_et": wmean(g, "dln_et"), "et_lo": lo[0], "et_hi": hi[0],
                               "dln_kc": wmean(g, "dln_kc"), "kc_lo": lo[1], "kc_hi": hi[1]})
        c.to_csv(PROC / f"structure_soil_cells_{period}.csv", float_format="%.3f")
    regs, strata = pd.DataFrame(regs), pd.DataFrame(strata)
    regs.to_csv(PROC / "structure_soil_regression.csv", index=False, float_format="%.3f")
    strata.to_csv(PROC / "structure_soil_strata.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 230)
    print("\nCell-level change ~ dTCC + AWS (intercept = average-soil cell with no canopy change):")
    print(regs.round(2).to_string(index=False))
    print(f"\nStable-canopy cells (|dTCC| <= {STABLE_PP} pp) by AWS tercile:")
    print(strata.round(2).to_string(index=False))


def otsu(counts, edges):
    c = np.asarray(counts, float)
    mids = (edges[:-1] + edges[1:]) / 2
    best, thr = -1.0, np.nan
    for k in range(1, len(c)):
        w0, w1 = c[:k].sum(), c[k:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (c[:k] * mids[:k]).sum() / w0, (c[k:] * mids[k:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, edges[k]
    return thr


def naip_indices(cells):
    """Canopy index per cell-year under three threshold rules, from the saved histograms:
    cell (Otsu threshold per cell and year; the main index), year (one Otsu threshold per
    acquisition year from the pooled histogram of all cells) and year_region (one per year
    and region). Also the acquisition day of year."""
    h = pd.read_csv(PROC / "naip_hist_cells.csv").merge(cells[["cell_id", "region"]], on="cell_id")
    edges = np.linspace(-0.4, 0.8, 61)
    H = np.vstack(h["counts"].str.split(";").map(lambda x: np.array(x, float)).values)
    out = h[["cell_id", "year", "region", "doy"]].copy()
    out["thr_cell"] = [otsu(r, edges) for r in H]
    pooled = {}
    for key, cols in (("year", ["year"]), ("year_region", ["year", "region"])):
        thr = {}
        for k, g in out.groupby(cols):
            thr[k if isinstance(k, tuple) else (k,)] = otsu(H[g.index].sum(0), edges)
        pooled[key] = thr
        out[f"thr_{key}"] = [thr[tuple(r)]
                             for r in out[cols].itertuples(index=False)]
    tot = H.sum(1)
    for key in ("cell", "year", "year_region"):
        above = edges[:-1][None, :] >= out[f"thr_{key}"].values[:, None]
        out[f"canopy_{key}"] = (H * above).sum(1) / tot
    return out, pooled


def naip_changes(nc, var):
    p = nc.pivot_table(index="cell_id", columns="year", values=var)
    base = p[[y for y in (2017, 2019) if y in p]].mean(axis=1)
    return 100 * (p[2023] - base), 100 * base


def naip_analysis():
    """Recent ET change against the NAIP canopy index change (2017/2019 -> 2023) and soil water
    storage over the citrus pixels, with sensitivity to the ET window, the threshold rule, the
    stable-canopy limit and the soil layer (second internal review)."""
    d, _, cells = data()
    soil = pd.read_csv(PROC / "soils_cells_citrus.csv")
    nc, pooled = naip_indices(cells)
    nc.to_csv(PROC / "naip_canopy_indices.csv", index=False, float_format="%.4f")
    regs, strata, stab = [], [], []
    for period in ("recent", "aligned", "to2023"):
        c0 = cell_changes(d, soil, period)
        for rule in ("cell", "year", "year_region"):
            dn, base = naip_changes(nc, f"canopy_{rule}")
            c = c0.join(dn.rename("dnaip")).join(base.rename("naip0")).dropna(subset=["dnaip"])
            c["dtcc_saved"] = c["dtcc"]
            c["dtcc"] = c["dnaip"]
            for aws in ("aws0_100", "aws0_30"):
                if aws == "aws0_30" and (period, rule) != ("recent", "cell"):
                    continue
                for y in ("dln_et", "dln_kc"):
                    for g_name, cc in [("All", c)] + [(r, c[c["region"] == r]) for r in sorted(c["region"].unique())]:
                        regs.append({"period": period, "rule": rule, "aws": aws, "response": y,
                                     "group": g_name, **regression(cc.dropna(subset=[aws]), y, aws)})
            if rule != "cell":
                continue
            c["aws_tercile"] = pd.qcut(c["aws0_100"], 3, labels=["low AWS", "mid AWS", "high AWS"])
            for lim in (3, 5, 7.5, 10):
                st = c[c["dnaip"].abs() <= lim]
                for g_name, cc in [("All", st)] + [(r, st[st["region"] == r]) for r in sorted(st["region"].unique())]:
                    if len(cc) < 15:
                        continue
                    lo, hi = boot(cc, lambda x: np.array([wmean(x, "dln_et"), wmean(x, "dln_kc")]))
                    tot = c if g_name == "All" else c[c["region"] == g_name]
                    stab.append({"period": period, "limit_pp": lim, "group": g_name, "cells": len(cc),
                                 "share_cells": len(cc) / len(tot),
                                 "citrus_ha": cc["citrus_ha"].sum(),
                                 "share_area": cc["citrus_ha"].sum() / tot["citrus_ha"].sum(),
                                 "dnaip_mean": wmean(cc, "dnaip"), "dln_et": wmean(cc, "dln_et"),
                                 "et_lo": lo[0], "et_hi": hi[0], "dln_kc": wmean(cc, "dln_kc"),
                                 "kc_lo": lo[1], "kc_hi": hi[1]})
                if lim == 5:
                    for g_name, cc in [("All", st)] + [(r, st[st["region"] == r]) for r in sorted(st["region"].unique())]:
                        for terc, g in cc.groupby("aws_tercile", observed=True):
                            if len(g) < 15:
                                continue
                            lo, hi = boot(g, lambda x: np.array([wmean(x, "dln_et"), wmean(x, "dln_kc")]))
                            strata.append({"period": period, "group": g_name, "aws_tercile": terc,
                                           "cells": len(g), "aws_median": g["aws0_100"].median(),
                                           "dnaip_mean": wmean(g, "dnaip"), "dln_et": wmean(g, "dln_et"),
                                           "et_lo": lo[0], "et_hi": hi[0], "dln_kc": wmean(g, "dln_kc"),
                                           "kc_lo": lo[1], "kc_hi": hi[1]})
            if period == "recent":
                c.to_csv(PROC / "structure_soil_cells_naip.csv", float_format="%.5f")
                main_c = c
    regs, strata, stab = pd.DataFrame(regs), pd.DataFrame(strata), pd.DataFrame(stab)
    regs.to_csv(PROC / "structure_soil_regression_naip.csv", index=False, float_format="%.5f")
    strata.to_csv(PROC / "structure_soil_strata_naip.csv", index=False, float_format="%.5f")
    stab.to_csv(PROC / "structure_soil_stable_naip.csv", index=False, float_format="%.5f")
    # Summary of the index by region, flight dates and thresholds.
    c = main_c
    summ = c.groupby("region").apply(lambda g: pd.Series({
        "naip_2017_19_pct": wmean(g, "naip0"), "dnaip_pp": wmean(g, "dnaip"),
        "share_cells_loss_gt10pp": float((g["dnaip"] < -10).mean()),
        "corr_dnaip_dtcc": float(np.corrcoef(g["dnaip"], g["dtcc_saved"])[0, 1])}),
        include_groups=False)
    summ.loc["All"] = [wmean(c, "naip0"), wmean(c, "dnaip"), float((c["dnaip"] < -10).mean()),
                       float(np.corrcoef(c["dnaip"], c["dtcc_saved"])[0, 1])]
    summ.to_csv(PROC / "structure_soil_naip_summary.csv", float_format="%.5f")
    dates = (nc.assign(month=pd.to_datetime(nc["year"].astype(str), format="%Y")
                       + pd.to_timedelta(nc["doy"] - 1, unit="D"))
             .groupby(["year", "region"]).agg(
                 doy_median=("doy", "median"), doy_p10=("doy", lambda x: x.quantile(0.1)),
                 doy_p90=("doy", lambda x: x.quantile(0.9)),
                 thr_cell_median=("thr_cell", "median"), thr_year=("thr_year", "first"),
                 thr_year_region=("thr_year_region", "first")))
    dates.to_csv(PROC / "naip_dates_thresholds.csv", float_format="%.5f")
    pd.set_option("display.width", 240)
    print("NAIP canopy index by region (2017/2019 level, change to 2023):")
    print(summ.round(3).to_string())
    print("\nFlight day of year and thresholds by year and region:")
    print(dates.round(3).to_string())
    key = regs[(regs["response"] == "dln_et") & (regs["aws"] == "aws0_100")]
    print("\nET change ~ dNAIP + AWS (intercept = no canopy change, average AWS):")
    print(key[["period", "rule", "group", "intercept", "int_lo", "int_hi", "per_pp_tcc",
               "tcc_lo", "tcc_hi", "per_10mm_aws", "aws_lo", "aws_hi", "n"]].round(2).to_string(index=False))
    print("\nSoil layer 0-30 cm (recent, cell rule):")
    print(regs[regs["aws"] == "aws0_30"].round(2).to_string(index=False))
    print("\nStable-canopy limit sensitivity:")
    print(stab.round(2).to_string(index=False))
    print("\nStable canopy (|change| <= 5 pp) by AWS tercile:")
    print(strata.round(2).to_string(index=False))


if __name__ == "__main__":
    import sys
    naip_analysis() if "naip" in sys.argv[1:] else main()
