"""Trend inference on the annual regional series: trends, period contrasts, breaks.

For each ET0 scenario and region (annual series from script 03, where regional Kc_act is
the ratio of area-weighted mean ET to area-weighted mean ET0 over a common set of
cell-years):

  * Monotonic trend. Sen slope of annual Kc_act. 95% interval: residual moving-block
    bootstrap (residuals from the Sen line resampled in blocks of 3 consecutive years and
    added back to the line, so the trend and the time axis are kept; 2000 replicates).
    Mann-Kendall test with the Hamed & Rao (1998) variance correction.
  * Period contrasts. For periods A and B, Kc_A = mean(ET)/mean(ET0) over the years of A,
    so that dln Kc = dln ET - dln ET0 exactly (all changes reported as 100 x dln).
    Primary 95% interval: moving-block bootstrap of years within each period (circular
    blocks of 2 consecutive years, ET and ET0 of a year resampled together; 2000
    replicates). Sensitivity intervals: independent resampling of years, and a spatial
    block bootstrap (20-km blocks of cells resampled with replacement, years fixed).
  * Break. Continuous two-segment (hinge) regression, break year searched from 2003 to
    2022. Significance of the best hinge against a straight line uses the sup-F statistic
    with a bootstrap null that repeats the whole search: residuals of the straight-line fit
    are resampled in moving blocks of 3 years, added to the line, and the maximum F over all
    candidate years is recomputed (999 replicates). 95% interval for the break year:
    residual bootstrap around the fitted hinge, repeating the search.

Reads  data/processed/annual_region[_<eto>].csv, annual_cells[_<eto>].csv, citrus_cells_1km.csv
Writes data/processed/trend_robustness.csv
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RNG = np.random.default_rng(20261117)
NBOOT = 2000
NBOOT_BREAK = 999
BLOCK_KM = 20

SCENARIOS = {  # label -> file suffix of script 03 outputs
    "FAWN, wind-standardized": "_fawn_windclim",
    "FAWN": "_fawn",
    "gridMET, OpenET bias-corrected": "_gridmet_bc",
    "gridMET, raw": "",
}
CONTRASTS = [  # (name, reference years, comparison years)
    ("2000-05 vs 2019-25", (2000, 2005), (2019, 2025)),
    ("2015-19 vs 2021-25 (no 2020)", (2015, 2019), (2021, 2025)),
    ("2016-20 vs 2021-25", (2016, 2020), (2021, 2025)),
    ("2010-19 vs 2023-25", (2010, 2019), (2023, 2025)),
    ("2015-19 vs 2021-23", (2015, 2019), (2021, 2023)),
    ("2020 vs 2023-25", (2020, 2020), (2023, 2025)),
]


def sen(y, x=None):
    y = np.asarray(y, float)
    x = np.arange(len(y)) if x is None else np.asarray(x, float)
    i, j = np.triu_indices(len(y), 1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


def mk_hamed_rao(y):
    """Two-sided Mann-Kendall p with the Hamed & Rao (1998) variance correction."""
    y = np.asarray(y, float)
    n = len(y)
    i, j = np.triu_indices(n, 1)
    s = np.sign(y[j] - y[i]).sum()
    var = n * (n - 1) * (2 * n + 5) / 18
    t = np.arange(n)
    resid = y - sen(y) * t
    r = stats.rankdata(resid)
    r = r - r.mean()
    corr = 0.0
    for k in range(1, n - 1):
        rk = (r[:-k] * r[k:]).sum() / (r ** 2).sum()
        if abs(rk) > 1.96 / math.sqrt(n):
            corr += (n - k) * (n - k - 1) * (n - k - 2) * rk
    nstar = 1 + 2 / (n * (n - 1) * (n - 2)) * corr
    var *= max(nstar, 1e-6)
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    return float(math.erfc(abs(z) / math.sqrt(2))), float(nstar)


def mb_index(n, block, circular=False):
    """Indices of one moving-block bootstrap sample of length n."""
    idx = []
    while len(idx) < n:
        if circular:
            st = RNG.integers(0, n)
            idx.extend((st + k) % n for k in range(block))
        else:
            st = RNG.integers(0, n - block + 1)
            idx.extend(range(st, st + block))
    return np.array(idx[:n])


def block_boot_sen(years, y, block=3):
    years, y = np.asarray(years, float), np.asarray(y, float)
    b = sen(y, years)
    a = np.median(y - b * years)
    fit = a + b * years
    res = y - fit
    out = [sen(fit + res[mb_index(len(y), block)], years) for _ in range(NBOOT)]
    return np.percentile(out, [2.5, 97.5])


def dln_ratio(et_a, eto_a, et_b, eto_b):
    """100 x dln of ET, ET0 and Kc between period means."""
    d_et = 100 * np.log(np.mean(et_b) / np.mean(et_a))
    d_eto = 100 * np.log(np.mean(eto_b) / np.mean(eto_a))
    return d_et, d_eto, d_et - d_eto


def contrast(df, a, b, block=2):
    """Point estimate and temporal bootstrap intervals (moving-block and iid) for one contrast."""
    A = df[df["year"].between(*a)][["et_mm", "eto_mm"]].values
    B = df[df["year"].between(*b)][["et_mm", "eto_mm"]].values
    d_et, d_eto, d_kc = dln_ratio(A[:, 0], A[:, 1], B[:, 0], B[:, 1])
    res = {"dln_et": d_et, "dln_eto": d_eto, "dln_kc": d_kc}
    if len(A) < 2 or len(B) < 2:
        return res
    for name, fn in (("mb", lambda n: mb_index(n, min(block, n), circular=True)),
                     ("iid", lambda n: RNG.integers(0, n, n))):
        bs = np.array([dln_ratio(*A[fn(len(A))].T, *B[fn(len(B))].T) for _ in range(NBOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        suffix = "" if name == "mb" else "_iid"
        res.update({f"et_lo{suffix}": lo[0], f"et_hi{suffix}": hi[0],
                    f"kc_lo{suffix}": lo[2], f"kc_hi{suffix}": hi[2]})
    return res


def spatial_ci(cells_df, region, a, b):
    """Spatial block bootstrap (20-km blocks of cells, years fixed) of dln Kc and dln ET."""
    c = cells_df if region == "All" else cells_df[cells_df["region"] == region]
    c = c.assign(wet=c["citrus_ha"] * c["et_year"], weto=c["citrus_ha"] * c["eto_year"])
    years = sorted(c["year"].unique())
    piv = {k: c.pivot_table(index="block", columns="year", values=k, aggfunc="sum")
           .reindex(columns=years).fillna(0.0) for k in ("wet", "weto", "citrus_ha")}
    blocks = piv["wet"].index
    ya = [y for y in years if a[0] <= y <= a[1]]
    yb = [y for y in years if b[0] <= y <= b[1]]
    W, E, O = (piv[k].values for k in ("citrus_ha", "wet", "weto"))
    yrs = np.array(years)
    ia, ib = np.isin(yrs, ya), np.isin(yrs, yb)
    out = []
    for _ in range(NBOOT):
        cnt = np.bincount(RNG.integers(0, len(blocks), len(blocks)), minlength=len(blocks))
        w, e, o = cnt @ W, cnt @ E, cnt @ O
        et, eto = e / w, o / w
        out.append(dln_ratio(et[ia], eto[ia], et[ib], eto[ib]))
    lo, hi = np.percentile(np.array(out), [2.5, 97.5], axis=0)
    return {"et_lo_spatial": lo[0], "et_hi_spatial": hi[0],
            "kc_lo_spatial": lo[2], "kc_hi_spatial": hi[2]}


def hinge_design(years, b):
    return np.column_stack([np.ones_like(years), years - b, np.clip(years - b, 0, None)])


def sup_f(years, y, lo=2003, hi=2022):
    """Best hinge (lowest SSE) and its F statistic against a straight line."""
    n = len(y)
    X0 = np.column_stack([np.ones_like(years), years])
    sse0 = float(((X0 @ np.linalg.lstsq(X0, y, rcond=None)[0] - y) ** 2).sum())
    best = None
    for b in range(lo, hi + 1):
        X = hinge_design(years, b)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[1]:
            best = (b, sse, coef, X @ coef)
    b, sse1, coef, fit = best
    f = (sse0 - sse1) / (sse1 / (n - 3))
    return f, b, coef, fit


def hinge(years, y, lo=2003, hi=2022):
    years, y = np.asarray(years, float), np.asarray(y, float)
    n = len(y)
    f_obs, b, coef, fit = sup_f(years, y, lo, hi)
    p_naive = float(stats.f.sf(f_obs, 1, n - 3))  # break year treated as known (not used)
    # Null distribution of sup-F: straight line + block-resampled residuals, full search.
    X0 = np.column_stack([np.ones_like(years), years])
    line = X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    r0 = y - line
    fs = np.array([sup_f(years, line + r0[mb_index(n, 3)], lo, hi)[0] for _ in range(NBOOT_BREAK)])
    p_boot = float((1 + (fs >= f_obs).sum()) / (1 + NBOOT_BREAK))
    res = y - fit
    bs = [sup_f(years, fit + res[mb_index(n, 3)], lo, hi)[1] for _ in range(500)]
    return {"break_year": b, "slope_before": coef[1], "slope_after": coef[1] + coef[2],
            "sup_f": f_obs, "hinge_p_naive": p_naive, "hinge_p_supf_boot": p_boot,
            "break_ci_lo": np.percentile(bs, 2.5), "break_ci_hi": np.percentile(bs, 97.5)}


def main():
    cl = pd.read_csv(PROC / "citrus_cells_1km.csv")
    cl["block"] = ((cl["x_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str) + "_"
                   + (cl["y_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str))
    rows = []
    for label, sfx in SCENARIOS.items():
        reg = pd.read_csv(PROC / f"annual_region{sfx}.csv")
        cells = pd.read_csv(PROC / f"annual_cells{sfx}.csv").merge(cl[["cell_id", "block"]],
                                                                   on="cell_id")
        for region, g in reg.groupby("region"):
            g = g.sort_values("year").dropna(subset=["kc"])
            yrs, kc = g["year"].values, g["kc"].values
            p_hr, nstar = mk_hamed_rao(kc)
            lo, hi = block_boot_sen(yrs, kc)
            base = {"scenario": label, "region": region, "sen_per_yr": sen(kc, yrs),
                    "sen_ci_lo": lo, "sen_ci_hi": hi, "mk_p_hamed_rao": p_hr,
                    "mk_variance_factor": nstar, **hinge(yrs, kc)}
            for name, a, c in CONTRASTS:
                row = {**base, "contrast": name, **contrast(g, a, c)}
                if name.startswith(("2000-05", "2015-19 vs 2021-25")):
                    row.update(spatial_ci(cells, region, a, c))
                rows.append(row)
    out = pd.DataFrame(rows)
    out["share_from_et"] = out["dln_et"] / out["dln_kc"]
    out.to_csv(PROC / "trend_robustness.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 260)
    first = out.drop_duplicates(["scenario", "region"])
    print(first[["scenario", "region", "sen_per_yr", "sen_ci_lo", "sen_ci_hi", "mk_p_hamed_rao",
                 "break_year", "break_ci_lo", "break_ci_hi", "sup_f", "hinge_p_naive",
                 "hinge_p_supf_boot"]].round(4).to_string(index=False))
    print()
    cols = ["scenario", "region", "contrast", "dln_kc", "kc_lo", "kc_hi", "kc_lo_iid", "kc_hi_iid",
            "kc_lo_spatial", "kc_hi_spatial", "dln_et", "et_lo", "et_hi", "dln_eto", "share_from_et"]
    print(out[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
