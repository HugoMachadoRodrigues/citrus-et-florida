"""Summarize the monthly cell series into annual ET, ET0 and Kc_act (ET/ET0) by region.

Reads   data/processed/citrus_cells_1km.csv and data/processed/series/[<model>/]openet_cells_*.csv,
        the ET0 series of all four scenarios (script 02, script 17 and script 06)
Writes  data/processed/annual_cells[_<eto>][_<model>].csv   one row per cell x year
        data/processed/annual_region[_<eto>][_<model>].csv  one row per region x year

Construction (revised after the second internal review):
  * ET is built once, independently of the ET0 scenario. A cell-month is valid when its
    valid ET pixels reach at least half of that cell's maximum monthly pixel count. Years
    with 12 valid months are summed directly. Years with 9-11 valid months are completed
    with a single scenario-free imputation: the valid-month sum is scaled by the ratio of
    the cell's mean annual ET to its mean ET over the same calendar months (ratio to the
    cell's own ET climatology). Cell-years with fewer than 9 valid months are dropped.
  * ET0 is summed over the 12 months for each scenario. The FAWN scenarios have a few
    cell-months without any station within 80 km (mostly 2000-2001, June 2007 and
    August-September 2009); up to 4 such months per cell-year are filled with the cell's
    calendar-month mean of the same scenario.
  * Common support: a cell-year is kept only if it has ET and full-year ET0 under all four
    scenarios, so every scenario (and every model) uses the same cells and weights.
  * Kc_act = ET / ET0 per cell-year. Regional values are ratios of citrus-area-weighted
    means: Kc_r = sum(w ET) / sum(w ET0) over the same cell-years, so that
    dln Kc = dln ET - dln ET0 holds exactly at every level of aggregation.
  * Flags: et_complete (12 valid ET months) and eto_filled (number of filled ET0 months,
    maximum over scenarios) allow the complete-year sensitivity of script 21.
"""

import argparse
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

REGIONS = {
    "Ridge": ["Polk", "Highlands", "Lake", "Osceola", "Orange"],
    "Indian River": ["Indian River", "St. Lucie", "Martin", "Okeechobee", "Brevard"],
    # everything else in the cell table (DeSoto, Hardee, Hendry, Collier, ...) is
    # southwest flatwoods
}
PRE_HLB = (2000, 2005)
RECENT = (2019, 2025)
ETO_SCENARIOS = ["gridmet", "gridmet_bc", "fawn", "fawn_windclim"]
MAX_ETO_FILL = 4
MIN_ET_MONTHS = 9


def region_of(county):
    for name, counties in REGIONS.items():
        if county in counties:
            return name
    return "Southwest Flatwoods"


def mann_kendall(y):
    """Mann-Kendall S, two-sided p (normal approx., no tie correction) and Sen slope."""
    y = np.asarray(y, float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / math.sqrt(var) if s != 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return s, p, float(np.median(slopes))


def hinge_break(years, y, lo=2003, hi=2022):
    """Continuous two-segment fit; returns the break year with the lowest SSE."""
    years, y = np.asarray(years, float), np.asarray(y, float)
    best = None
    for b in range(lo, hi + 1):
        X = np.column_stack([np.ones_like(years), years - b, np.clip(years - b, 0, None)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((X @ coef - y) ** 2).sum())
        if best is None or sse < best[1]:
            best = (b, sse, coef)
    b, _, coef = best
    return b, coef[1], coef[1] + coef[2]  # break year, slope before, slope after


def load_et(model="ensemble"):
    sdir = PROC / "series" if model == "ensemble" else PROC / "series" / model
    m = pd.concat([pd.read_csv(f) for f in sorted(sdir.glob("openet_cells_*.csv"))],
                  ignore_index=True)
    return m


@lru_cache(maxsize=None)
def eto_monthly(scenario):
    """Cell-month ET0 (mm) for one scenario, with the FAWN gap fill; columns cell_id, year,
    month, eto_mm, filled."""
    if scenario == "gridmet":
        e = load_et("ensemble")[["cell_id", "year", "month", "eto_mm"]]
    elif scenario == "gridmet_bc":
        e = pd.concat([pd.read_csv(f) for f in sorted(
            (PROC / "series" / "extra").glob("eto_bc_cells_*.csv"))], ignore_index=True)
        e = e.rename(columns={"value": "eto_mm"})[["cell_id", "year", "month", "eto_mm"]]
    else:
        sfx = scenario.removeprefix("fawn")
        e = pd.read_csv(PROC / "series" / f"fawn_eto_cells{sfx}.csv")
        e = e.rename(columns={"eto_fawn_mm": "eto_mm"})[["cell_id", "year", "month", "eto_mm"]]
    e = e.copy()
    e["filled"] = e["eto_mm"].isna()
    clim = e.groupby(["cell_id", "month"])["eto_mm"].transform("mean")
    nmiss = e.groupby(["cell_id", "year"])["filled"].transform("sum")
    fill_ok = e["filled"] & (nmiss <= MAX_ETO_FILL)
    e.loc[fill_ok, "eto_mm"] = clim[fill_ok]
    return e


@lru_cache(maxsize=None)
def eto_annual(scenario):
    e = eto_monthly(scenario)
    g = e.groupby(["cell_id", "year"])
    a = pd.DataFrame({"eto_year": g["eto_mm"].sum(min_count=12).where(g["eto_mm"].count() == 12),
                      "eto_filled": g["filled"].sum()})
    return a


def common_support():
    """Cell-years with full-year ET0 (after the gap fill) under all four scenarios."""
    frames = [eto_annual(s)["eto_year"].rename(s) for s in ETO_SCENARIOS]
    a = pd.concat(frames, axis=1)
    ok = a.notna().all(axis=1)
    filled = pd.concat([eto_annual(s)["eto_filled"].rename(s) for s in ETO_SCENARIOS],
                       axis=1).max(axis=1)
    return ok, filled


def annual_et(m):
    """Scenario-free annual ET per cell-year (see module docstring)."""
    m = m.copy()
    max_px = m.groupby("cell_id")["et_px"].transform("max")
    m["valid"] = (m["et_px"] >= 0.5 * max_px) & m["et_mm"].notna()
    v = m[m["valid"]]
    clim = v.groupby(["cell_id", "month"])["et_mm"].mean().rename("clim")
    clim_year = clim.groupby("cell_id").sum().rename("clim_year")
    v = v.join(clim, on=["cell_id", "month"])
    a = v.groupby(["cell_id", "year"]).agg(et_valid=("et_mm", "sum"), clim_valid=("clim", "sum"),
                                           months=("month", "size"))
    a = a.join(clim_year, on="cell_id")
    a["et_year"] = np.where(a["months"] == 12, a["et_valid"],
                            a["et_valid"] * a["clim_year"] / a["clim_valid"])
    a["et_complete"] = a["months"] == 12
    a = a[a["months"] >= MIN_ET_MONTHS]
    return a[["et_year", "months", "et_complete"]], m["valid"].mean()


def build(eto="fawn_windclim", model="ensemble", write=True, verbose=True):
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    cells["region"] = cells["county"].map(region_of)
    et, share_valid = annual_et(load_et(model))
    ok, filled = common_support()
    a = et.join(eto_annual(eto)["eto_year"], how="inner")
    a = a.join(ok.rename("support")).join(filled.rename("eto_filled"))
    a = a[a["support"].fillna(False).astype(bool)].drop(columns="support")
    a["kc"] = a["et_year"] / a["eto_year"]
    a = a.reset_index().merge(cells[["cell_id", "county", "region", "citrus_ha"]], on="cell_id")
    reg = aggregate(a)
    if write:
        suffix = ("" if eto == "gridmet" else f"_{eto}") + ("" if model == "ensemble" else f"_{model}")
        a.to_csv(PROC / f"annual_cells{suffix}.csv", index=False)
        reg.to_csv(PROC / f"annual_region{suffix}.csv", index=False)
    if verbose:
        print(f"Reference ET: {eto}; OpenET: {model}")
        print(f"Cell-months with valid ET: {share_valid:.1%}; cell-years kept: {len(a):,} "
              f"(complete ET years {a['et_complete'].mean():.1%}; "
              f"with any filled ET0 month {(a['eto_filled'] > 0).mean():.1%})")
    return a, reg


def aggregate(a, weight="citrus_ha"):
    """Ratio-of-weighted-means aggregation by region (and all cells) and year."""
    rows = []
    for (region, year), g in pd.concat([a, a.assign(region="All")]).groupby(["region", "year"]):
        w = g[weight]
        et, eto = np.average(g["et_year"], weights=w), np.average(g["eto_year"], weights=w)
        rows.append({"region": region, "year": year, "cells": len(g), "et_mm": et,
                     "eto_mm": eto, "kc": et / eto})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eto", choices=ETO_SCENARIOS, default="gridmet")
    p.add_argument("--model", choices=["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl",
                                       "sims", "ssebop"], default="ensemble")
    args = p.parse_args()
    a, reg = build(args.eto, args.model)
    print("\nAnnual Kc_act, ratio of area-weighted means:")
    print(reg.pivot(index="year", columns="region", values="kc").round(3).to_string())
    print("\nAnnual ET (mm), area-weighted:")
    print(reg.pivot(index="year", columns="region", values="et_mm").round(0).to_string())
    print("\nRegion        pre-HLB Kc  recent Kc  change   Sen slope/yr  MK p     break")
    for region, g in reg.groupby("region"):
        g = g.sort_values("year")
        pre, rec = g[g["year"].between(*PRE_HLB)], g[g["year"].between(*RECENT)]
        kpre, krec = pre["et_mm"].mean() / pre["eto_mm"].mean(), rec["et_mm"].mean() / rec["eto_mm"].mean()
        _, pv, sen = mann_kendall(g["kc"])
        b, s1, s2 = hinge_break(g["year"], g["kc"])
        print(f"{region:13} {kpre:9.3f} {krec:10.3f} {100 * np.log(krec / kpre):+7.1f}  "
              f"{sen:+12.4f}  {pv:6.3f}  {b}")


if __name__ == "__main__":
    main()
