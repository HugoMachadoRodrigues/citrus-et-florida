"""Compare FAWN and gridMET reference ET, and map FAWN ETo onto the citrus cells.

Steps:
    python scripts/06_fawn_vs_gridmet.py gridmet   # gridMET monthly ETo at FAWN stations (Earth Engine)
    python scripts/06_fawn_vs_gridmet.py compare   # bias and trend, FAWN vs gridMET
    python scripts/06_fawn_vs_gridmet.py cells     # FAWN ETo per cell-month (inverse distance)
    python scripts/06_fawn_vs_gridmet.py drivers   # trends of radiation, wind and VPD:
                                                   # FAWN vs gridMET vs ERA5-Land (Earth Engine)

Cell ETo uses a fixed set of stations (valid in >= `--min-share` of months, 2000-2025) so
the interpolation does not change as stations open; each cell-month takes the
inverse-distance-squared mean of the nearest `--k` stations with data within `--max-km`.

Inputs:  data/processed/fawn_station_monthly.csv, fawn_stations.csv (script 05)
         data/processed/citrus_cells_1km.csv (script 02)
Outputs: data/processed/gridmet_at_fawn.csv, eto_station_compare.csv, eto_drivers_trend.csv,
         data/processed/series/fawn_eto_cells.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
START, END = 2000, 2025
PRE_HLB, RECENT = (2000, 2005), (2019, 2025)


def gridmet(args):
    import ee
    try:
        ee.Initialize(project=os.environ["EE_PROJECT"])
    except Exception as exc:
        sys.exit(f"Earth Engine init failed: {exc}")
    st = pd.read_csv(PROC / "fawn_stations.csv")
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"station": int(r.station)})
        for r in st.itertuples()])
    col = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select("eto")
    rows = []
    for year in range(START, END + 1):
        bands = []
        for m in range(1, 13):
            s = ee.Date.fromYMD(year, m, 1)
            bands.append(col.filterDate(s, s.advance(1, "month")).sum().rename(f"m{m:02d}"))
        img = ee.Image.cat(bands)
        res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=4000)
        df = ee.data.computeFeatures({"expression": res.map(
            lambda f: ee.Feature(None, f.toDictionary())), "fileFormat": "PANDAS_DATAFRAME"})
        for m in range(1, 13):
            rows.append(pd.DataFrame({"station": df["station"].astype(int), "year": year,
                                      "month": m, "eto_gridmet_mm": df[f"m{m:02d}"]}))
        print(f"{year}: {len(df)} stations")
    pd.concat(rows).to_csv(PROC / "gridmet_at_fawn.csv", index=False, float_format="%.2f")


def sen_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(y) - 1)
         for j in range(i + 1, len(y)) if x[j] != x[i]]
    return float(np.median(s)) if s else np.nan


def load_fawn(var):
    f = pd.read_csv(PROC / "fawn_station_monthly.csv")
    return f.assign(eto_mm=f[var])[["station", "year", "month", "eto_mm"]]


def compare(args):
    fawn = qc_relative(load_fawn(args.var))
    gm = pd.read_csv(PROC / "gridmet_at_fawn.csv")
    st = pd.read_csv(PROC / "fawn_stations.csv")
    d = fawn.merge(gm, on=["station", "year", "month"]).dropna(subset=["eto_mm", "eto_gridmet_mm"])
    d = d.merge(st[["station", "name"]], on="station")
    diff = d["eto_gridmet_mm"] - d["eto_mm"]
    print(f"All station-months ({len(d)}): gridMET - FAWN = {diff.mean():+.1f} mm/month "
          f"({diff.sum() / d['eto_mm'].sum():+.1%}), RMSE {np.sqrt((diff ** 2).mean()):.1f}, "
          f"r {np.corrcoef(d['eto_mm'], d['eto_gridmet_mm'])[0, 1]:.3f}")

    # Annual totals for station-years with all 12 months.
    yr = d.groupby(["station", "name", "year"]).agg(
        n=("month", "size"), fawn=("eto_mm", "sum"), gridmet=("eto_gridmet_mm", "sum"))
    yr = yr[yr["n"] == 12].reset_index()
    yr["ratio"] = yr["gridmet"] / yr["fawn"]
    rows = []
    for (s, name), g in yr.groupby(["station", "name"]):
        pre = g[g["year"].between(*PRE_HLB)]
        rec = g[g["year"].between(*RECENT)]
        rows.append({
            "station": s, "name": name, "years": len(g),
            "fawn_mm": g["fawn"].mean(), "gridmet_mm": g["gridmet"].mean(),
            "ratio": g["ratio"].mean(),
            "fawn_sen_mm_yr": sen_slope(g["year"], g["fawn"]),
            "gridmet_sen_mm_yr": sen_slope(g["year"], g["gridmet"]),
            "ratio_sen_per_decade": 10 * sen_slope(g["year"], g["ratio"]),
            "ratio_pre": pre["ratio"].mean() if len(pre) >= 3 else np.nan,
            "ratio_recent": rec["ratio"].mean() if len(rec) >= 3 else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("years", ascending=False)
    sfx = "" if args.var == "eto_mm" else "_windclim"
    out.to_csv(PROC / f"eto_station_compare{sfx}.csv", index=False, float_format="%.4f")
    yr.to_csv(PROC / f"eto_station_annual{sfx}.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 200)
    print("\nPer station (complete years only):")
    print(out.round(3).to_string(index=False))
    long = out[out["years"] >= 20]
    print(f"\nStations with >= 20 complete years: {len(long)}; median gridMET/FAWN ratio "
          f"{long['ratio'].median():.3f}; median ratio trend {long['ratio_sen_per_decade'].median():+.3f}"
          f"/decade; median trend FAWN {long['fawn_sen_mm_yr'].median():+.1f} mm/yr, "
          f"gridMET {long['gridmet_sen_mm_yr'].median():+.1f} mm/yr")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


REL_TOL = 0.15  # max departure of a station's FAWN/gridMET ratio from the network median


def qc_relative(fawn):
    """Drop station-months whose FAWN/gridMET ratio departs from that month's network
    median ratio by more than REL_TOL. gridMET is only the yardstick here: a shared
    regional signal moves every station alike and passes; one station's sensor drift
    does not."""
    gm = pd.read_csv(PROC / "gridmet_at_fawn.csv")
    f = fawn.merge(gm, on=["station", "year", "month"], how="left")
    f["r"] = f["eto_mm"] / f["eto_gridmet_mm"]
    med = f.groupby(["year", "month"])["r"].transform("median")
    n = f.groupby(["year", "month"])["r"].transform("count")
    bad = (n >= 5) & ((f["r"] / med - 1).abs() > REL_TOL)
    out = f.loc[bad].groupby("station").size()
    print(f"Relative QC dropped {int(bad.sum())} of {int(f['r'].notna().sum())} station-months: "
          + ", ".join(f"{s}:{c}" for s, c in out.sort_values(ascending=False).items()))
    fawn = fawn.copy()
    fawn.loc[bad.values, "eto_mm"] = np.nan
    return fawn


# Station gaps filled from a close neighbour, scaled by their monthly ratio over the
# months both report. Fort Pierce (430) is off in 2014-2020; St. Lucie West (435)
# is 23 km away and runs from late 2013.
FILL_PAIRS = {430: 435}


def fill_from_neighbour(fawn):
    fawn = fawn.copy()
    for target, donor in FILL_PAIRS.items():
        a = fawn[fawn["station"] == target].set_index(["year", "month"])["eto_mm"]
        b = fawn[fawn["station"] == donor].set_index(["year", "month"])["eto_mm"]
        both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        ratio = both.groupby(level="month").apply(lambda g: g["a"].sum() / g["b"].sum())
        full = pd.MultiIndex.from_product([range(START, END + 1), range(1, 13)],
                                          names=["year", "month"])
        a = a.reindex(full)  # months with no record at all become NaN rows
        fill = b.reindex(full) * ratio.reindex(full.get_level_values("month")).values
        n = int((a.isna() & fill.notna()).sum())
        filled = a.fillna(fill).rename("eto_mm").reset_index().assign(station=target)
        fawn = pd.concat([fawn[fawn["station"] != target], filled], ignore_index=True)
        print(f"Station {target}: {n} months filled from {donor} "
              f"({len(both)} overlap months, ratio {ratio.min():.2f}-{ratio.max():.2f})")
    return fawn


def cells(args):
    fawn = load_fawn(args.var)
    st = pd.read_csv(PROC / "fawn_stations.csv")
    cl = pd.read_csv(PROC / "citrus_cells_1km.csv")
    fawn = fill_from_neighbour(qc_relative(fawn))
    n_months = (END - START + 1) * 12
    share = fawn.groupby("station")["eto_mm"].count() / n_months
    keep = share[share >= args.min_share].index
    st = st[st["station"].isin(keep)].reset_index(drop=True)
    print(f"Fixed station set ({len(st)} stations, >= {args.min_share:.0%} valid months): "
          + ", ".join(f"{r.name} ({share[r.station]:.0%})" for r in st.itertuples()))

    dist = haversine_km(cl["lat"].values[:, None], cl["lon"].values[:, None],
                        st["lat"].values[None, :], st["lon"].values[None, :])
    wide = (fawn[fawn["station"].isin(keep)]
            .pivot_table(index=["year", "month"], columns="station", values="eto_mm")
            .reindex(columns=st["station"]))
    out = []
    for (year, month), vals in wide.iterrows():
        v = vals.values
        d = np.where(np.isnan(v)[None, :] | (dist > args.max_km), np.inf, dist)
        idx = np.argsort(d, axis=1)[:, :args.k]
        dk = np.take_along_axis(d, idx, axis=1)
        w = np.where(np.isfinite(dk), 1 / np.maximum(dk, 1.0) ** 2, 0)
        vk = np.nan_to_num(v[idx])
        eto = np.where(w.sum(1) > 0, (w * vk).sum(1) / np.where(w.sum(1) > 0, w.sum(1), 1), np.nan)
        out.append(pd.DataFrame({"cell_id": cl["cell_id"], "year": year, "month": month,
                                 "eto_fawn_mm": eto,
                                 "nearest_km": dk[:, 0]}))
    out = pd.concat(out, ignore_index=True)
    suffix = "" if args.var == "eto_mm" else "_windclim"
    path = PROC / "series" / f"fawn_eto_cells{suffix}.csv"
    out.to_csv(path, index=False, float_format="%.2f")
    near = dist.min(axis=1)
    print(f"Cells: nearest kept station median {np.median(near):.0f} km, max {near.max():.0f} km; "
          f"cell-months with FAWN ETo: {out['eto_fawn_mm'].notna().mean():.1%} -> {path}")


LONG_STATIONS = [280, 290, 302, 303, 304, 320, 330, 360, 380, 450]  # records from 2000


def ee_init():
    import ee
    try:
        ee.Initialize(project=os.environ["EE_PROJECT"])
    except Exception as exc:
        sys.exit(f"Earth Engine init failed: {exc}")
    return ee


def drivers(args):
    """Why do FAWN and gridMET ETo trend apart? Annual means of the ETo drivers at the
    long-record stations from FAWN, gridMET and ERA5-Land (independent of both), and
    their Sen trends in % per decade."""
    ee = ee_init()
    st = pd.read_csv(PROC / "fawn_stations.csv")
    st = st[st["station"].isin(LONG_STATIONS)]
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"station": int(r.station)})
        for r in st.itertuples()])
    gm = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").select(["srad", "vs", "vpd"])
    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(
        ["surface_solar_radiation_downwards_sum", "u_component_of_wind_10m",
         "v_component_of_wind_10m"])
    rows = []
    for year in range(START, END + 1):
        s = ee.Date.fromYMD(year, 1, 1)
        g = gm.filterDate(s, s.advance(1, "year")).mean().rename(
            ["gm_srad", "gm_wind10", "gm_vpd"])
        a = era.filterDate(s, s.advance(1, "year")).mean()
        img = g.addBands([
            a.select("surface_solar_radiation_downwards_sum").divide(86400).rename("era_srad"),
            a.select("u_component_of_wind_10m").hypot(a.select("v_component_of_wind_10m"))
             .rename("era_wind10")])
        df = ee.data.computeFeatures({"expression": img.reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=4000).map(
            lambda f: ee.Feature(None, f.toDictionary())), "fileFormat": "PANDAS_DATAFRAME"})
        rows.append(df.assign(year=year))
    grid = pd.concat(rows)

    d = pd.read_csv(PROC / "fawn_daily.csv", parse_dates=["date"])
    d = d[d["station"].isin(LONG_STATIONS)].assign(year=lambda x: x["date"].dt.year)
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    d["vpd"] = 0.5 * (es(d["tmax"]) + es(d["tmin"])) - es(d["tdew"])
    f = d.groupby(["station", "year"]).agg(
        n=("rs_wm2", "count"), fawn_srad=("rs_wm2", "mean"),
        fawn_wind10=("wind10", "mean"), fawn_vpd=("vpd", "mean")).reset_index()
    t = f[f["n"] >= 330].merge(grid, on=["station", "year"])
    t.to_csv(PROC / "eto_drivers_annual.csv", index=False, float_format="%.3f")

    cols = ["fawn_srad", "gm_srad", "era_srad", "fawn_wind10", "gm_wind10", "era_wind10",
            "fawn_vpd", "gm_vpd"]
    res = []
    for s_, h in t.groupby("station"):
        row = {"station": s_, "years": len(h)}
        for c in cols:
            row[c] = 1000 * sen_slope(h["year"], h[c]) / h[c].mean()  # % per decade
        res.append(row)
    res = pd.DataFrame(res)
    res.to_csv(PROC / "eto_drivers_trend.csv", index=False, float_format="%.2f")
    pd.set_option("display.width", 200)
    print("Sen trend, % per decade (annual means; srad W/m2, wind 10 m, VPD kPa):")
    print(res.round(1).to_string(index=False))
    print("\nMedian across stations:")
    print(res[cols].median().round(1).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["gridmet", "compare", "cells", "drivers"])
    p.add_argument("--min-share", type=float, default=0.8)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--max-km", type=float, default=80)
    p.add_argument("--var", choices=["eto_mm", "eto_windclim_mm"], default="eto_mm",
                   help="FAWN ETo column to map (windclim = wind held at monthly climatology)")
    args = p.parse_args()
    {"gridmet": gridmet, "compare": compare, "cells": cells,
     "drivers": drivers}[args.step](args)


if __name__ == "__main__":
    main()
