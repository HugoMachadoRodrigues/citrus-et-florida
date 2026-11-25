"""Validate OpenET against the only public citrus ET tower in Florida.

USGS 271042081462301, "2X4 Citrus Climate Station near Arcadia" (DeSoto County, flatwoods
orange grove; Jia et al. 2007, Trans. ASABE). Daily ET from May 2004 to May 2005 in NWIS
(parameter 72159). Monthly tower ET (months with >= 27 days, scaled to the full month) is
compared with OpenET ensemble and its six models over the orange pixels (CDL 212/72 in
2008) within `--radius` m of the tower.

Steps:
    python scripts/12_validate_tower.py fetch     # NWIS daily values -> data/raw/usgs/
    python scripts/12_validate_tower.py extract   # OpenET monthly at the tower (Earth Engine)
    python scripts/12_validate_tower.py compare   # bias, RMSE, r, KGE per model; Kc

Outputs: data/processed/tower_monthly.csv, tower_metrics.csv
"""

import argparse
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "usgs" / "2x4_citrus_dv.rdb"
SITE = "271042081462301"
LON, LAT = -81.77285919, 27.17866155
NWIS_URL = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=" + SITE +
            "&startDT=2004-01-01&endDT=2010-12-31"
            "&parameterCd=72159,99772,62608,72124,00052,00020")
MODELS = ["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"]
CITRUS_CLASSES = [72, 212]


def fetch(args):
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(urllib.request.urlopen(NWIS_URL, timeout=120).read())
    print(f"Wrote {RAW}")


def extract(args):
    import ee
    try:
        ee.Initialize(project=os.environ["EE_PROJECT"])
    except Exception as exc:
        sys.exit(f"Earth Engine init failed: {exc}")
    region = ee.Geometry.Point([LON, LAT]).buffer(args.radius)
    mask = (ee.Image("USDA/NASS/CDL/2008").select("cropland")
            .remap(CITRUS_CLASSES, [1] * len(CITRUS_CLASSES), 0).selfMask())
    rows = []
    for model in MODELS:
        band = "et_ensemble_mad" if model == "ensemble" else "et"
        col = ee.ImageCollection(f"projects/openet/assets/{model}/conus/gridmet/monthly/v2_1")
        for ym in pd.period_range("2004-01", "2005-12", freq="M"):
            s = ee.Date.fromYMD(ym.year, ym.month, 1)
            img = (col.filterDate(s, s.advance(1, "month")).select(band).mosaic()
                   .updateMask(mask))
            st = img.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                                  region, 30).getInfo()
            rows.append({"model": model, "year": ym.year, "month": ym.month,
                         "et_mm": st.get(f"{band}_mean"), "px": st.get(f"{band}_count")})
        print(f"{model}: done")
    pd.DataFrame(rows).to_csv(PROC / "tower_openet_monthly.csv", index=False)


def tower_monthly():
    d = pd.read_csv(RAW, sep="\t", comment="#").iloc[1:]
    d["date"] = pd.to_datetime(d["datetime"])
    col = next(c for c in d.columns if c.endswith("_72159_00006"))
    d["et"] = pd.to_numeric(d[col], errors="coerce")
    g = d.groupby([d["date"].dt.year.rename("year"), d["date"].dt.month.rename("month")])
    m = g["et"].agg(["count", "mean"]).reset_index()
    m["days"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=1)).dt.days_in_month
    m = m[m["count"] >= 27]
    m["tower_mm"] = m["mean"] * m["days"]
    return m[["year", "month", "count", "tower_mm"]]


def tower_eto():
    """Central ETo (FAWN, wind held) at the tower: same IDW and station set as script 06."""
    spec = importlib.util.spec_from_file_location("s06", ROOT / "scripts" / "06_fawn_vs_gridmet.py")
    s06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s06)
    fawn = s06.fill_from_neighbour(s06.qc_relative(s06.load_fawn("eto_windclim_mm")))
    st = pd.read_csv(PROC / "fawn_stations.csv")
    share = fawn.groupby("station")["eto_mm"].count() / ((s06.END - s06.START + 1) * 12)
    st = st[st["station"].isin(share[share >= 0.8].index)].copy()
    st["d"] = s06.haversine_km(LAT, LON, st["lat"], st["lon"])
    rows = []
    for (y, mo), g in fawn.merge(st[["station", "d"]], on="station").dropna(
            subset=["eto_mm"]).groupby(["year", "month"]):
        g = g[g["d"] <= 80].nsmallest(3, "d")
        if len(g):
            w = 1 / np.maximum(g["d"], 1) ** 2
            rows.append({"year": y, "month": mo, "eto_mm": (w * g["eto_mm"]).sum() / w.sum()})
    return pd.DataFrame(rows)


def kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (sim.std() / obs.std() - 1) ** 2
                       + (sim.mean() / obs.mean() - 1) ** 2)


def compare(args):
    tw = tower_monthly()
    oe = pd.read_csv(PROC / "tower_openet_monthly.csv")
    eto = tower_eto()
    d = oe.merge(tw, on=["year", "month"]).merge(eto, on=["year", "month"], how="left")
    d.to_csv(PROC / "tower_monthly.csv", index=False, float_format="%.2f")
    rows = []
    for model, g in d.groupby("model", sort=False):
        g = g.dropna(subset=["et_mm"])
        diff = g["et_mm"] - g["tower_mm"]
        rows.append({"model": model, "months": len(g),
                     "tower_mm": g["tower_mm"].sum(), "openet_mm": g["et_mm"].sum(),
                     "bias_pct": 100 * diff.sum() / g["tower_mm"].sum(),
                     "rmse_mm_month": np.sqrt((diff ** 2).mean()),
                     "r": np.corrcoef(g["et_mm"], g["tower_mm"])[0, 1],
                     "kge": kge(g["et_mm"].values, g["tower_mm"].values)})
    met = pd.DataFrame(rows)
    met.to_csv(PROC / "tower_metrics.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 200)
    e = d[d["model"] == "ensemble"].assign(
        kc_tower=lambda x: x["tower_mm"] / x["eto_mm"], kc_openet=lambda x: x["et_mm"] / x["eto_mm"])
    print("Monthly, ensemble vs tower (mm) and Kc with the central ETo:")
    print(e[["year", "month", "count", "tower_mm", "et_mm", "eto_mm", "kc_tower", "kc_openet"]]
          .round(2).to_string(index=False))
    print("\nMetrics per model (monthly totals):")
    print(met.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["fetch", "extract", "compare"])
    p.add_argument("--radius", type=float, default=300, help="footprint radius, m")
    args = p.parse_args()
    {"fetch": fetch, "extract": extract, "compare": compare}[args.step](args)


if __name__ == "__main__":
    main()
