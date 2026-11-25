"""Extract monthly OpenET ET and gridMET ETo over persistent Florida citrus, 2000-2025.

Analysis unit (first version, before FSAID field polygons): 1-km cells (EPSG:5070)
holding at least `--min-frac` of persistent citrus. A 30-m pixel is persistent citrus
when the USDA CDL labels it Citrus (72) or Oranges (212) in at least `--persist` of
the CDL years since 2008 and in at least one of the last two CDL years (so groves
abandoned before the end of the record are excluded).

Steps (each cached on disk, so the script can be re-run and resumes):
    python scripts/02_extract_openet_series.py cells
    python scripts/02_extract_openet_series.py years --start 2000 --end 2025

Outputs:
    data/processed/citrus_cells_1km.csv            one row per cell
    data/processed/series/openet_cells_<year>.csv  one row per cell x month
    data/processed/series/<model>/openet_cells_<year>.csv  with --model (single models)
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import ee
import pandas as pd

MODELS = ["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"]


def openet_asset(model):
    return f"projects/openet/assets/{model}/conus/gridmet/monthly/v2_1"


def et_band(model):
    return "et_ensemble_mad" if model == "ensemble" else "et"


GRIDMET = "IDAHO_EPSCOR/GRIDMET"
CDL = "USDA/NASS/CDL"
CITRUS_CLASSES = [72, 212]
ALBERS = "EPSG:5070"
CELL_M = 1000
# Peninsular citrus belt (all commercial citrus counties fall inside), lon/lat.
CITRUS_BELT_BOX = [-83.0, 25.8, -79.9, 29.4]

ROOT = Path(__file__).resolve().parents[1]
CELLS_CSV = ROOT / "data" / "processed" / "citrus_cells_1km.csv"
SERIES_DIR = ROOT / "data" / "processed" / "series"


def series_dir(model):
    """Ensemble series stay in series/; single models go to series/<model>/."""
    return SERIES_DIR if model == "ensemble" else SERIES_DIR / model


def init_ee():
    try:
        ee.Initialize(project=os.environ["EE_PROJECT"])
    except Exception as exc:
        sys.exit(f"Earth Engine init failed: {exc}")


def florida():
    return ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.eq("NAME", "Florida"))


def persistent_citrus(persist):
    years = sorted(
        int(y) for y in ee.ImageCollection(CDL).aggregate_array("system:index").getInfo()
        if y.isdigit() and int(y) >= 2008
    )
    def is_citrus(y):
        return (ee.Image(f"{CDL}/{y}").select("cropland")
                .remap(CITRUS_CLASSES, [1] * len(CITRUS_CLASSES), 0))
    freq = ee.ImageCollection([is_citrus(y) for y in years]).sum()
    need = math.ceil(persist * len(years))
    recent = is_citrus(years[-1]).Or(is_citrus(years[-2]))
    mask = freq.gte(need).And(recent).selfMask().rename("citrus")
    print(f"CDL years {years[0]}-{years[-1]} (n={len(years)}); persistent = citrus in "
          f">= {need} years and in {years[-2]} or {years[-1]}")
    return mask


def build_cells(args):
    mask = persistent_citrus(args.persist)
    proj = ee.Projection(ALBERS).atScale(CELL_M)
    frac = (mask.unmask(0)
            .setDefaultProjection(ee.Projection(ALBERS).atScale(30))
            .reduceResolution(ee.Reducer.mean(), maxPixels=2048)
            .reproject(proj)
            .rename("frac"))
    keep = frac.gte(args.min_frac)
    # One point per kept 1-km pixel (its centre), sampled in tiles to stay within
    # Earth Engine memory limits. The cell id is derived from the Albers centre below.
    points = frac.updateMask(keep).sample(
        region=ee.Geometry.Rectangle(CITRUS_BELT_BOX).intersection(florida().geometry(), 100),
        projection=proj, scale=CELL_M, geometries=True, dropNulls=True, tileScale=16,
    )
    counties = (ee.FeatureCollection("TIGER/2018/Counties")
                .filter(ee.Filter.eq("STATEFP", "12")))

    def describe(f):
        c = f.geometry()
        cx = c.transform(ALBERS, 1).coordinates()
        county = ee.Feature(counties.filterBounds(c).first()).get("NAME")
        return ee.Feature(None, {
            "citrus_frac": f.get("frac"),
            "lon": c.coordinates().get(0), "lat": c.coordinates().get(1),
            "x_albers": cx.get(0), "y_albers": cx.get(1),
            "county": county,
        })

    df = ee.data.computeFeatures({
        "expression": points.map(describe),
        "fileFormat": "PANDAS_DATAFRAME",
    })
    df = df.drop(columns=[c for c in ("geo", "cell_id") if c in df.columns])
    df.insert(0, "cell_id", cell_ids(df))
    if df["cell_id"].duplicated().any():
        sys.exit("Duplicate cell ids: check the grid projection.")
    df["citrus_ha"] = (df["citrus_frac"] * CELL_M * CELL_M / 1e4).round(1)
    CELLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("cell_id").to_csv(CELLS_CSV, index=False)
    print(f"{len(df)} cells, {df['citrus_ha'].sum():,.0f} ha of persistent citrus -> {CELLS_CSV}")
    print(df.groupby("county")["citrus_ha"].agg(["count", "sum"]).round(0)
          .sort_values("sum", ascending=False).head(15).to_string())


def cell_ids(df):
    """Unique id from the 1-km Albers grid indices: x_index * 10000 + y_index."""
    return ((df["x_albers"] // CELL_M).astype(int) * 10000
            + (df["y_albers"] // CELL_M).astype(int))


def cells_fc(df):
    feats = []
    for r in df.itertuples():
        x0, y0 = r.x_albers - CELL_M / 2, r.y_albers - CELL_M / 2
        geom = ee.Geometry.Rectangle([x0, y0, x0 + CELL_M, y0 + CELL_M],
                                     proj=ALBERS, geodesic=False)
        feats.append(ee.Feature(geom, {"cell_id": int(r.cell_id)}))
    return ee.FeatureCollection(feats)


def year_image(year, mask, model="ensemble"):
    openet = ee.ImageCollection(openet_asset(model))
    gridmet = ee.ImageCollection(GRIDMET).select("eto")
    bands = []
    for m in range(1, 13):
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")
        bands.append(openet.filterDate(start, end).select(et_band(model)).mosaic()
                     .rename(f"et_{m:02d}"))
        bands.append(gridmet.filterDate(start, end).sum().rename(f"eto_{m:02d}"))
    return ee.Image.cat(bands).updateMask(mask)


def extract_years(args):
    if not CELLS_CSV.exists():
        sys.exit("Run the `cells` step first.")
    cells = pd.read_csv(CELLS_CSV)
    fc = cells_fc(cells)
    mask = persistent_citrus(args.persist)
    out_dir = series_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    for year in range(args.start, args.end + 1):
        out = out_dir / f"openet_cells_{year}.csv"
        if out.exists() and not args.force:
            print(f"{year}: cached")
            continue
        t = time.time()
        stats = year_image(year, mask, args.model).reduceRegions(
            collection=fc, reducer=reducer, crs=ALBERS, scale=30, tileScale=8,
        ).map(lambda f: ee.Feature(None, f.toDictionary()))
        wide = ee.data.computeFeatures({"expression": stats,
                                        "fileFormat": "PANDAS_DATAFRAME"})
        rows = []
        for m in range(1, 13):
            rows.append(pd.DataFrame({
                "cell_id": wide["cell_id"].astype(int),
                "year": year, "month": m,
                "et_mm": wide.get(f"et_{m:02d}_mean"),
                "eto_mm": wide.get(f"eto_{m:02d}_mean"),
                "et_px": wide.get(f"et_{m:02d}_count"),
            }))
        long = pd.concat(rows, ignore_index=True).sort_values(["cell_id", "month"])
        long.to_csv(out, index=False)
        print(f"{args.model} {year}: {len(long)} rows, mean ET {long['et_mm'].mean():.1f} mm/month, "
              f"ETo {long['eto_mm'].mean():.1f} mm/month ({time.time() - t:.0f} s)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["cells", "years"])
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--persist", type=float, default=0.85,
                   help="share of CDL years a pixel must be citrus")
    p.add_argument("--min-frac", type=float, default=0.25,
                   help="minimum persistent-citrus share of a 1-km cell")
    p.add_argument("--force", action="store_true", help="recompute cached years")
    p.add_argument("--model", choices=MODELS, default="ensemble",
                   help="OpenET ensemble or one of its six models (years step)")
    args = p.parse_args()
    init_ee()
    build_cells(args) if args.step == "cells" else extract_years(args)


if __name__ == "__main__":
    main()
