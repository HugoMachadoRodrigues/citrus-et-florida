"""Check that OpenET v2.0 monthly ET has valid values over Florida citrus groves.

Goal: confirm the pre-HLB baseline (2000-2005) exists before committing to the
paper design. For each month and each main citrus county, the script reports the
mean ensemble ET over citrus pixels and the fraction of citrus pixels that hold a
valid ET value.

Usage:
    export EE_PROJECT=<your-gcp-project-id>
    python scripts/01_check_openet_florida.py --start 2000 --end 2005

Output: data/processed/openet_check_<start>_<end>.csv
"""

import argparse
import os
import sys
from pathlib import Path

import ee
import pandas as pd

# OpenET Ensemble, CONUS, gridMET ETo, monthly. Only v2.1 holds the eastern US
# (Florida: Oct 1999 onward); v2.0 covers the western states only.
OPENET_ASSETS = [
    "projects/openet/assets/ensemble/conus/gridmet/monthly/v2_1",
]
ET_BAND = "et_ensemble_mad"  # ensemble value after MAD outlier filtering, mm/month

# Main commercial citrus counties (Ridge and Flatwoods).
CITRUS_COUNTIES = [
    "Polk", "Highlands", "Hardee", "DeSoto", "Hendry",
    "Collier", "Lake", "St. Lucie", "Indian River",
]
# USDA NASS Cropland Data Layer: 72 = "Citrus", 212 = "Oranges". In Florida most
# groves fall in 212 (Polk County, 2010: ~62,000 ha in 212 vs ~1,000 ha in 72).
CDL_CITRUS_CLASSES = [72, 212]

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", type=int, default=2000, help="first year (inclusive)")
    p.add_argument("--end", type=int, default=2005, help="last year (inclusive)")
    p.add_argument("--cdl-year", type=int, default=2010,
                   help="CDL year used for the citrus mask")
    p.add_argument("--scale", type=int, default=60,
                   help="reduction scale in metres (30 = native, slower)")
    return p.parse_args()


def init_ee():
    project = os.environ.get("EE_PROJECT")
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # auth or project errors
        sys.exit(
            f"Earth Engine init failed: {exc}\n"
            "Run `earthengine authenticate` and set EE_PROJECT to your GCP project id."
        )


def load_openet():
    for asset in OPENET_ASSETS:
        try:
            coll = ee.ImageCollection(asset)
            coll.limit(1).size().getInfo()
            print(f"Using OpenET asset: {asset}")
            return coll
        except ee.EEException as exc:
            print(f"  not available: {asset} ({exc})")
    sys.exit("No OpenET asset could be opened. Check access in the Earth Engine catalog.")


def main():
    args = parse_args()
    init_ee()

    openet = load_openet()
    bands = ee.Image(openet.first()).bandNames().getInfo()
    print(f"Bands: {bands}")
    if ET_BAND not in bands:
        sys.exit(f"Band {ET_BAND!r} not found; pick one of the bands above.")

    counties = (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.eq("STATEFP", "12"))
        .filter(ee.Filter.inList("NAME", CITRUS_COUNTIES))
    )

    citrus = (
        ee.Image(f"USDA/NASS/CDL/{args.cdl_year}")
        .select("cropland")
        .remap(CDL_CITRUS_CLASSES, [1] * len(CDL_CITRUS_CLASSES))
        .selfMask()
        .rename("citrus")
    )

    citrus_px = {
        f["properties"]["NAME"]: f["properties"]["count"]
        for f in citrus.reduceRegions(
            collection=counties,
            reducer=ee.Reducer.count(),
            scale=args.scale,
            tileScale=4,
        ).select(["NAME", "count"], None, False).getInfo()["features"]
    }
    print(f"Citrus pixels per county (CDL {args.cdl_year}, {args.scale} m): {citrus_px}")

    # The collection is tiled: each month is stored as many tile images. Restrict to
    # the counties first, then mosaic the tiles of each month into one image.
    region = counties.geometry().bounds()
    openet_fl = openet.filterBounds(region)
    n_months = 12 * (args.end - args.start + 1)
    t0 = ee.Date.fromYMD(args.start, 1, 1)

    def month_image(i):
        start = t0.advance(i, "month")
        tiles = openet_fl.filterDate(start, start.advance(1, "month"))
        empty = ee.Image.constant(0).updateMask(0).rename(ET_BAND).toFloat()
        img = ee.Image(ee.Algorithms.If(
            tiles.size().gt(0), tiles.select(ET_BAND).mosaic(), empty
        ))
        return img.set("date", start.format("YYYY-MM"), "n_tiles", tiles.size())

    monthly = ee.ImageCollection.fromImages(
        ee.List.sequence(0, n_months - 1).map(month_image)
    )
    tiles_per_month = monthly.aggregate_array("n_tiles").getInfo()
    empty_months = sum(1 for n in tiles_per_month if n == 0)
    print(f"Months {args.start}-{args.end}: {n_months}; months with no OpenET tiles over "
          f"the counties: {empty_months}; tiles per month: {min(tiles_per_month)}-{max(tiles_per_month)}")
    if empty_months == n_months:
        sys.exit("No OpenET data in this period: the pre-HLB baseline is missing.")

    reducer = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

    def per_month(img):
        et = img.select(ET_BAND).updateMask(citrus).rename("et")
        date = img.get("date")
        return et.reduceRegions(
            collection=counties, reducer=reducer, scale=args.scale, tileScale=4
        ).map(lambda f: f.set("date", date))

    stats = (
        ee.FeatureCollection(monthly.map(per_month))
        .flatten()
        .select(["NAME", "date", "mean", "count"], None, False)
        .getInfo()["features"]
    )

    rows = []
    for f in stats:
        p = f["properties"]
        total = citrus_px.get(p["NAME"]) or 0
        rows.append({
            "county": p["NAME"],
            "date": p["date"],
            "et_mm": p.get("mean"),
            "valid_px": p.get("count", 0),
            "citrus_px": total,
            "valid_frac": (p.get("count", 0) / total) if total else None,
        })

    df = pd.DataFrame(rows).sort_values(["county", "date"])
    out = ROOT / "data" / "processed" / f"openet_check_{args.start}_{args.end}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")

    df["year"] = df["date"].str[:4]
    summary = df.groupby("year").agg(
        months=("date", "nunique"),
        mean_valid_frac=("valid_frac", "mean"),
        months_below_50pct=("valid_frac", lambda s: int((s < 0.5).sum())),
        mean_monthly_et_mm=("et_mm", "mean"),
    )
    print("\nSummary by year (all counties):")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
