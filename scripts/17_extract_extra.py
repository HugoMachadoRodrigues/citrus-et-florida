"""Extra per-cell monthly series requested in review (Earth Engine).

Steps (cached per year, resumable):
    python scripts/17_extract_extra.py eto_bc   # OpenET bias-corrected gridMET ETo
    python scripts/17_extract_extra.py count    # clear-sky Landsat observations per month

eto_bc: projects/openet/assets/reference_et/conus/gridmet/monthly/v1, band "eto" (grass
        reference, mm/month, scale factor 1) -- the reference ET used inside OpenET v2.1,
        unlike the raw IDAHO_EPSCOR/GRIDMET "eto" used in script 02.
count:  band "count" of the OpenET SSEBop v2.1 monthly asset: number of cloud-free Landsat
        observations used to build each pixel's monthly value (mean over the cell's
        persistent-citrus pixels).

Outputs: data/processed/series/extra/<step>_cells_<year>.csv (cell_id, year, month, value)
"""

import argparse
import importlib.util
import time
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "series" / "extra"
SOURCES = {
    "eto_bc": ("projects/openet/assets/reference_et/conus/gridmet/monthly/v1", "eto", 4000),
    "count": ("projects/openet/assets/ssebop/conus/gridmet/monthly/v2_1", "count", 30),
}


def load_s02():
    spec = importlib.util.spec_from_file_location(
        "s02", ROOT / "scripts" / "02_extract_openet_series.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=list(SOURCES))
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2025)
    args = p.parse_args()
    s02 = load_s02()
    s02.init_ee()
    cells = pd.read_csv(s02.CELLS_CSV)
    fc = s02.cells_fc(cells)
    mask = s02.persistent_citrus(0.85)
    asset, band, scale = SOURCES[args.step]
    col = ee.ImageCollection(asset).select(band)
    OUT.mkdir(parents=True, exist_ok=True)
    for year in range(args.start, args.end + 1):
        out = OUT / f"{args.step}_cells_{year}.csv"
        if out.exists():
            print(f"{year}: cached")
            continue
        t = time.time()
        bands = []
        for m in range(1, 13):
            s = ee.Date.fromYMD(year, m, 1)
            bands.append(col.filterDate(s, s.advance(1, "month")).mosaic().rename(f"v{m:02d}"))
        img = ee.Image.cat(bands)
        if args.step == "count":
            img = img.updateMask(mask)
        stats = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), crs=s02.ALBERS,
                                  scale=scale, tileScale=8)
        df = ee.data.computeFeatures({"expression": stats.map(
            lambda f: ee.Feature(None, f.toDictionary())), "fileFormat": "PANDAS_DATAFRAME"})
        rows = [pd.DataFrame({"cell_id": df["cell_id"].astype(int), "year": year, "month": m,
                              "value": df.get(f"v{m:02d}")}) for m in range(1, 13)]
        long = pd.concat(rows, ignore_index=True)
        long.to_csv(out, index=False)
        print(f"{args.step} {year}: mean {long['value'].mean():.2f} ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
