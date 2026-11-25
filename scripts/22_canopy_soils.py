"""Grove structure and soil attributes per 1-km cell (review, Tier C).

Steps:
    python scripts/22_canopy_soils.py canopy   # USFS Tree Canopy Cover, annual 2000-2025 (Earth Engine)
    python scripts/22_canopy_soils.py soils    # gNATSGO rasters (Microsoft Planetary Computer)
    python scripts/22_canopy_soils.py soils_citrus  # gNATSGO over persistent-citrus pixels only

canopy: projects/gtac-data-publish/assets/TCC/Product_Version/2025-6, band
        Science_Percent_Tree_Canopy_Cover (%, 30 m, modelled from Landsat and other inputs),
        mean over each cell's persistent-citrus pixels, one value per year.
soils:  gNATSGO (July 2020 release), 10 m in EPSG:5070, read as windows of the cloud-
        optimized GeoTIFFs over each 1-km cell (whole cell, not only citrus pixels):
          aws0_100   available water storage 0-100 cm (mm)
          rootznaws  available water storage in the root zone (mm)
          droughty   share of the cell on droughty soils (0/1 per pixel)
          pwsl1pomu  share of the cell on potential wetland soil landscapes (poorly drained)

Outputs: data/processed/canopy_tcc_cells.csv (cell_id, year, tcc_pct),
         data/processed/soils_cells.csv
"""

import argparse
import importlib.util
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TCC = "projects/gtac-data-publish/assets/TCC/Product_Version/2025-6"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/soils/gnatsgo"
SOIL_VARS = ["aws0_100", "rootznaws", "droughty", "pwsl1pomu"]


def load_s02():
    spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts" / "02_extract_openet_series.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def canopy(args):
    import ee
    s02 = load_s02()
    s02.init_ee()
    cells = pd.read_csv(s02.CELLS_CSV)
    fc = s02.cells_fc(cells)
    mask = s02.persistent_citrus(0.85)
    col = ee.ImageCollection(TCC).select("Science_Percent_Tree_Canopy_Cover")
    rows = []
    for year in range(2000, 2026):
        t = time.time()
        img = col.filter(ee.Filter.eq("year", year)).mosaic().updateMask(mask).rename("tcc")
        st = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), crs=s02.ALBERS,
                               scale=30, tileScale=8)
        df = ee.data.computeFeatures({"expression": st.map(
            lambda f: ee.Feature(None, f.toDictionary())), "fileFormat": "PANDAS_DATAFRAME"})
        rows.append(pd.DataFrame({"cell_id": df["cell_id"].astype(int), "year": year,
                                  "tcc_pct": df.get("mean")}))
        print(f"{year}: mean TCC {rows[-1]['tcc_pct'].mean():.1f}% ({time.time() - t:.0f} s)")
    pd.concat(rows).to_csv(PROC / "canopy_tcc_cells.csv", index=False, float_format="%.2f")


def soils(args):
    import rasterio
    from rasterio.windows import from_bounds
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_CACHEMAX", "1024")
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    body = json.dumps({"collections": ["gnatsgo-rasters"], "bbox": [-83.0, 25.8, -79.9, 29.4],
                       "limit": 50}).encode()
    req = urllib.request.Request(STAC, data=body, headers={"Content-Type": "application/json"})
    items = json.load(urllib.request.urlopen(req, timeout=60))["features"]
    items = [it for it in items if all(v in it["assets"] for v in SOIL_VARS)]
    token = json.load(urllib.request.urlopen(TOKEN, timeout=60))["token"]

    def tile_bounds(it):  # conus_xmin_ymax_xmax_ymin in EPSG:5070
        _, xmin, ymax, xmax, ymin = it["id"].split("_")
        return float(xmin), float(ymin), float(xmax), float(ymax)

    tiles = [(it, tile_bounds(it)) for it in items]
    out = []
    t0 = time.time()
    for it, (x0, y0, x1, y1) in tiles:
        sel = cells[(cells["x_albers"] >= x0) & (cells["x_albers"] < x1)
                    & (cells["y_albers"] >= y0) & (cells["y_albers"] < y1)]
        if sel.empty:
            continue
        srcs = {v: rasterio.open(f"{it['assets'][v]['href']}?{token}") for v in SOIL_VARS}
        for r in sel.itertuples():
            rec = {"cell_id": r.cell_id}
            for v, src in srcs.items():
                win = from_bounds(r.x_albers - 500, r.y_albers - 500, r.x_albers + 500,
                                  r.y_albers + 500, src.transform)
                a = src.read(1, window=win, masked=True).astype(float)
                rec[v] = float(a.mean()) if a.count() else np.nan
            out.append(rec)
        for src in srcs.values():
            src.close()
        print(f"{it['id']}: {len(sel)} cells ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(out).drop_duplicates("cell_id")
    df.to_csv(PROC / "soils_cells.csv", index=False, float_format="%.3f")
    print(f"{len(df)} of {len(cells)} cells with soil data")
    print(df[SOIL_VARS].describe().round(2).to_string())


def soils_citrus(args):
    """gNATSGO over the persistent-citrus pixels of each cell (same support as ET and the
    canopy index; second internal review). The persistent-citrus mask (30-m CDL) is
    sampled in Earth Engine on the gNATSGO 10-m grid (EPSG:5070), one 20-km block at a time, and multiplied with the gNATSGO windows of the same block.
    Writes data/processed/soils_cells_citrus.csv."""
    import ee
    import rasterio
    from rasterio.windows import from_bounds
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    s02 = load_s02()
    s02.init_ee()
    mask = s02.persistent_citrus(0.85).unmask(0).toByte().rename("m")
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    vars_ = ["aws0_30", "aws0_100", "rootznaws", "droughty", "pwsl1pomu"]
    body = json.dumps({"collections": ["gnatsgo-rasters"], "bbox": [-83.0, 25.8, -79.9, 29.4],
                       "limit": 50}).encode()
    req = urllib.request.Request(STAC, data=body, headers={"Content-Type": "application/json"})
    items = [it for it in json.load(urllib.request.urlopen(req, timeout=60))["features"]
             if all(v in it["assets"] for v in vars_)]
    token = json.load(urllib.request.urlopen(TOKEN, timeout=60))["token"]

    def tile_bounds(it):
        _, xmin, ymax, xmax, ymin = it["id"].split("_")
        return float(xmin), float(ymin), float(xmax), float(ymax)

    B, RES, OFF = 20000, 10, 5  # gNATSGO rasters: 10 m, pixel edges at 5 m modulo 10 m
    cells["bx"] = (cells["x_albers"] // B).astype(int)
    cells["by"] = (cells["y_albers"] // B).astype(int)
    out, t0 = [], time.time()
    blocks = list(cells.groupby(["bx", "by"]))
    for n, ((bx, by), g) in enumerate(blocks):
        # Snap the block (plus the cells it holds) to the 30-m grid with edges at 15 mod 30.
        x0 = np.floor((g["x_albers"].min() - 500 - OFF) / RES) * RES + OFF
        y1 = np.ceil((g["y_albers"].max() + 500 - OFF) / RES) * RES + OFF
        x1 = np.ceil((g["x_albers"].max() + 500 - OFF) / RES) * RES + OFF
        y0 = np.floor((g["y_albers"].min() - 500 - OFF) / RES) * RES + OFF
        nx, ny = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
        arr = ee.data.computePixels({
            "expression": mask, "fileFormat": "NUMPY_NDARRAY",
            "grid": {"dimensions": {"width": nx, "height": ny},
                     "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": float(x0),
                                         "shearY": 0, "scaleY": -RES, "translateY": float(y1)},
                     "crsCode": "EPSG:5070"}})
        m = arr["m"].astype(bool)
        layers = {}
        for v in vars_:
            full = np.full((ny, nx), np.nan)
            for it, (tx0, ty0, tx1, ty1) in ((it, tile_bounds(it)) for it in items):
                ix0, iy0, ix1, iy1 = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                with rasterio.open(f"{it['assets'][v]['href']}?{token}") as src:
                    a = src.read(1, window=from_bounds(ix0, iy0, ix1, iy1, src.transform),
                                 masked=True).astype(float).filled(np.nan)
                r0, c0 = int(round((y1 - iy1) / RES)), int(round((ix0 - x0) / RES))
                full[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
            if v == "pwsl1pomu":
                full[full > 100] = np.nan  # 999 = not rated
            layers[v] = full
        xc = x0 + RES / 2 + RES * np.arange(nx)
        yc = y1 - RES / 2 - RES * np.arange(ny)
        for r in g.itertuples():
            cs = (xc >= r.x_albers - 500) & (xc < r.x_albers + 500)
            rs = (yc >= r.y_albers - 500) & (yc < r.y_albers + 500)
            mm = m[np.ix_(rs, cs)]
            rec = {"cell_id": r.cell_id, "citrus_px": int(mm.sum())}
            for v, L in layers.items():
                vals = L[np.ix_(rs, cs)][mm]
                rec[v] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            out.append(rec)
        print(f"block {n + 1}/{len(blocks)}: {len(g)} cells ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(PROC / "soils_cells_citrus.csv", index=False, float_format="%.3f")
    print(df.describe().round(2).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["canopy", "soils", "soils_citrus"])
    args = p.parse_args()
    {"canopy": canopy, "soils": soils, "soils_citrus": soils_citrus}[args.step](args)


if __name__ == "__main__":
    main()
