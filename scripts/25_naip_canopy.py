"""Grove canopy index from NAIP aerial imagery (independent of Landsat), Tier C check.

For each cell and NAIP year, the NDVI histogram of the persistent-citrus pixels at 2 m is
computed (USDA/NAIP/DOQQ, bands N and R), and an Otsu threshold separates the brighter-NDVI
class (tree crowns, plus any green row middles) from the rest. The canopy index is the share
of pixels above that cell-year threshold. A per-image threshold avoids the large radiometric
differences between NAIP acquisitions. Only winter-season acquisitions are used, when grass
in the row middles is mostly dormant: 2017 (Oct-Dec), 2019 (Nov-Dec) and 2023 (Jan-Mar).

Step:   python scripts/25_naip_canopy.py        # canopy index per cell-year
        python scripts/25_naip_canopy.py hist   # full histograms and flight day of year
Writes: data/processed/naip_canopy_cells.csv (cell_id, year, otsu_ndvi, canopy_frac, px)
"""

import importlib.util
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
YEARS = [2017, 2019, 2023]
LO, HI, NBINS = -0.4, 0.8, 60
CHUNK = 150


def load_s02():
    spec = importlib.util.spec_from_file_location("s02", ROOT / "scripts" / "02_extract_openet_series.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def main():
    s02 = load_s02()
    s02.init_ee()
    cells = pd.read_csv(s02.CELLS_CSV)
    mask = s02.persistent_citrus(0.85)
    naip = ee.ImageCollection("USDA/NAIP/DOQQ")
    edges = np.linspace(LO, HI, NBINS + 1)
    out_path = PROC / "naip_canopy_cells.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["cell_id", "year"])
    rows = [done]
    for year in YEARS:
        nd = (naip.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
              .normalizedDifference(["N", "R"]).rename("nd").updateMask(mask))
        todo = cells[~cells["cell_id"].isin(done.loc[done["year"] == year, "cell_id"])]
        for i in range(0, len(todo), CHUNK):
            t = time.time()
            fc = s02.cells_fc(todo.iloc[i:i + CHUNK])
            st = nd.reduceRegions(collection=fc, reducer=ee.Reducer.fixedHistogram(LO, HI, NBINS),
                                  crs=s02.ALBERS, scale=2, tileScale=16)
            feats = st.map(lambda f: ee.Feature(None, {"cell_id": f.get("cell_id"),
                                                       "hist": f.get("histogram")})).getInfo()
            recs = []
            for f in feats["features"]:
                p = f["properties"]
                h = p.get("hist")
                if not h:
                    continue
                counts = np.array([b[1] for b in h])
                thr = otsu(counts, edges)
                frac = counts[edges[:-1] >= thr].sum() / counts.sum() if counts.sum() else np.nan
                recs.append({"cell_id": int(p["cell_id"]), "year": year, "otsu_ndvi": thr,
                             "canopy_frac": frac, "px": int(counts.sum())})
            rows.append(pd.DataFrame(recs))
            pd.concat(rows, ignore_index=True).to_csv(out_path, index=False, float_format="%.4f")
            print(f"{year} cells {i}-{i + len(fc.getInfo()['features'])}: {len(recs)} done "
                  f"({time.time() - t:.0f} s)", flush=True)
    d = pd.read_csv(out_path)
    print(d.groupby("year")[["otsu_ndvi", "canopy_frac"]].describe().round(3).T.to_string())


def hist():
    """Save the full NDVI histogram and the acquisition day of year of each cell-year, so that
    alternative thresholds (pooled by year or by year and region) and the flight dates can be
    examined offline (second internal review). Writes data/processed/naip_hist_cells.csv."""
    s02 = load_s02()
    s02.init_ee()
    cells = pd.read_csv(s02.CELLS_CSV)
    mask = s02.persistent_citrus(0.85)
    naip = ee.ImageCollection("USDA/NAIP/DOQQ")
    out_path = PROC / "naip_hist_cells.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["cell_id", "year"])
    rows = [done]
    for year in YEARS:
        col = (naip.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
               .filterBounds(ee.Geometry.Rectangle(s02.CITRUS_BELT_BOX)))
        nd = col.mosaic().normalizedDifference(["N", "R"]).rename("nd").updateMask(mask)
        # Day of year carried on each image's own footprint (a bare constant has none).
        doy = (col.map(lambda im: im.select("N").multiply(0)
                       .add(im.date().getRelative("day", "year").add(1)).toInt16().rename("doy"))
               .mosaic().updateMask(mask))
        todo = cells[~cells["cell_id"].isin(done.loc[done["year"] == year, "cell_id"])]
        for i in range(0, len(todo), CHUNK):
            t = time.time()
            fc = s02.cells_fc(todo.iloc[i:i + CHUNK])
            st = nd.reduceRegions(collection=fc, reducer=ee.Reducer.fixedHistogram(LO, HI, NBINS),
                                  crs=s02.ALBERS, scale=2, tileScale=16)
            st = doy.reduceRegions(collection=st, reducer=ee.Reducer.mode().setOutputs(["doy"]),
                                   crs=s02.ALBERS, scale=30, tileScale=4)
            feats = st.map(lambda f: ee.Feature(None, {"cell_id": f.get("cell_id"),
                                                       "hist": f.get("histogram"),
                                                       "doy": f.get("doy")})).getInfo()
            recs = []
            for f in feats["features"]:
                p = f["properties"]
                h = p.get("hist")
                if not h:
                    continue
                recs.append({"cell_id": int(p["cell_id"]), "year": year, "doy": p.get("doy"),
                             "counts": ";".join(str(int(b[1])) for b in h)})
            rows.append(pd.DataFrame(recs))
            pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
            print(f"{year} cells {i}: {len(recs)} done ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    import sys
    hist() if "hist" in sys.argv[1:] else main()
