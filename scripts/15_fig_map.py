"""Figure 1: study area. Persistent-citrus cells by region, FAWN stations, the USGS citrus
ET tower and the hurricane tracks of the events analysed.

Reads  data/raw/boundaries/*.geojson (TIGER 2018 via Earth Engine), citrus_cells_1km.csv,
       fawn_stations.csv, fawn_station_monthly.csv, HURDAT2 (data/raw/hurdat)
Writes figures/fig_study_area.png and .pdf
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BND = ROOT / "data" / "raw" / "boundaries"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

INK, INK_2, GRID = fs.INK, fs.INK_2, fs.GRID
LAND, WATER, COUNTY = "#fbfaf7", "#e8eff4", "#d9d7d0"
TRACK = "#6f6d67"
REGION_COLORS = fs.REGION
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
EXTENT = (-83.1, -79.85, 25.6, 29.35)  # lon min, lon max, lat min, lat max
TOWER = (-81.77285919, 27.17866155)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def polygons(g):
    """Outer rings of a GeoJSON geometry; simplify() can return GeometryCollections."""
    if g["type"] == "Polygon":
        return [g["coordinates"][0]]
    if g["type"] == "MultiPolygon":
        return [p[0] for p in g["coordinates"]]
    if g["type"] == "GeometryCollection":
        return [r for h in g["geometries"] for r in polygons(h)]
    return []


def draw_geojson(ax, path, **kw):
    for f in json.loads(path.read_text())["features"]:
        for ring in polygons(f["geometry"]):
            ring = np.asarray(ring)
            ax.fill(ring[:, 0], ring[:, 1], **kw)


def fixed_stations():
    s06 = load_module("s06", ROOT / "scripts" / "06_fawn_vs_gridmet.py")
    with contextlib.redirect_stdout(io.StringIO()):
        fawn = s06.fill_from_neighbour(s06.qc_relative(s06.load_fawn("eto_windclim_mm")))
    share = fawn.groupby("station")["eto_mm"].count() / ((s06.END - s06.START + 1) * 12)
    return set(share[share >= 0.8].index)


def main():
    s03 = load_module("s03", ROOT / "scripts" / "03_summarize_series.py")
    s08 = load_module("s08", ROOT / "scripts" / "08_hurricane_events.py")
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    cells["region"] = cells["county"].map(s03.region_of)
    st = pd.read_csv(PROC / "fawn_stations.csv")
    used = fixed_stations()

    fs.setup()
    fig, ax = plt.subplots(figsize=(5.4, 6.3))
    ax.grid(False)
    ax.set_facecolor(WATER)
    draw_geojson(ax, BND / "states_se_tiger2018.geojson", facecolor=LAND, edgecolor=INK_2,
                 lw=0.6, zorder=1)
    draw_geojson(ax, BND / "fl_counties_tiger2018.geojson", facecolor="none",
                 edgecolor=COUNTY, lw=0.35, zorder=2)

    # 1-km cells as squares (half-width in degrees at each latitude).
    dlat = 0.5 / 111.0
    patches, colors = [], []
    for r in cells.itertuples():
        dlon = dlat / np.cos(np.radians(r.lat))
        patches.append(Rectangle((r.lon - dlon, r.lat - dlat), 2 * dlon, 2 * dlat))
        colors.append(REGION_COLORS[r.region])
    ax.add_collection(PatchCollection(patches, facecolors=colors, edgecolors="none", zorder=3))

    # Hurricane tracks.
    ids = {sid: name for name, ev in s08.EVENTS.items() for sid in ev["storms"]}
    tracks = s08.read_hurdat(s08.HURDAT, set(ids))
    names = {"AL032004": "Charley 2004", "AL062004": "Frances 2004", "AL112004": "Jeanne 2004",
             "AL252005": "Wilma 2005", "AL112017": "Irma 2017", "AL092022": "Ian 2022",
             "AL092024": "Helene 2024", "AL142024": "Milton 2024"}
    # Latitude where each name is written (chosen to keep labels apart).
    label_at = {"AL032004": 26.75, "AL062004": 27.6, "AL112004": 27.05, "AL252005": 26.05,
                "AL112017": 25.85, "AL092022": 28.55, "AL092024": 29.15, "AL142024": 28.05}
    for sid, tr in tracks.items():
        ax.plot(tr["lon"], tr["lat"], color=TRACK, lw=0.8, ls=(0, (4, 2)), alpha=0.8, zorder=4)
        # Label where the track crosses a chosen latitude inside the map.
        seg = s08.interpolate(tr, "30min")
        seg = seg[seg["lon"].between(EXTENT[0] + 0.1, EXTENT[1] - 0.6)
                  & seg["lat"].between(EXTENT[2] + 0.1, EXTENT[3] - 0.1)]
        if seg.empty:
            continue
        i = (seg["lat"] - label_at[sid]).abs().idxmin()
        ax.text(seg.loc[i, "lon"] + 0.05, seg.loc[i, "lat"], names[sid], fontsize=6,
                color=INK_2, va="center", zorder=7, clip_on=True, style="italic",
                path_effects=HALO)

    # FAWN stations and the tower.
    in_set = st["station"].isin(used)
    ax.scatter(st.loc[in_set, "lon"], st.loc[in_set, "lat"], marker="^", s=26, color=INK,
               edgecolors="white", lw=0.5, zorder=6)
    ax.scatter(st.loc[~in_set, "lon"], st.loc[~in_set, "lat"], marker="^", s=22,
               facecolors="white", edgecolors=INK, lw=0.8, zorder=6)
    ax.scatter(*TOWER, marker="*", s=120, color=fs.ACCENT, edgecolors="white", lw=0.6, zorder=8)
    ax.annotate("USGS ET tower", TOWER, xytext=(-14, -20), textcoords="offset points",
                fontsize=6, color=INK, ha="right", zorder=9, path_effects=HALO)

    # Region names next to their groves.
    for r, (lon, lat, ha) in {"Ridge": (-81.22, 27.17, "left"),
                              "Southwest Flatwoods": (-81.62, 26.25, "left"),
                              "Indian River": (-80.55, 27.95, "center")}.items():
        ax.text(lon, lat, r if r != "Southwest Flatwoods" else "Southwest\nFlatwoods",
                fontsize=7.5, weight="bold", color=INK, ha=ha, va="center", zorder=9,
                path_effects=HALO)

    # Citrus area by region (inset bar chart in the Atlantic corner).
    area = cells.groupby("region")["citrus_ha"].sum().reindex(
        ["Southwest Flatwoods", "Ridge", "Indian River"])
    bx = ax.inset_axes([0.70, 0.79, 0.27, 0.14])
    bx.barh(range(3), area.values / 1000, color=[REGION_COLORS[r] for r in area.index],
            height=0.62)
    for i, (r, v) in enumerate(area.items()):
        bx.text(v / 1000 + 1, i, f"{v / 1000:.1f}", va="center", fontsize=5.8, color=INK)
    bx.set_yticks(range(3))
    bx.set_yticklabels(["SW Flatw.", "Ridge", "Indian R."], fontsize=5.8)
    bx.invert_yaxis()
    bx.set_xlim(0, 62)
    bx.set_xticks([])
    bx.tick_params(axis="y", length=0, pad=1.5)
    bx.set_facecolor("none")
    bx.grid(False)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_title(f"Persistent citrus, 10$^3$ ha\n(total {area.sum() / 1000:.1f})", fontsize=6,
                 color=INK, loc="left", pad=2)

    # North arrow.
    ax.annotate("N", xy=(EXTENT[1] - 0.22, 26.55), xytext=(EXTENT[1] - 0.22, 26.3),
                ha="center", fontsize=7, color=INK, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8), zorder=9)

    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1 / np.cos(np.radians(27.4)))
    ax.set_xticks(np.arange(-83, -79.9, 1.0))
    ax.set_yticks(np.arange(26, 29.5, 1.0))
    ax.set_xticklabels([f"{abs(x):.0f}°W" for x in ax.get_xticks()])
    ax.set_yticklabels([f"{y:.0f}°N" for y in ax.get_yticks()])
    ax.tick_params(labelsize=7, length=2)

    # Scale bar (50 km) in the Atlantic corner.
    km = 50 / (111.32 * np.cos(np.radians(25.8)))
    x0, y0 = EXTENT[1] - 0.15 - km, 25.78
    ax.plot([x0, x0 + km], [y0, y0], color=INK, lw=2, solid_capstyle="butt", zorder=9)
    ax.text(x0 + km / 2, y0 + 0.07, "50 km", ha="center", fontsize=6.5, color=INK, zorder=9)

    handles = [Patch(color=c, label=f"{r}") for r, c in REGION_COLORS.items()] + [
        Line2D([], [], marker="^", ls="none", color=INK, markersize=5,
               label=f"FAWN station used for ET₀ ({len(used)})"),
        Line2D([], [], marker="^", ls="none", markerfacecolor="white", markeredgecolor=INK,
               markersize=5, label="Other FAWN station"),
        Line2D([], [], marker="*", ls="none", color=fs.ACCENT, markersize=9,
               label="USGS citrus ET tower"),
        Line2D([], [], color=TRACK, lw=1.0, ls=(0, (5, 2)), label="Hurricane track"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.6,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handletextpad=0.5,
               title="1-km cells with ≥ 25% persistent citrus, by region", title_fontsize=6.6)
    fig.subplots_adjust(bottom=0.14)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(fs.AXIS)

    # Locator inset: Florida in the southeastern United States.
    # Lower-left corner of the map (Gulf of Mexico) in axes coordinates.
    ins = ax.inset_axes([0.01, 0.01, 0.27, 0.27])
    ins.set_facecolor(WATER)
    draw_geojson(ins, BND / "states_se_tiger2018.geojson", facecolor=LAND, edgecolor=INK_2, lw=0.4)
    ins.add_patch(Rectangle((EXTENT[0], EXTENT[2]), EXTENT[1] - EXTENT[0],
                            EXTENT[3] - EXTENT[2], fill=False, edgecolor=fs.ACCENT, lw=0.8))
    ins.set_xlim(-88.6, -79.6)
    ins.set_ylim(24.3, 35.2)
    ins.set_aspect(1 / np.cos(np.radians(30)))
    ins.set_xticks([])
    ins.set_yticks([])
    for side in ins.spines.values():
        side.set_visible(True)
        side.set_edgecolor(INK_2)
        side.set_linewidth(0.5)

    fs.save(fig, "fig_study_area")


if __name__ == "__main__":
    main()
