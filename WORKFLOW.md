# Citrus ET under HLB and hurricanes — Florida, 2000–2025

Paper for the *Land* (MDPI) Special Issue "Monitoring and Modelling the Water and Carbon
Dynamics in the Soil-Plant-Atmosphere Continuum" (section "Land, Soil and Water").
Submission deadline: **2027-03-20**.


## Question

Did grove-scale evapotranspiration (ET) of Florida citrus fall after huanglongbing (HLB, 2005),
how did hurricanes (2004–05, Irma 2017, Ian 2022, Milton 2024) change it, and are the pre-HLB
crop coefficients (Kc) behind UF/IFAS irrigation recommendations still valid?

## Folder layout

| Folder | Content |
| --- | --- |
| `data/raw/` | Downloads as received (FAWN, CDL, FSAID, HURDAT2, NASS). Not edited by hand. |
| `data/processed/` | Tables produced by the scripts. |
| `scripts/` | Numbered analysis scripts, run in order. |
| `figures/` | Figures for the manuscript. |
| `manuscript/` | Drafts, cover letter, response to reviewers. |

## Setup

The virtual environment lives outside OneDrive, at `~/.venvs/citrus-et` (Python 3.13),
so the sync does not copy thousands of package files.

```bash
uv venv ~/.venvs/citrus-et --python python3.13
source ~/.venvs/citrus-et/bin/activate
uv pip install -r requirements.txt
earthengine authenticate --auth_mode=localhost --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform
export EE_PROJECT=<your-earth-engine-project>
```

The default `earthengine authenticate` (gcloud mode) asks for Google Drive access and is
blocked by Google; the command above requests only the Earth Engine scopes.

## Reference ET (decision, 2026-11-16)

Central estimate: FAWN ETo recomputed with FAO-56, scaled to FAWN's published level, with
10-m wind held at each station's monthly climatology (`--eto fawn_windclim`). gridMET and
unadjusted FAWN are sensitivity bounds. Reason: FAWN wind falls ~22%/decade and gridMET
radiation rises ~2%/decade, while ERA5-Land shows neither.

## Scripts

1. `scripts/01_check_openet_florida.py` — checks that OpenET v2.1 monthly ET has valid
   values over citrus pixels in the main citrus counties, 2000–2005 (pre-HLB baseline).
   Writes `data/processed/openet_check_<start>_<end>.csv`.
2. `scripts/02_extract_openet_series.py` — builds 1-km analysis cells of persistent citrus
   (CDL classes 72/212 in >= 85% of CDL years since 2008 and in one of the last two years;
   >= 25 ha per cell), then extracts monthly OpenET ET and gridMET ETo per cell.
   Run `cells` once, then `years --start 2000 --end 2025` (cached per year, resumable).
   Writes `data/processed/citrus_cells_1km.csv` and `data/processed/series/openet_cells_<year>.csv`.
3. `scripts/03_summarize_series.py` — annual ET, ET0 and Kc_act (ET/ET0) per cell and per
   region (Ridge, Indian River, Southwest Flatwoods). ET is built once, independently of the
   ET0 scenario (complete years summed; 9–11 valid months completed by the cell's own ET
   climatology); FAWN ET0 gaps of up to 4 months per cell-year are filled with the cell's
   calendar-month mean; only cell-years with ET0 under all four scenarios are kept (common
   support). Regional Kc is the ratio of area-weighted mean ET to mean ET0, so
   dln Kc = dln ET − dln ET0 holds exactly. Writes `data/processed/annual_cells*.csv` and
   `data/processed/annual_region*.csv`.

   `--eto fawn` or `--eto fawn_windclim` uses FAWN reference ET (script 6) instead of gridMET
   and writes `annual_*_fawn*.csv`.
4. `scripts/04_fig_kc_trend.py` — Figure of annual satellite Kc by region with HLB and
   hurricane years marked, with the central ETo (FAWN, wind held at climatology).
   Writes `figures/fig_kc_trend.png` and `.pdf`; `--eto gridmet` or `--eto fawn` writes
   `fig_kc_trend_gridmet.*` / `fig_kc_trend_fawn.*` for the sensitivity versions.
5. `scripts/05_fawn_eto.py` — `fetch` downloads FAWN daily weather for the 34 citrus-belt
   stations (report generator; cached in `data/raw/fawn/report/`); `build` recomputes daily
   FAO-56 ETo (FAWN publishes ETo only from 2009), scales it to FAWN's published level per
   station and month, and adds a variant with wind held at the station's monthly climatology
   (FAWN wind falls ~22%/decade, ERA5-Land ~3%). Writes `data/processed/fawn_station_monthly.csv`.
6. `scripts/06_fawn_vs_gridmet.py` — `gridmet` (Earth Engine) extracts gridMET ETo at the
   stations; `compare` reports bias and trends; `cells` maps FAWN ETo to the 1-km cells
   (inverse distance, 3 nearest of a fixed station set, after a network-relative QC).
   Add `--var eto_windclim_mm` for the wind-held variant. `drivers` (Earth Engine) compares
   trends of radiation, wind and VPD at the long-record stations: FAWN vs gridMET vs
   ERA5-Land. Writes `data/processed/eto_drivers_trend.csv`.
7. `scripts/07_fig_eto_sensitivity.py` — ETo and Kc under gridMET, FAWN and wind-held FAWN.
   Writes `figures/fig_eto_sensitivity.png` and `.pdf`.

8. `scripts/08_hurricane_events.py` — hurricane event study. `exposure` estimates each
   cell's peak wind from HURDAT2 tracks and 34/50/64-kt radii (2004 on); `events` builds
   monthly Kc anomalies (central ETo) against a 3-year baseline, compares peak-wind
   terciles, fits the change per +10 kt, and repeats it on hurricane-free placebo years.
   Writes `data/processed/hurricane_*.csv`.
9. `scripts/09_fig_hurricane_events.py` — event curves (windier minus least windy cells)
   and the wind gradient against placebo. Writes `figures/fig_hurricane_events.png` and `.pdf`.

10. `scripts/10_kc_vs_ifas.py` — observed monthly Kc by region and period (2000–2005,
    2006–2020, 2021–2025) vs. the UF/IFAS citrus Kc (EDIS AE456 Table 1, from Morgan et
    al. 2006), and the annual gap in consumptive use (mm and hm³) for each ETo source.
    Writes `data/processed/kc_monthly_vs_ifas.csv` and `kc_gap_vs_ifas.csv`.
11. `scripts/11_fig_kc_vs_ifas.py` — monthly Kc curves by period against UF/IFAS, per
    region. Writes `figures/fig_kc_vs_ifas.png` and `.pdf`.

12. `scripts/12_validate_tower.py` — validation against the USGS citrus ET tower near
    Arcadia (site 271042081462301, May 2004 – May 2005). `fetch` downloads NWIS daily values
    to `data/raw/usgs/`; `extract` (Earth Engine) gets the OpenET ensemble and six models
    over orange pixels within 300 m; `compare` writes `data/processed/tower_monthly.csv`
    and `tower_metrics.csv` (bias, RMSE, r, KGE).
13. `scripts/13_fig_tower.py` — monthly ET at the tower and bias by model. Writes
    `figures/fig_tower_validation.png` and `.pdf`.

14. `scripts/14_model_comparison.py` — the ensemble and all six OpenET models side by side,
    ordered by bias at the tower, on the cell-years common to all seven products (2001–2025):
    Kc contrasts (2001–05→2019–25, 2015–19→2021–25, 2020→2023–25) with block-bootstrap CIs,
    gap to UF/IFAS and Ian wind gradient vs. placebo. Writes `data/processed/model_comparison.csv` and
    `figures/fig_model_comparison.png`/`.pdf`.

15. `scripts/15_fig_map.py` — Figure 1, study area: citrus cells by region, FAWN stations
    (the 16 used for ET₀ filled), the USGS tower and the hurricane tracks. Boundaries are
    TIGER 2018 counties and states from Earth Engine, saved in `data/raw/boundaries/`.
    Writes `figures/fig_study_area.png` and `.pdf`.

16. `scripts/16_build_manuscript.py` — assembles the manuscript. Reads
    `manuscript/manuscript_source.md` (the text of the doc tabs, with provisional citation
    numbers), renumbers citations in order of first appearance, and builds
    `manuscript/manuscript.docx` with pandoc (figures from `figures/`, equations as Word
    equations). It uses the official *Land* Word template
    (`manuscript/template/land-template.dot`) when present; otherwise the official
    *Agriculture* template (`manuscript/template/agriculture-template.dot`, 81 KB, no macros)
    with the running head and footer changed to *Land* and the journal logo removed. It then
    gives every paragraph the template's `MDPI_*` style. Edit the source, not the build files, and rerun.

### Revision after internal review (2026-11-16)

17. `scripts/17_extract_extra.py` — Earth Engine: `eto_bc` extracts OpenET's bias-corrected
    gridMET ET₀ (`projects/openet/assets/reference_et/conus/gridmet/monthly/v1`, the reference
    used inside OpenET v2.1) per cell-month; `count` extracts the SSEBop cloud-free observation
    count. Writes `data/processed/series/extra/`. Script 03 and 10 take `--eto gridmet_bc`.
18. `scripts/18_fawn_cv.py` — out-of-sample checks of the FAWN ET₀: leave-one-year-out scaling
    and leave-one-station-out interpolation. Writes `data/processed/fawn_cv*.csv`.
19. `scripts/19_trend_robustness.py` — Sen slope with a residual moving-block bootstrap CI,
    Mann–Kendall with the Hamed–Rao correction, period contrasts (100 × dln of ratios of
    period means) with moving-block, iid and spatial-block bootstrap CIs, a hinge-vs-line
    test whose sup-F bootstrap null repeats the search over break years, and the exact
    decomposition dln Kc = dln ET − dln ET₀, for the four ET₀ scenarios. Writes
    `trend_robustness.csv`.
20. `scripts/20_fig_decomposition.py` — figure of ET, ET₀ and Kc changes by scenario.
21. `scripts/21_observability.py` — observability summary and sensitivity of the contrasts to
    observed-only months and complete years.

22. `scripts/22_canopy_soils.py` — `canopy` (Earth Engine): USFS Tree Canopy Cover v2025-6,
    annual 2000–2025, over each cell's persistent-citrus pixels; `soils`: gNATSGO rasters
    (available water storage 0–100 cm, root-zone storage, droughty share, wetland-soil share)
    read as 1-km windows from Microsoft Planetary Computer; `soils_citrus`: the same plus
    0–30 cm storage over the persistent-citrus pixels only (mask sampled in Earth Engine on the
    10-m gNATSGO grid), `soils_cells_citrus.csv`. Needs rasterio.
23. `scripts/23_structure_soil.py` — two-way fixed-effects panel with and without canopy,
    cell-level change regressions on canopy change and soil water storage, and stable-canopy
    cells by water-storage tercile (spatial block bootstrap).
24. `scripts/24_fig_structure_soil.py` — figure: NAIP canopy index by region, ET change against
    canopy change, and ET change by soil water storage in stable-canopy cells.
25. `scripts/25_naip_canopy.py` — NAIP canopy index (Earth Engine): per cell and winter
    acquisition (2017, 2019, 2023), share of persistent-citrus area above the cell-year NDVI
    Otsu threshold at 2 m. Writes `data/processed/naip_canopy_cells.csv`. The `hist` step
    saves the full NDVI histograms and flight day of year (`naip_hist_cells.csv`). Then run
    `python scripts/23_structure_soil.py naip` for the NAIP-based tests: three ET windows
    (2015–19→2021–25, 2017–19→2023, 2015–19→2021–23), three threshold rules (per cell-year,
    pooled per year, pooled per year and region), stable-canopy limits of 3–10 pp, and soil
    storage 0–100 and 0–30 cm over citrus pixels.
`scripts/figstyle.py` — shared visual system of the figures (typeface, validated colour roles for
    regions and ET₀ scenarios, panel labels, direct labels, 600-dpi PNG and vector PDF output).
26. `scripts/26_canopy_tables.py` — Tables 7 and 8 and the numbers of Section 3.7 from the outputs of
    `23_structure_soil.py naip`, with one rounding rule (half away from zero); writes
    `manuscript/tables_canopy.md`.

Script 08 was redesigned: follow-up stops at the next event, nine placebo Septembers whose
whole window is storm-free, months 0–6/7–12/13–24, spatial block-bootstrap CIs and a
pre-event gradient check (`hurricane_gradient_summary*.csv`). The `pretrend` step adds the
pre-event level, the exposure × time trend and quarterly gradients with spatial-block CIs
(`hurricane_pretrend*.csv`).

Single OpenET models: scripts 02, 03, 08 and 10 take `--model` (ensemble, disalexi,
eemetric, geesebal, ptjpl, sims, ssebop). Series go to `data/processed/series/<model>/` and
outputs get a `_<model>` suffix:

```bash
for m in disalexi eemetric geesebal ptjpl sims ssebop; do
  python scripts/02_extract_openet_series.py years --start 2000 --end 2025 --model $m
  python scripts/03_summarize_series.py --eto fawn_windclim --model $m
  python scripts/10_kc_vs_ifas.py --model $m
  python scripts/08_hurricane_events.py events --model $m
done
python scripts/14_model_comparison.py
```

Full run of the ETo comparison:

```bash
python scripts/05_fawn_eto.py fetch && python scripts/05_fawn_eto.py build
python scripts/06_fawn_vs_gridmet.py gridmet && python scripts/06_fawn_vs_gridmet.py compare
python scripts/06_fawn_vs_gridmet.py cells && python scripts/06_fawn_vs_gridmet.py cells --var eto_windclim_mm
python scripts/06_fawn_vs_gridmet.py drivers
for e in gridmet gridmet_bc fawn fawn_windclim; do python scripts/03_summarize_series.py --eto $e; done
for e in fawn_windclim gridmet fawn; do python scripts/04_fig_kc_trend.py --eto $e; done
python scripts/07_fig_eto_sensitivity.py
```

Analysis unit, first version: 1-km cells, not individual groves. Replace with FDACS FSAID
field polygons once downloaded.

## Data sources

- OpenET Ensemble CONUS monthly **v2.1** (`projects/openet/assets/ensemble/conus/gridmet/monthly/v2_1`)
  — monthly ET since Oct 1999, 30 m. Only v2.1 covers Florida; v2.0 is western states only.
- USDA NASS Cropland Data Layer — citrus mask (classes 72 "Citrus" and 212 "Oranges").
- FAWN — daily weather from the report generator (fawn.ifas.ufl.edu/data/reports) and the
  yearly archive (fawn.ifas.ufl.edu/data/fawnpub/daily_summaries/BY_YEAR, incomplete for the
  southern stations in 2008–2017); station coordinates from the FAWN map feed.
- gridMET (`IDAHO_EPSCOR/GRIDMET`) — reference ET used by OpenET; ~37% above FAWN in Florida.
- ERA5-Land (`ECMWF/ERA5_LAND/DAILY_AGGR`) — independent check of radiation and wind trends.
- FDACS FSAID — irrigated field polygons.
- NOAA HURDAT2 (nhc.noaa.gov/data/hurdat, file `hurdat2-1851-2025-092326.txt` in
  `data/raw/hurdat/`) — tracks and wind radii for hurricane exposure.
- USGS NWIS site 271042081462301 (2X4 Citrus Climate Station near Arcadia) — daily ET,
  the only public citrus ET tower in Florida (Jia et al. 2007); used for validation.
