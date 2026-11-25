# Data and code for: Evapotranspiration of Persistent Florida Citrus Groves during the Huanglongbing Era: Sensitivity to Reference Evapotranspiration and Links to Canopy and Soil Water Storage

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24534652.svg)](https://doi.org/10.5281/zenodo.24534652)

Hugo Rodrigues, Range Cattle Research and Education Center, Institute of Food and Agricultural
Sciences, University of Florida, Ona, FL, USA.

This archive holds the scripts, derived data and figures behind every number, table and figure
of the article submitted to *Land* (MDPI), Special Issue "Monitoring and Modelling the Water
and Carbon Dynamics in the Soil-Plant-Atmosphere Continuum". Version 1.0 corresponds to the
submitted manuscript and is archived in Zenodo as
https://doi.org/10.5281/zenodo.24534652 (all versions: https://doi.org/10.5281/zenodo.24534651).

## Contents

| Folder | Content |
| --- | --- |
| `scripts/` | Python scripts 01–26 (workflow, figures and the manuscript build). `WORKFLOW.md` describes each script and the order of execution. |
| `data/raw/` | Inputs that are not served by Google Earth Engine: FAWN daily reports and station list (`fawn/`), HURDAT2 best tracks (`hurdat/`), the USGS NWIS daily ET record of site 271042081462301 (`usgs/`), and county and state boundaries from Census TIGER 2018 (`boundaries/`). |
| `data/processed/` | Derived tables (CSV). Monthly cell series are in `data/processed/series/`. |
| `figures/` | Figures 1–9 of the article (PNG and PDF) and supplementary figures. |

Earth Engine inputs (OpenET v2.1, gridMET, ERA5-Land, USDA CDL, NAIP, USFS TCC) are not
redistributed; their identifiers are in Table 1 of the article and in the scripts. gNATSGO
rasters are read from the Microsoft Planetary Computer (`scripts/22_canopy_soils.py`).

## Main derived tables

| File | Content |
| --- | --- |
| `citrus_cells_1km.csv` | The 1698 one-kilometre analysis cells: id, Albers centre (EPSG:5070), county, persistent-citrus share and area (ha). |
| `series/openet_cells_<year>.csv`, `series/<model>/` | Monthly OpenET ET (ensemble and six models), gridMET ET₀ and valid-pixel count per cell. |
| `series/fawn_eto_cells[_windclim].csv` | Monthly FAWN ET₀ per cell (measured wind / wind-standardized). |
| `series/extra/` | OpenET bias-corrected gridMET ET₀ and the SSEBop clear-observation count per cell-month. |
| `fawn_stations.csv`, `fawn_daily.csv`, `fawn_station_monthly.csv` | FAWN station list, daily FAO-56 inputs and monthly station ET₀ (raw, scaled, wind-standardized). |
| `annual_cells_<eto>[_<model>].csv`, `annual_region_<eto>[_<model>].csv` | Annual ET, ET₀ and K~c,act~ per cell and per region (common cell-year support across the four ET₀ scenarios). No suffix = raw gridMET; `_gridmet_bc`, `_fawn`, `_fawn_windclim`. |
| `trend_robustness.csv` | Trends, period contrasts (100 × Δln) with three bootstrap intervals, decomposition and sup-F break test. |
| `model_comparison.csv` | Results per OpenET product on the cell-years common to all seven. |
| `hurricane_*.csv` | Exposure per cell and event, event curves, wind gradients, placebo slopes and pre-event checks. |
| `kc_*_vs_ifas*.csv` | Monthly observed coefficients and the difference from the historical UF/IFAS curve. |
| `tower_*.csv` | Validation at the Arcadia citrus ET tower. |
| `naip_*.csv`, `soils_cells_citrus.csv`, `structure_soil_*.csv` | NAIP canopy index, histograms and flight dates; gNATSGO soil water storage over citrus pixels; canopy and soil analyses. |

## Where each result comes from

| Article item | Script | Output |
| --- | --- | --- |
| Figure 1 | `15_fig_map.py` | `figures/fig_study_area` |
| Section 3.1, Figure 2 | `05`, `06`, `18`, `07_fig_eto_sensitivity.py` | `eto_station_*`, `eto_drivers_*`, `fawn_cv*`, `figures/fig_eto_sensitivity` |
| Table 3, Figure 3 | `12_validate_tower.py`, `13_fig_tower.py` | `tower_metrics.csv`, `figures/fig_tower_validation` |
| Section 3.3, Table 4, Figure 4 | `03_summarize_series.py`, `19_trend_robustness.py`, `20_fig_decomposition.py` | `annual_region_*`, `trend_robustness.csv`, `figures/fig_decomposition` |
| Figure 5 (and other ET₀ scenarios) | `04_fig_kc_trend.py` | `figures/fig_kc_trend*` |
| Section 3.4 | `17_extract_extra.py`, `21_observability.py` | `observability_*.csv` |
| Table 5, Figure 6 | `14_model_comparison.py` | `model_comparison.csv`, `figures/fig_model_comparison` |
| Table 6, Figure 7 | `08_hurricane_events.py` (`exposure`, `events`, `pretrend`), `09_fig_hurricane_events.py` | `hurricane_gradient_summary*.csv`, `hurricane_pretrend*.csv`, `figures/fig_hurricane_events` |
| Tables 7 and 8, Figure 8 | `22_canopy_soils.py`, `25_naip_canopy.py`, `23_structure_soil.py naip`, `24_fig_structure_soil.py`, `26_canopy_tables.py` | `structure_soil_*_naip.csv`, `figures/fig_structure_soil` |
| Table 9, Figure 9 | `10_kc_vs_ifas.py`, `11_fig_kc_vs_ifas.py` | `kc_gap_vs_ifas*.csv`, `figures/fig_kc_vs_ifas` |

## Reproducing

Python 3.13 with the packages in `requirements.txt`. Scripts that read Earth Engine need an
authenticated account and the environment variable `EE_PROJECT` set to your own Earth Engine
project. Every analysis step after the extraction (scripts 03, 04, 07, 09–11, 13, 14, 19–21,
23 and 24) runs from the CSV files in this archive without Earth Engine. Bootstrap intervals
use fixed random seeds.

## Licences

Code (`scripts/`): MIT License (`LICENSE-CODE`). Derived data and figures: Creative Commons
Attribution 4.0 International (`LICENSE-DATA`). Third-party inputs in `data/raw/` keep the terms
of their providers (FAWN, University of Florida; NOAA National Hurricane Center; U.S. Geological
Survey; U.S. Census Bureau); the U.S. federal data are in the public domain.

## Citation

Please cite the article and this archive (see `CITATION.cff`).
