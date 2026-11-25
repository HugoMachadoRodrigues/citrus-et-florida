"""Observed monthly satellite Kc vs. the UF/IFAS citrus Kc, and what the gap means in water.

UF/IFAS Kc for mature citrus (EDIS AE456, Migliaccio et al., Table 1; values from Morgan
et al. 2006, measured before HLB) are used with FAWN reference ET in the UF/IFAS
scheduling tools. Observed Kc = OpenET ET / ETo for the persistent-citrus cells, by region,
calendar month and period. The gap is turned into consumptive use per year:
    dET = sum over months (Kc_IFAS - Kc_obs) * ETo
i.e. how much more water the UF/IFAS coefficients assume the groves use than the satellite
sees. This is consumptive use, not applied irrigation (rain, soil storage and system
efficiency sit in between).

Reads  data/processed/citrus_cells_1km.csv, series/openet_cells_*.csv,
       series/fawn_eto_cells[_windclim].csv
Writes data/processed/kc_monthly_vs_ifas.csv, kc_gap_vs_ifas.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# EDIS AE456 Table 1, citrus column (Morgan et al. 2006), January..December.
KC_IFAS = [0.79, 0.86, 0.93, 0.97, 1.03, 1.05, 1.05, 1.03, 1.00, 0.95, 0.87, 0.79]
PERIODS = {"2000-2005 (pre-HLB)": (2000, 2005), "2006-2020": (2006, 2020),
           "2021-2025": (2021, 2025)}
REGIONS = {
    "Ridge": ["Polk", "Highlands", "Lake", "Osceola", "Orange"],
    "Indian River": ["Indian River", "St. Lucie", "Martin", "Okeechobee", "Brevard"],
}
ETO_FILES = {"fawn_windclim": "fawn_eto_cells_windclim.csv", "fawn": "fawn_eto_cells.csv",
             "gridmet_bc": "extra/eto_bc_cells_*.csv", "gridmet": None}


def region_of(county):
    return next((r for r, cs in REGIONS.items() if county in cs), "Southwest Flatwoods")


def cell_months(eto, model="ensemble"):
    sdir = PROC / "series" if model == "ensemble" else PROC / "series" / model
    files = sorted(sdir.glob("openet_cells_*.csv"))
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if eto == "gridmet_bc":
        e = pd.concat([pd.read_csv(f) for f in sorted((PROC / "series").glob(ETO_FILES[eto]))])
        m = (m.drop(columns="eto_mm").merge(e.rename(columns={"value": "eto_mm"}),
                                            on=["cell_id", "year", "month"], how="left"))
    elif ETO_FILES[eto]:
        e = pd.read_csv(PROC / "series" / ETO_FILES[eto])
        m = (m.drop(columns="eto_mm").merge(e[["cell_id", "year", "month", "eto_fawn_mm"]],
                                            on=["cell_id", "year", "month"], how="left")
             .rename(columns={"eto_fawn_mm": "eto_mm"}))
    max_px = m.groupby("cell_id")["et_px"].transform("max")
    m["valid"] = (m["et_px"] >= 0.5 * max_px) & m["et_mm"].notna() & (m["eto_mm"] > 0)
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    cells["region"] = cells["county"].map(region_of)
    return m.merge(cells[["cell_id", "region", "citrus_ha"]], on="cell_id"), cells


def summarize(eto, model="ensemble"):
    m, cells = cell_months(eto, model)
    v = m[m["valid"]]
    rows = []
    for region, g in pd.concat([v, v.assign(region="All")]).groupby("region"):
        # Ratio of area-weighted sums per year-month, then the mean over years.
        ym = g.assign(etw=g["et_mm"] * g["citrus_ha"], etow=g["eto_mm"] * g["citrus_ha"]) \
              .groupby(["year", "month"])[["etw", "etow", "citrus_ha"]].sum()
        ym["kc"] = ym["etw"] / ym["etow"]
        ym["eto"] = ym["etow"] / ym["citrus_ha"]
        ym = ym.reset_index()
        for pname, (y0, y1) in PERIODS.items():
            p = ym[ym["year"].between(y0, y1)].groupby("month").agg(
                kc=("kc", "mean"), kc_sd=("kc", "std"), eto=("eto", "mean"))
            for mo, r in p.iterrows():
                rows.append({"eto_source": eto, "region": region, "period": pname,
                             "month": mo, "kc_obs": r["kc"], "kc_sd": r["kc_sd"],
                             "eto_mm": r["eto"], "kc_ifas": KC_IFAS[mo - 1]})
    area = cells.groupby("region")["citrus_ha"].sum()
    area["All"] = cells["citrus_ha"].sum()
    return pd.DataFrame(rows), area


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"], default="ensemble",
                   help="OpenET ensemble or a single model (series from script 02 --model)")
    args = p.parse_args()
    sfx = "" if args.model == "ensemble" else f"_{args.model}"
    monthly, gaps = [], []
    for eto in ETO_FILES:
        mon, area = summarize(eto, args.model)
        monthly.append(mon)
        for (region, period), g in mon.groupby(["region", "period"]):
            et_ifas = (g["kc_ifas"] * g["eto_mm"]).sum()
            et_obs = (g["kc_obs"] * g["eto_mm"]).sum()
            gaps.append({"eto_source": eto, "region": region, "period": period,
                         "kc_obs_mean": np.average(g["kc_obs"], weights=g["eto_mm"]),
                         "kc_ifas_mean": np.average(g["kc_ifas"], weights=g["eto_mm"]),
                         "eto_mm_yr": g["eto_mm"].sum(), "etc_ifas_mm": et_ifas,
                         "et_obs_mm": et_obs, "gap_mm": et_ifas - et_obs,
                         "gap_pct": 100 * (et_ifas / et_obs - 1),
                         "citrus_ha": area[region],
                         "gap_hm3": (et_ifas - et_obs) * area[region] * 10 / 1e6})
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.to_csv(PROC / f"kc_monthly_vs_ifas{sfx}.csv", index=False, float_format="%.4f")
    gaps = pd.DataFrame(gaps)
    gaps.to_csv(PROC / f"kc_gap_vs_ifas{sfx}.csv", index=False, float_format="%.3f")

    pd.set_option("display.width", 220)
    c = monthly[monthly["eto_source"] == "fawn_windclim"]
    print("Monthly Kc, central ETo (FAWN, wind held), All cells, vs UF/IFAS:")
    print(c[c["region"] == "All"].pivot(index="month", columns="period", values="kc_obs")
          .assign(UF_IFAS=KC_IFAS).round(3).to_string())
    print("\nAnnual gap, UF/IFAS-based ETc minus observed ET (consumptive use):")
    print(gaps.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
