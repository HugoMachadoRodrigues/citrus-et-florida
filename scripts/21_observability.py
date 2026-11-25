"""Observability of the monthly ET record and its effect on the results (review).

Uses the SSEBop "count" band (cloud-free Landsat observations behind each monthly value,
script 17) averaged over each cell's persistent-citrus pixels.

1. Summary: mean count and share of cell-months with no clear observation (count < 0.5),
   by year, by month of the year, and in hurricane months.
2. Sensitivity: annual ET and Kc (FAWN wind-standardized ET0, construction of script 03)
   recomputed (a) only from months with at least one clear observation, and (b) only from
   complete years (12 valid ET months and no gap-filled ET0 month), and the period contrasts
   of script 19 repeated on these series.

Writes data/processed/observability_summary.csv, observability_sensitivity.csv
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EVENT_MONTHS = ["2004-08", "2004-09", "2005-10", "2017-09", "2022-09", "2024-09", "2024-10"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    s03 = load("s03", ROOT / "scripts" / "03_summarize_series.py")
    s19 = load("s19", ROOT / "scripts" / "19_trend_robustness.py")
    cnt = pd.concat([pd.read_csv(f) for f in sorted(
        (PROC / "series" / "extra").glob("count_cells_*.csv"))], ignore_index=True)
    cnt = cnt.rename(columns={"value": "clear_obs"})
    m = pd.concat([pd.read_csv(f) for f in sorted((PROC / "series").glob("openet_cells_*.csv"))],
                  ignore_index=True)
    eto = pd.read_csv(PROC / "series" / "fawn_eto_cells_windclim.csv")
    m = (m.drop(columns="eto_mm").merge(eto[["cell_id", "year", "month", "eto_fawn_mm"]],
                                        on=["cell_id", "year", "month"], how="left")
         .rename(columns={"eto_fawn_mm": "eto_mm"})
         .merge(cnt, on=["cell_id", "year", "month"], how="left"))
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    cells["region"] = cells["county"].map(s03.region_of)
    m = m.merge(cells[["cell_id", "region", "citrus_ha"]], on="cell_id")
    max_px = m.groupby("cell_id")["et_px"].transform("max")
    m["valid"] = (m["et_px"] >= 0.5 * max_px) & m["et_mm"].notna() & m["eto_mm"].notna()
    m["observed"] = m["clear_obs"] >= 0.5

    # 1. Summary.
    m["ym"] = m["year"].astype(str) + "-" + m["month"].map("{:02d}".format)
    rows = []
    for key, g in [("year", m.groupby("year")), ("month", m.groupby("month"))]:
        s = g.agg(mean_clear_obs=("clear_obs", "mean"),
                  share_unobserved=("observed", lambda x: 1 - x.mean())).reset_index()
        s.insert(0, "by", key)
        s = s.rename(columns={key: "value"})
        rows.append(s)
    ev = m[m["ym"].isin(EVENT_MONTHS)].groupby("ym").agg(
        mean_clear_obs=("clear_obs", "mean"),
        share_unobserved=("observed", lambda x: 1 - x.mean())).reset_index()
    ev.insert(0, "by", "event month")
    rows.append(ev.rename(columns={"ym": "value"}))
    summ = pd.concat(rows, ignore_index=True)
    summ.to_csv(PROC / "observability_summary.csv", index=False, float_format="%.3f")

    # 2. Sensitivity of the contrasts (same construction as script 03).
    base = pd.read_csv(PROC / "annual_cells_fawn_windclim.csv")
    ok, filled = s03.common_support()
    eto_y = s03.eto_annual("fawn_windclim")["eto_year"]

    def from_et(et):
        a = et.join(eto_y, how="inner").join(ok.rename("support")).join(filled.rename("eto_filled"))
        a = a[a["support"].fillna(False).astype(bool)]
        a = a.reset_index().merge(cells[["cell_id", "region", "citrus_ha"]], on="cell_id")
        return a

    mo = m[["cell_id", "year", "month", "et_mm", "et_px"]].copy()
    mo.loc[~m["observed"].values, "et_mm"] = np.nan
    variants = {
        "all valid months (as in the paper)": base,
        "observed months only (count >= 1)": from_et(s03.annual_et(mo)[0]),
        "complete years only (12 valid ET months, no filled ET0)":
            base[base["et_complete"] & (base["eto_filled"] == 0)],
    }
    res = []
    for name, a in variants.items():
        reg = s03.aggregate(a)
        for region, g in reg.groupby("region"):
            g = g.sort_values("year")
            for cname, pa, pb in s19.CONTRASTS:
                c = s19.contrast(g, pa, pb)
                res.append({"variant": name, "region": region, "contrast": cname,
                            "dln_kc": c["dln_kc"], "ci_lo": c.get("kc_lo"), "ci_hi": c.get("kc_hi"),
                            "dln_et": c["dln_et"], "et_lo": c.get("et_lo"), "et_hi": c.get("et_hi"),
                            "median_cells_per_year": g["cells"].median()})
    res = pd.DataFrame(res)
    res.to_csv(PROC / "observability_sensitivity.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print(summ[summ["by"] != "year"].round(2).to_string(index=False))
    print(summ[summ["by"] == "year"][["value", "mean_clear_obs", "share_unobserved"]]
          .round(2).T.to_string())
    print()
    print(res[res["contrast"].isin(["2000-05 vs 2019-25", "2015-19 vs 2021-25 (no 2020)"])]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
