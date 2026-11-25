"""Out-of-sample checks of the FAWN reference ET (requested in review).

1. Leave-one-year-out scaling: the monthly factor (FAWN published / daily FAO-56) of each
   station and calendar month is fitted on all overlap years except the test year and
   applied to that year; the scaled ET0 is compared with FAWN's published ET0.
2. Leave-one-station-out interpolation: each station of the fixed set is predicted from
   the inverse-distance mean of the three nearest other stations (within 80 km), and the
   prediction is compared with the station's own monthly ET0.

Reads  data/processed/fawn_station_monthly.csv, fawn_stations.csv
Writes data/processed/fawn_cv_scaling.csv, fawn_cv_interpolation.csv (metrics)
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CAL_FROM = 2009


def metrics(pred, obs):
    d = pred - obs
    return {"n": len(d), "bias_pct": 100 * d.sum() / obs.sum(),
            "rmse_mm_month": float(np.sqrt((d ** 2).mean())),
            "r": float(np.corrcoef(pred, obs)[0, 1])}


def scaling_cv(mon):
    both = mon[(mon["year"] >= CAL_FROM)].dropna(subset=["eto_pm_mm", "eto_fawn_mm"])
    preds = []
    for y in sorted(both["year"].unique()):
        train, test = both[both["year"] != y], both[both["year"] == y]
        f_sm = (train.groupby(["station", "month"])["eto_fawn_mm"].sum()
                / train.groupby(["station", "month"])["eto_pm_mm"].sum()).rename("f")
        n_sm = train.groupby(["station", "month"]).size().rename("n")
        f_net = (train.groupby("month")["eto_fawn_mm"].sum()
                 / train.groupby("month")["eto_pm_mm"].sum()).rename("fn")
        t = test.join(f_sm, on=["station", "month"]).join(n_sm, on=["station", "month"]) \
                .join(f_net, on="month")
        # Same rule as script 05 in spirit: station factor when it has >= 5 training months.
        t["factor"] = np.where(t["n"].fillna(0) >= 5, t["f"], t["fn"])
        t["pred"] = t["eto_pm_mm"] * t["factor"]
        preds.append(t)
    p = pd.concat(preds)
    rows = [{"test": "leave-one-year-out scaling", "subset": "all", **metrics(p["pred"], p["eto_fawn_mm"])},
            {"test": "in-sample scaling (as in script 05)", "subset": "all",
             **metrics(both["eto_mm"], both["eto_fawn_mm"])},
            {"test": "unscaled daily FAO-56", "subset": "all",
             **metrics(both["eto_pm_mm"], both["eto_fawn_mm"])}]
    by_year = p.groupby("year").apply(lambda g: pd.Series(metrics(g["pred"], g["eto_fawn_mm"])),
                                      include_groups=False)
    return pd.DataFrame(rows), by_year


def interpolation_cv(var):
    spec = importlib.util.spec_from_file_location("s06", ROOT / "scripts" / "06_fawn_vs_gridmet.py")
    s06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s06)
    with contextlib.redirect_stdout(io.StringIO()):
        fawn = s06.fill_from_neighbour(s06.qc_relative(s06.load_fawn(var)))
    st = pd.read_csv(PROC / "fawn_stations.csv")
    share = fawn.groupby("station")["eto_mm"].count() / ((s06.END - s06.START + 1) * 12)
    st = st[st["station"].isin(share[share >= 0.8].index)].reset_index(drop=True)
    wide = fawn.pivot_table(index=["year", "month"], columns="station", values="eto_mm")[st["station"]]
    D = s06.haversine_km(st["lat"].values[:, None], st["lon"].values[:, None],
                         st["lat"].values[None, :], st["lon"].values[None, :])
    preds = []
    for i, s in enumerate(st["station"]):
        others = [j for j in range(len(st)) if j != i]
        for (y, m), v in wide.iterrows():
            if np.isnan(v.iloc[i]):
                continue
            cand = [(D[i, j], v.iloc[j]) for j in others if not np.isnan(v.iloc[j]) and D[i, j] <= 80]
            if not cand:
                continue
            cand = sorted(cand)[:3]
            w = np.array([1 / max(d, 1) ** 2 for d, _ in cand])
            preds.append({"station": s, "year": y, "month": m, "obs": v.iloc[i],
                          "pred": float((w * np.array([x for _, x in cand])).sum() / w.sum()),
                          "nearest_km": cand[0][0]})
    p = pd.DataFrame(preds)
    per = p.groupby("station").apply(lambda g: pd.Series(metrics(g["pred"], g["obs"])),
                                     include_groups=False)
    # Does interpolation distort the long-term change? Annual totals, 2000-05 vs 2019-25.
    ann = p.groupby(["station", "year"]).agg(n=("obs", "size"), obs=("obs", "sum"),
                                             pred=("pred", "sum")).reset_index()
    ann = ann[ann["n"] == 12]
    def change(g, col):
        a, b = g[g["year"] <= 2005][col].mean(), g[g["year"] >= 2019][col].mean()
        return 100 * (b / a - 1) if len(g[g["year"] <= 2005]) >= 3 and len(g[g["year"] >= 2019]) >= 3 else np.nan
    ch = ann.groupby("station").apply(lambda g: pd.Series({"obs_change_pct": change(g, "obs"),
                                                           "pred_change_pct": change(g, "pred")}),
                                      include_groups=False).dropna()
    return {"test": f"leave-one-station-out ({var})", "subset": f"{len(st)} stations",
            **metrics(p["pred"], p["obs"]),
            "median_nearest_km": p["nearest_km"].median(),
            "change_obs_minus_pred_pp_median": float((ch["obs_change_pct"] - ch["pred_change_pct"]).median()),
            "stations_with_change": len(ch)}, per


def main():
    mon = pd.read_csv(PROC / "fawn_station_monthly.csv")
    sc, by_year = scaling_cv(mon)
    loso = []
    for var in ("eto_mm", "eto_windclim_mm"):
        row, per = interpolation_cv(var)
        loso.append(row)
    out = pd.concat([sc, pd.DataFrame(loso)], ignore_index=True)
    out.to_csv(PROC / "fawn_cv.csv", index=False, float_format="%.3f")
    by_year.to_csv(PROC / "fawn_cv_scaling_by_year.csv", float_format="%.3f")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print("\nLeave-one-year-out scaling by year (bias %, RMSE):")
    print(by_year[["bias_pct", "rmse_mm_month", "r"]].round(2).T.to_string())


if __name__ == "__main__":
    main()
