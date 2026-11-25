"""FAWN reference ET (ETo) for the citrus belt, 2000-2025.

FAWN publishes ETo only from 2009 in its report generator, and the yearly archive
(daily_summaries/BY_YEAR) lacks the southern stations in 2008-2017. To get one
consistent series, this script recomputes daily grass ETo with FAO-56 Penman-Monteith
from FAWN daily weather (2-m air temperature, dew point, 10-m wind, solar radiation),
and keeps FAWN's own ETo only as a check.

Steps (downloads are cached):
    python scripts/05_fawn_eto.py fetch    # station list + daily weather per station
    python scripts/05_fawn_eto.py build    # daily FAO-56 ETo -> monthly per station

Inputs:
    data/raw/fawn/daily/<year>_daily*.{csv,zip}  yearly archive (fills 2000 and gaps)
Outputs:
    data/raw/fawn/stations_latestmap.json          FAWN station list with coordinates
    data/raw/fawn/report/<station>.csv             report-generator daily weather
    data/processed/fawn_stations.csv               citrus-belt stations
    data/processed/fawn_daily.csv                  daily weather + ETo per station
    data/processed/fawn_station_monthly.csv        monthly ETo per station
        eto_mm           daily FAO-56 scaled to FAWN's published level (per station x month)
        eto_windclim_mm  same, with wind fixed at the station's monthly climatology
"""

import argparse
import glob
import json
import math
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fawn"
PROC = ROOT / "data" / "processed"
STATIONS_URL = "https://fawn.ifas.ufl.edu/controller.php/latestmapjson/"
REPORT_URL = "https://fawn.ifas.ufl.edu/data/reports/?res"
CITRUS_BELT_BOX = [-83.0, 25.8, -79.9, 29.4]  # same box as script 02
START, END = 2000, 2025
MIN_DAYS = 22  # valid ETo days needed for a monthly total
CAL_FROM = 2009  # FAWN publishes ETo with its current method from 2009
CAL_MIN_DAYS = 150  # overlap days needed for a station-specific monthly factor


# ---------------------------------------------------------------- fetch

def fetch(args):
    RAW.mkdir(parents=True, exist_ok=True)
    meta_path = RAW / "stations_latestmap.json"
    if not meta_path.exists() or args.force:
        req = urllib.request.Request(STATIONS_URL, headers={"Accept-Encoding": "identity"})
        meta_path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    stations = belt_stations(meta_path)
    PROC.mkdir(parents=True, exist_ok=True)
    stations.to_csv(PROC / "fawn_stations.csv", index=False)
    print(f"{len(stations)} FAWN stations in the citrus belt")

    out_dir = RAW / "report"
    out_dir.mkdir(exist_ok=True)
    for r in stations.itertuples():
        out = out_dir / f"{r.station}.csv"
        if out.exists() and not args.force:
            continue
        form = {
            f"locs__{r.station}": "on", "reportType": "daily", "presetRange": "dates",
            "fromDate_m": 1, "fromDate_d": 1, "fromDate_y": START,
            "toDate_m": 12, "toDate_d": 31, "toDate_y": END,
            "vars__AirTemp9": "on", "vars__DewPoint": "on", "vars__RelHumAvg": "on",
            "vars__WindSpeed": "on", "vars__TotalRad": "on", "vars__Rainfall": "on",
            "vars__ET": "on", "vars__BP": "on", "format": ".CSV (Excel)",
        }
        t = time.time()
        data = urllib.parse.urlencode(form).encode()
        body = urllib.request.urlopen(REPORT_URL, data=data, timeout=600).read()
        if not body.lstrip().startswith(b'"FAWN Station"'):
            print(f"{r.station} {r.name}: not offered by the report generator, skipped")
            continue
        out.write_bytes(body)
        print(f"{r.station} {r.name}: {len(body) / 1e6:.1f} MB ({time.time() - t:.0f} s)")


def belt_stations(meta_path):
    st = pd.DataFrame(json.loads(meta_path.read_text())["stnsWxData"])
    st = st.rename(columns={"stnID": "station", "stnName": "name", "lng": "lon"})
    st["station"] = st["station"].astype(int)
    st["elev_m"] = st["elevation_feet"].astype(float) * 0.3048
    # Florida stations sit below ~100 m; Dade City is listed at 10,000 ft (a typo).
    st.loc[st["elev_m"] > 150, "elev_m"] = 30.0
    w, s, e, n = CITRUS_BELT_BOX
    st = st[st["lon"].between(w, e) & st["lat"].between(s, n)]
    return st[["station", "name", "county", "lat", "lon", "elev_m"]].sort_values("station")


# ---------------------------------------------------------------- build

def read_report(path, station):
    d = pd.read_csv(path, na_values=["N/A", ""])
    f2c = lambda f: (f - 32) * 5 / 9
    return pd.DataFrame({
        "station": station,
        "date": pd.to_datetime(d["Period"], format="%d %b %Y"),
        "tavg": f2c(d["2m T avg (F)"]), "tmin": f2c(d["2m T min (F)"]),
        "tmax": f2c(d["2m T max (F)"]), "tdew": f2c(d["2m DewPt avg (F)"]),
        "wind10": d["10m Wind avg (mph)"] * 0.44704,
        "rs_wm2": d["SolRad avg2m  (w/m^2)"],
        "rain_mm": d["2m Rain tot (in)"] * 25.4,
        "nobs": d["N (# obs)"],
        "eto_fawn": d["ET (in)"] * 25.4,
    })


def read_archive():
    """Yearly archive files; used for days the report generator does not return."""
    frames = []
    for f in sorted(glob.glob(str(RAW / "daily" / "*"))):
        if f.endswith(".zip"):
            z = zipfile.ZipFile(f)
            d = pd.read_csv(z.open([n for n in z.namelist() if n.endswith(".csv")][0]),
                            low_memory=False, on_bad_lines="skip")
        else:
            d = pd.read_csv(f, low_memory=False, on_bad_lines="skip")
        col = lambda *keys: next(c for c in d.columns if all(k in c for k in keys))
        num = lambda c: pd.to_numeric(d[c], errors="coerce")
        if "date" in d.columns:  # 2000-2020 layout
            frame = pd.DataFrame({
                "date": pd.to_datetime(d["date"], errors="coerce"),
                "tavg": num("avg_temp_air_2m_C"), "tmin": num("min_temp_air_2m_C"),
                "tmax": num("max_temp_air_2m_C"), "tdew": num("avg_temp_dp_2m_C"),
                "wind10": num("avg_wind_speed_10m_mph") * 0.44704,
                "rs_wm2": num("avg_rfd_2m_wm2"),
                "rain_mm": num("sum_rain_2m_inches") * 25.4,
                "nobs": num("num_observations"), "eto_fawn": num("ETo_grass_mm"),
            })
        else:  # 2021+ layout
            frame = pd.DataFrame({
                "date": pd.to_datetime(d[col("Date")], format="mixed", errors="coerce"),
                "tavg": num(col("2m", "Avg")), "tmin": num(col("2m", "Min")),
                "tmax": num(col("2m", "Max")), "tdew": num(col("Dew", "Avg")),
                "wind10": num(col("Wind Speed", "Avg")) * 0.44704,
                "rs_wm2": num(col("w/m2")), "rain_mm": num(col("Rain")) * 25.4,
                "nobs": num(col("Observations")), "eto_fawn": num(col("ETo")),
            })
        frame.insert(0, "station", pd.to_numeric(d["StationID"], errors="coerce"))
        frames.append(frame.dropna(subset=["station", "date"]))
    out = pd.concat(frames, ignore_index=True)
    out["station"] = out["station"].astype(int)
    return out.drop_duplicates(["station", "date"])


def fao56_eto(df, lat_deg, elev_m):
    """Daily grass reference ET (mm/day), FAO-56 eq. 6, from daily means."""
    tmax, tmin, tdew = df["tmax"], df["tmin"], df["tdew"]
    tmean = (tmax + tmin) / 2
    es_t = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    es = (es_t(tmax) + es_t(tmin)) / 2
    ea = es_t(tdew)
    delta = 4098 * es_t(tmean) / (tmean + 237.3) ** 2
    p = 101.3 * ((293 - 0.0065 * elev_m) / 293) ** 5.26
    gamma = 0.000665 * p
    u2 = df["wind10"] * 4.87 / math.log(67.8 * 10 - 5.42)
    rs = df["rs_wm2"] * 0.0864  # W/m2 daily mean -> MJ/m2/day
    j = df["date"].dt.day_of_year
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dec = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(-math.tan(phi) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * np.sin(dec) + math.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev_m) * ra
    rns = 0.77 * rs
    sigma = 4.903e-9
    rnl = (sigma * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(ea.clip(lower=0)))
           * (1.35 * (rs / rso).clip(upper=1.0) - 0.35))
    rn = rns - rnl
    eto = ((0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (es - ea))
           / (delta + gamma * (1 + 0.34 * u2)))
    return eto.clip(lower=0)


def qc(d):
    """Flag physically implausible daily values (set to NaN)."""
    d = d.copy()
    bad = ((d["tmax"] < d["tmin"]) | ~d["tmax"].between(-10, 45) | ~d["tmin"].between(-15, 35)
           | (d["tdew"] > d["tmax"] + 1) | ~d["rs_wm2"].between(5, 450)
           | ~d["wind10"].between(0, 25) | (d["nobs"] < 77))  # < 80% of the 96 15-min records
    d.loc[bad, ["tmax", "tmin", "tdew", "rs_wm2", "wind10"]] = np.nan
    return d, int(bad.sum())


def build(args):
    stations = pd.read_csv(PROC / "fawn_stations.csv")
    archive = read_archive()
    daily = []
    for r in stations.itertuples():
        path = RAW / "report" / f"{r.station}.csv"
        rep = read_report(path, r.station) if path.exists() else pd.DataFrame()
        arc = archive[archive["station"] == r.station]
        # Report generator first; archive only for days the report does not have.
        d = pd.concat([rep, arc[~arc["date"].isin(rep.get("date", []))]], ignore_index=True)
        d = d[d["date"].dt.year.between(START, END)].sort_values("date")
        d, n_bad = qc(d)
        d["eto_pm"] = fao56_eto(d, r.lat, r.elev_m)
        # Wind-homogenized variant: FAWN 10-m wind falls ~20%/decade at most stations
        # (ERA5-Land: ~-3%), a sign of growing vegetation around the masts or sensor
        # ageing. Replace daily wind with the station's calendar-month mean.
        clim = d.groupby(d["date"].dt.month)["wind10"].transform("mean")
        d["eto_pm_wc"] = fao56_eto(d.assign(wind10=clim.where(d["wind10"].notna())),
                                   r.lat, r.elev_m)
        daily.append(d)
        print(f"{r.station:>4} {r.name:<18} {d['date'].min():%Y-%m-%d} -> "
              f"{d['date'].max():%Y-%m-%d}  days {len(d):>5}  ETo days "
              f"{d['eto_pm'].notna().sum():>5}  QC removed {n_bad}")
    daily = pd.concat(daily, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily.to_csv(PROC / "fawn_daily.csv", index=False, float_format="%.3f")

    daily["year"], daily["month"] = daily["date"].dt.year, daily["date"].dt.month
    g = daily.groupby(["station", "year", "month"])
    mon = g.agg(n_valid=("eto_pm", "count"), eto_mean=("eto_pm", "mean"),
                wc_mean=("eto_pm_wc", "mean"),
                fawn_n=("eto_fawn", "count"), fawn_mean=("eto_fawn", "mean"),
                rain_mm=("rain_mm", "sum")).reset_index()
    ndays = pd.to_datetime(dict(year=mon["year"], month=mon["month"], day=1)).dt.days_in_month
    ok = mon["n_valid"] >= MIN_DAYS
    mon["eto_pm_mm"] = np.where(ok, mon["eto_mean"] * ndays, np.nan)
    mon["eto_fawn_mm"] = np.where(mon["fawn_n"] >= MIN_DAYS, mon["fawn_mean"] * ndays, np.nan)
    mon["eto_pm_wc_mm"] = np.where(ok, mon["wc_mean"] * ndays, np.nan)
    mon = mon.drop(columns=["eto_mean", "fawn_mean", "wc_mean"])

    # Daily FAO-56 runs above FAWN's published (hourly-based) ETo, mostly in winter
    # when the daily temperature range is wide. Scale it to FAWN's level with one
    # factor per station and calendar month, fitted on days with both (2009 on).
    fac = calibration_factors(daily)
    mon = mon.merge(fac, on=["station", "month"], how="left")
    mon["eto_mm"] = mon["eto_pm_mm"] * mon["factor"]
    mon["eto_windclim_mm"] = mon["eto_pm_wc_mm"] * mon["factor"]
    mon.to_csv(PROC / "fawn_station_monthly.csv", index=False, float_format="%.2f")

    both = mon[mon["year"] >= CAL_FROM].dropna(subset=["eto_mm", "eto_fawn_mm"])
    print(f"\nMonthly factor (FAWN / daily FAO-56), network median by month: " + ", ".join(
        f"{m}:{v:.2f}" for m, v in fac.groupby("month")["factor"].median().items()))
    for col, label in [("eto_pm_mm", "daily FAO-56, raw   "), ("eto_mm", "daily FAO-56, scaled")]:
        diff = both[col] - both["eto_fawn_mm"]
        r = np.corrcoef(both[col], both["eto_fawn_mm"])[0, 1]
        print(f"{label} vs FAWN published, {len(both)} station-months {CAL_FROM}+: "
              f"bias {diff.sum() / both['eto_fawn_mm'].sum():+.1%}, "
              f"RMSE {np.sqrt((diff ** 2).mean()):.1f} mm/month, r {r:.3f}")
    cov = (mon.assign(ok=mon["eto_mm"].notna()).groupby("station")["ok"].sum()
           .rename("valid_months").reset_index().merge(stations, on="station"))
    cov["share"] = cov["valid_months"] / ((END - START + 1) * 12)
    print("\nValid monthly ETo by station:")
    print(cov.sort_values("share", ascending=False)[
        ["station", "name", "county", "valid_months", "share"]].to_string(index=False))


def calibration_factors(daily):
    both = daily[(daily["year"] >= CAL_FROM) & daily["eto_pm"].notna()
                 & daily["eto_fawn"].notna()]
    g = both.groupby(["station", "month"]).agg(
        n=("eto_pm", "size"), pm=("eto_pm", "sum"), fawn=("eto_fawn", "sum")).reset_index()
    g["factor"] = g["fawn"] / g["pm"]
    net = (both.groupby("month")["eto_fawn"].sum() / both.groupby("month")["eto_pm"].sum())
    grid = pd.MultiIndex.from_product([daily["station"].unique(), range(1, 13)],
                                      names=["station", "month"]).to_frame(index=False)
    fac = grid.merge(g[g["n"] >= CAL_MIN_DAYS][["station", "month", "factor"]],
                     on=["station", "month"], how="left")
    fac["factor"] = fac["factor"].fillna(fac["month"].map(net))
    return fac


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["fetch", "build"])
    p.add_argument("--force", action="store_true", help="download again")
    args = p.parse_args()
    fetch(args) if args.step == "fetch" else build(args)


if __name__ == "__main__":
    main()
