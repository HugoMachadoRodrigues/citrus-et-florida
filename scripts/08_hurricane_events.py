"""Hurricane event study: drop and recovery of citrus satellite Kc after Florida hurricanes.

Exposure: HURDAT2 best tracks with wind radii (34/50/64 kt by quadrant, available since
2004), interpolated every 15 min. Each 1-km cell gets its peak wind during the event,
interpolated through the radii. Every cell reached tropical-storm force in every event, so
there is no unexposed control: the design compares cells along the wind gradient (terciles,
and a slope per 10 kt), with placebo dates in hurricane-free years as a check.

Response: monthly satellite Kc = OpenET ET / ETo (central ETo: FAWN, wind held at
climatology), as a relative anomaly against the same calendar month in a 3-year baseline
before the event (skipping years hit by, or just after, an earlier event).

Steps:
    python scripts/08_hurricane_events.py exposure   # wind class per cell and event
    python scripts/08_hurricane_events.py events     # event curves, drop and recovery
    python scripts/08_hurricane_events.py pretrend   # pre-event level, trend and quarterly gradients

Inputs:  data/raw/hurdat/hurdat2-*.txt, data/processed/citrus_cells_1km.csv,
         data/processed/series/openet_cells_*.csv, series/fawn_eto_cells_windclim.csv
Outputs: data/processed/hurricane_exposure.csv, hurricane_event_curves.csv,
         hurricane_event_summary.csv, hurricane_wind_gradient.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
HURDAT = sorted((ROOT / "data" / "raw" / "hurdat").glob("hurdat2-*.txt"))[-1]

# Event = storms that share a response window; month 0 = first Florida landfall month.
# "last": last month of follow-up, before the next event reaches the belt (or end of data).
EVENTS = {
    "2004 (Charley, Frances, Jeanne)": {"storms": ["AL032004", "AL062004", "AL112004"],
                                        "month": "2004-08", "baseline": [2001, 2002, 2003],
                                        "last": 13},   # Wilma = month 14
    "2005 (Wilma)": {"storms": ["AL252005"], "month": "2005-10",
                     "baseline": [2001, 2002, 2003], "last": 36},
    "2017 (Irma)": {"storms": ["AL112017"], "month": "2017-09",
                    "baseline": [2014, 2015, 2016], "last": 36},
    "2022 (Ian)": {"storms": ["AL092022"], "month": "2022-09",
                   "baseline": [2019, 2020, 2021], "last": 23},  # Helene = month 24
    "2024 (Helene, Milton)": {"storms": ["AL092024", "AL142024"], "month": "2024-09",
                              "baseline": [2019, 2020, 2021], "last": 15},  # end of data
}
WINDOW = (-12, 36)
RECOVERED = -0.02  # 3-month mean difference back within 2% of the control
NMI_KM = 1.852


# ---------------------------------------------------------------- exposure

def read_hurdat(path, ids):
    tracks, sid, rows = {}, None, []
    for line in path.read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        if f[0].startswith("AL") and len(f) <= 4:
            if sid in ids:
                tracks[sid] = pd.DataFrame(rows)
            sid, rows = f[0], []
            continue
        if sid not in ids:
            continue
        lat = float(f[4][:-1]) * (1 if f[4].endswith("N") else -1)
        lon = float(f[5][:-1]) * (-1 if f[5].endswith("W") else 1)
        r = [max(float(x), 0.0) * NMI_KM for x in f[8:20]]
        rows.append({"time": pd.to_datetime(f[0] + f[1], format="%Y%m%d%H%M"),
                     "lat": lat, "lon": lon, "vmax_kt": float(f[6]),
                     **{f"r{k}_{q}": r[i * 4 + j] for i, k in enumerate((34, 50, 64))
                        for j, q in enumerate(("ne", "se", "sw", "nw"))}})
    if sid in ids:
        tracks[sid] = pd.DataFrame(rows)
    return tracks


def interpolate(track, step="15min"):
    t = track.drop_duplicates("time").set_index("time")
    full = t.reindex(t.index.union(pd.date_range(t.index[0], t.index[-1], freq=step)))
    return full.interpolate(method="time").loc[lambda d: d.index.isin(
        pd.date_range(t.index[0], t.index[-1], freq=step))].reset_index()


def wind_at(d, vmax, rmw, r34, r50, r64):
    """Wind (kt) at distance d (km) from the centre in one quadrant, piecewise-linear
    through the HURDAT2 radii: vmax at the radius of maximum wind, 64/50/34 kt at their
    radii, then decaying as (r34/d)^0.5 outside the 34-kt radius."""
    pts_r, pts_v = [0.0, rmw], [0.0, vmax]
    for r, v in ((r64, 64), (r50, 50), (r34, 34)):
        if r > pts_r[-1] and v < pts_v[-1]:
            pts_r.append(r)
            pts_v.append(v)
    w = np.interp(d, pts_r, pts_v)
    edge_r, edge_v = pts_r[-1], pts_v[-1]
    return np.where(d > edge_r, edge_v * np.sqrt(edge_r / np.maximum(d, 1e-6)), w)


def exposure(args):
    cells = pd.read_csv(PROC / "citrus_cells_1km.csv")
    ids = {s for e in EVENTS.values() for s in e["storms"]}
    tracks = read_hurdat(HURDAT, ids)
    missing = ids - set(tracks)
    if missing:
        raise SystemExit(f"Storms not found in {HURDAT.name}: {missing}")
    lat_c, lon_c = np.radians(cells["lat"].values), np.radians(cells["lon"].values)
    out = cells[["cell_id", "county", "citrus_ha"]].copy()
    for name, ev in EVENTS.items():
        peak = np.zeros(len(cells))
        dmin = np.full(len(cells), np.inf)
        for sid in ev["storms"]:
            for r in interpolate(tracks[sid]).itertuples():
                la, lo = np.radians(r.lat), np.radians(r.lon)
                dlat, dlon = lat_c - la, lon_c - lo
                a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(lat_c) * np.sin(dlon / 2) ** 2
                d = 6371 * 2 * np.arcsin(np.sqrt(a))
                if d.min() > 600:
                    continue
                dmin = np.minimum(dmin, d)
                brg = (np.degrees(np.arctan2(
                    np.sin(dlon) * np.cos(lat_c),
                    np.cos(la) * np.sin(lat_c) - np.sin(la) * np.cos(lat_c) * np.cos(dlon)))
                    + 360) % 360
                quad = np.select([brg < 90, brg < 180, brg < 270], [0, 1, 2], 3)
                for qi, q in enumerate(("ne", "se", "sw", "nw")):
                    sel = quad == qi
                    if not sel.any():
                        continue
                    r64 = getattr(r, f"r64_{q}")
                    rmw = r64 * 0.5 if r64 > 0 else 30.0  # HURDAT2 has no RMW before 2021
                    v = wind_at(d[sel], r.vmax_kt, rmw, getattr(r, f"r34_{q}"),
                                getattr(r, f"r50_{q}"), r64)
                    peak[sel] = np.maximum(peak[sel], v)
        out[name] = peak.round(1)
        out[name + " dist_km"] = dmin.round(1)
        q = np.percentile(peak, [0, 33, 67, 100])
        print(f"{name:32} peak wind kt: min {q[0]:.0f}, terciles {q[1]:.0f} / {q[2]:.0f}, "
              f"max {q[3]:.0f}; nearest track median {np.median(dmin):.0f} km")
    out.to_csv(PROC / "hurricane_exposure.csv", index=False)


# ---------------------------------------------------------------- events

def monthly_kc(model="ensemble"):
    sdir = PROC / "series" if model == "ensemble" else PROC / "series" / model
    files = sorted(sdir.glob("openet_cells_*.csv"))
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    eto = pd.read_csv(PROC / "series" / "fawn_eto_cells_windclim.csv")
    m = m.merge(eto[["cell_id", "year", "month", "eto_fawn_mm"]],
                on=["cell_id", "year", "month"], how="left")
    max_px = m.groupby("cell_id")["et_px"].transform("max")
    ok = (m["et_px"] >= 0.5 * max_px) & m["et_mm"].notna() & (m["eto_fawn_mm"] > 0)
    m["kc"] = np.where(ok, m["et_mm"] / m["eto_fawn_mm"], np.nan)
    m["t"] = m["year"] * 12 + m["month"] - 1  # month index
    return m[["cell_id", "year", "month", "t", "kc"]]


def run_mean3(s):
    return s.rolling(3, center=True, min_periods=2).mean()


# Pseudo-events in years without a hurricane over the citrus belt (placebo test): each real
# event's wind field is applied to these dates, so any "effect" is geography, not wind.
# Every September whose whole window (-12 to +24 months) is free of the events above.
PLACEBO_MONTHS = [f"{y}-09" for y in range(2007, 2015)] + ["2019-09"]
WINDOWS = {"0-6": (0, 6), "7-12": (7, 12), "13-24": (13, 24)}
BLOCK_KM = 20
NBOOT = 1000
TERCILES = ["low", "mid", "high"]


def anomalies(m, month, baseline, last=WINDOW[1]):
    """Relative Kc anomaly per cell-month in the event window vs. the baseline years."""
    y0, mo0 = map(int, month.split("-"))
    t0 = y0 * 12 + mo0 - 1
    base = m[m["year"].isin(baseline)].groupby(["cell_id", "month"])["kc"].agg(["mean", "count"])
    base = base[base["count"] >= 2]["mean"].rename("kc_base").reset_index()
    w = m[m["t"].between(t0 + WINDOW[0], t0 + min(WINDOW[1], last))].merge(
        base, on=["cell_id", "month"])
    w["rel"] = w["kc"] / w["kc_base"] - 1
    w["k"] = w["t"] - t0
    w = w.dropna(subset=["rel"])
    lo, hi = w["rel"].quantile([0.01, 0.99])  # trims cloud-filled cell-months
    return w[w["rel"].between(lo, hi)]


def wls_slope(x, y, wt):
    """Weighted least-squares slope and its (naive, iid) standard error."""
    X = np.column_stack([np.ones_like(x), x])
    W = wt / wt.mean()
    XtW = X.T * W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    res = y - X @ beta
    sigma2 = (W * res ** 2).sum() / (len(y) - 2)
    cov = sigma2 * np.linalg.inv(XtW @ X)
    return beta[1], np.sqrt(cov[1, 1])


def gradient(w, wind, label, blocks=None, boot=False):
    """Cell-level change (window mean minus the 12 pre-event months) against peak wind.

    Also reports the pre-event gradient (mean of months -12..-1 against wind), a check that
    windier and calmer cells were on the same path before the event, and, with boot=True,
    95% intervals from a spatial block bootstrap (BLOCK_KM squares)."""
    bins = [-13, -1] + [WINDOWS[k][1] for k in WINDOWS]
    cell = w.assign(win=pd.cut(w["k"], bins, labels=["pre"] + list(WINDOWS)))
    c = cell.groupby(["cell_id", "win"], observed=True)["rel"].mean().unstack()
    c = c.join(wind, on="cell_id")
    out = {"event": label, "cells": int(c["pre"].notna().sum())}
    cc = c.dropna(subset=["pre"])
    b, _ = wls_slope(cc["peak_kt"].values, cc["pre"].values, cc["citrus_ha"].values)
    out["slope_pre_per10kt"] = 10 * b
    for win in WINDOWS:
        if win not in c or c[win].notna().sum() < 100 or w["k"].max() < WINDOWS[win][1]:
            out[f"slope_{win}_per10kt"] = np.nan
            continue
        cc = c.dropna(subset=["pre", win])
        x, y, wt = cc["peak_kt"].values, (cc[win] - cc["pre"]).values, cc["citrus_ha"].values
        out[f"slope_{win}_per10kt"] = 10 * wls_slope(x, y, wt)[0]
        if boot and blocks is not None:
            blk = blocks.reindex(cc.index).values
            ub = np.unique(blk)
            rng = np.random.default_rng(1)
            idx_by = {u: np.where(blk == u)[0] for u in ub}
            bs = []
            for _ in range(NBOOT):
                pick = rng.choice(ub, len(ub))
                ii = np.concatenate([idx_by[u] for u in pick])
                bs.append(10 * wls_slope(x[ii], y[ii], wt[ii])[0])
            out[f"ci_lo_{win}"], out[f"ci_hi_{win}"] = np.percentile(bs, [2.5, 97.5])
    return out


def events(args):
    expo = pd.read_csv(PROC / "hurricane_exposure.csv")
    cl = pd.read_csv(PROC / "citrus_cells_1km.csv").set_index("cell_id")
    blocks = ((cl["x_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str) + "_"
              + (cl["y_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str))
    m = monthly_kc(args.model)
    sfx = "" if args.model == "ensemble" else f"_{args.model}"
    curves, summary, grads = [], [], []
    for name, ev in EVENTS.items():
        wind = expo.set_index("cell_id")[[name, "citrus_ha"]].rename(columns={name: "peak_kt"})
        wind["tercile"] = pd.qcut(wind["peak_kt"], 3, labels=TERCILES)
        w = anomalies(m, ev["month"], ev["baseline"], ev["last"]).join(wind, on="cell_id")
        g = (w.groupby(["tercile", "k"], observed=True)
             .apply(lambda d: pd.Series({
                 "rel": np.average(d["rel"], weights=d["citrus_ha"]), "cells": len(d)}),
                 include_groups=False)
             .reset_index())
        low = g[g["tercile"] == "low"].set_index("k")["rel"]
        g["did"] = g["rel"] - g["k"].map(low)
        g.insert(0, "event", name)
        curves.append(g)

        for terc in TERCILES:
            c = g[g["tercile"] == terc].set_index("k").sort_index()
            pre = c.loc[c.index < 0, "rel"].mean()
            d = run_mean3(c["rel"] - pre)
            post = d.loc[0:12]
            trough_k = int(post.idxmin())
            rec = d.loc[trough_k:][lambda x: x >= RECOVERED]
            row = {"event": name, "tercile": terc,
                   "peak_kt": wind.loc[wind["tercile"] == terc, "peak_kt"].median(),
                   "drop_0_6": (c.loc[0:6, "rel"] - pre).mean(),
                   "trough": post.min(), "trough_month": trough_k,
                   "convergence_month": int(rec.index[0]) if len(rec) else np.nan,
                   "post_7_12": (c.loc[7:12, "rel"] - pre).mean(),
                   "post_13_24": (c.loc[13:24, "rel"] - pre).mean() if c.index.max() >= 24 else np.nan,
                   "last_month": int(c.index.max())}
            if terc != "low":
                cd = c["did"] - c.loc[c.index < 0, "did"].mean()
                row["did_0_6"] = cd.loc[0:6].mean()
                row["did_7_12"] = cd.loc[7:12].mean()
                row["did_13_24"] = cd.loc[13:24].mean() if c.index.max() >= 24 else np.nan
            summary.append(row)
        grads.append(gradient(w, wind, name, blocks, boot=True))
        for pm in PLACEBO_MONTHS:
            y = int(pm[:4])
            # Baseline: up to 3 of the 5 prior years, skipping event years and the year after.
            hit = {2004, 2005, 2006, 2017, 2018, 2022, 2023, 2024, 2025}
            base_years = [b for b in range(y - 5, y) if b not in hit][-3:]
            pw = anomalies(m, pm, base_years, 24).join(wind, on="cell_id")
            grads.append(gradient(pw, wind, f"placebo {pm} with {name} wind"))

    pd.concat(curves).to_csv(PROC / f"hurricane_event_curves{sfx}.csv", index=False,
                             float_format="%.4f")
    summary = pd.DataFrame(summary)
    summary.to_csv(PROC / f"hurricane_event_summary{sfx}.csv", index=False, float_format="%.4f")
    grads = pd.DataFrame(grads)
    grads.to_csv(PROC / f"hurricane_wind_gradient{sfx}.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print("Relative Kc anomaly vs. own baseline by peak-wind tercile (pre-event level removed);"
          " did = minus the low-wind tercile:")
    print(summary.round(3).to_string(index=False))
    real = grads[~grads["event"].str.startswith("placebo")].copy()
    for win in WINDOWS:
        col = f"slope_{win}_per10kt"
        for i, r in real.iterrows():
            pl = grads[grads["event"].str.startswith("placebo")
                       & grads["event"].str.endswith(f"with {r['event']} wind")][col].dropna()
            real.loc[i, f"placebo_min_{win}"] = pl.min()
            real.loc[i, f"placebo_max_{win}"] = pl.max()
            real.loc[i, f"placebo_rank_{win}"] = (pl < r[col]).sum() if pd.notna(r[col]) else np.nan
    real.to_csv(PROC / f"hurricane_gradient_summary{sfx}.csv", index=False, float_format="%.4f")
    print("\nChange in relative Kc per +10 kt of peak wind, 95% spatial-block CI, and the range"
          " of 9 placebo slopes (rank = placebos below the event slope):")
    print((real.set_index("event") * 1).round(2).T.to_string())


def pretrend(args):
    """Pre-event trajectories along the wind gradient (second internal review).

    For each event, over months -12 to -1: (i) the gradient of the mean pre-event anomaly
    (level) on peak wind; (ii) the gradient on peak wind of each cell's within-window linear
    trend of the anomaly (exposure x time interaction, per 12 months); (iii) the gradient for
    each pre-event quarter (months -12..-10, -9..-7, -6..-4, -3..-1). All per +10 kt, with
    95% spatial block-bootstrap intervals (20-km blocks, 1000 replicates)."""
    expo = pd.read_csv(PROC / "hurricane_exposure.csv")
    cl = pd.read_csv(PROC / "citrus_cells_1km.csv").set_index("cell_id")
    blocks = ((cl["x_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str) + "_"
              + (cl["y_albers"] // (BLOCK_KM * 1000)).astype(int).astype(str))
    m = monthly_kc(args.model)
    sfx = "" if args.model == "ensemble" else f"_{args.model}"
    rows = []
    for name, ev in EVENTS.items():
        wind = expo.set_index("cell_id")[[name, "citrus_ha"]].rename(columns={name: "peak_kt"})
        w = anomalies(m, ev["month"], ev["baseline"], ev["last"])  # same trimming as events
        w = w[w["k"] < 0]
        w["q"] = pd.cut(w["k"], [-13, -10, -7, -4, -1], labels=["-12..-10", "-9..-7", "-6..-4", "-3..-1"])
        cell = w.groupby("cell_id").apply(lambda d: pd.Series({
            "level": d["rel"].mean(),
            "trend12": 12 * np.polyfit(d["k"], d["rel"], 1)[0] if d["k"].nunique() >= 6 else np.nan}),
            include_groups=False)
        q = w.groupby(["cell_id", "q"], observed=True)["rel"].mean().unstack()
        cell = cell.join(q).join(wind)
        cell["block"] = blocks.reindex(cell.index).values
        for resp in ["level", "trend12", "-12..-10", "-9..-7", "-6..-4", "-3..-1"]:
            cc = cell.dropna(subset=[resp, "peak_kt"])
            x, y, wt = cc["peak_kt"].values, cc[resp].values, cc["citrus_ha"].values
            est = 10 * wls_slope(x, y, wt)[0]
            blk = cc["block"].values
            ub = np.unique(blk)
            idx_by = {u: np.where(blk == u)[0] for u in ub}
            rng = np.random.default_rng(2)
            bs = []
            for _ in range(NBOOT):
                ii = np.concatenate([idx_by[u] for u in rng.choice(ub, len(ub))])
                bs.append(10 * wls_slope(x[ii], y[ii], wt[ii])[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({"event": name, "response": resp, "cells": len(cc),
                         "slope_per10kt": est, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / f"hurricane_pretrend{sfx}.csv", index=False, float_format="%.5f")
    out[["slope_per10kt", "ci_lo", "ci_hi"]] *= 100
    pd.set_option("display.width", 200)
    print("Pre-event gradients on peak wind (% per +10 kt; trend12 = % per 12 months per +10 kt):")
    print(out.round(2).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("step", choices=["exposure", "events", "pretrend"])
    p.add_argument("--model", choices=["ensemble", "disalexi", "eemetric", "geesebal", "ptjpl", "sims", "ssebop"], default="ensemble",
                   help="OpenET ensemble or a single model (events step)")
    args = p.parse_args()
    {"exposure": exposure, "events": events, "pretrend": pretrend}[args.step](args)


if __name__ == "__main__":
    main()
