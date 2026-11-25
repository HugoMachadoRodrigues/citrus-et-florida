"""Tables 7 and 8 of the manuscript and the numbers quoted in Section 3.7, from one set of outputs.

All values are read from the CSV files written by `23_structure_soil.py naip` and rounded with
one rule (half away from zero, applied to the stored decimal text), so that every repeated
number in the manuscript comes from the same output. Values whose absolute size is below 0.05
are given with two decimals so that the side of zero stays visible.

Reads  data/processed/structure_soil_regression_naip.csv, structure_soil_stable_naip.csv,
       structure_soil_strata_naip.csv, structure_soil_naip_summary.csv
Writes manuscript/tables_canopy.md (Markdown pasted into manuscript_source.md) and prints the
       numbers quoted in the text of Section 3.7 and in the caption of Figure 8.
"""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GROUPS = ["All", "Ridge", "Southwest Flatwoods", "Indian River"]
LABEL = {"All": "All cells"}
WINDOWS = {"recent": "2015–2019 → 2021–2025", "aligned": "2017–2019 → 2023",
           "to2023": "2015–2019 → 2021–2023"}
RULES = {"cell": "Per cell and year", "year": "Per year", "year_region": "Per year and region"}


def read(name):
    return pd.read_csv(PROC / name, dtype=str)


def r(x, d=1, sign=False):
    """Round decimal text half away from zero; two decimals when |x| < 0.05."""
    v = Decimal(str(x))
    if d == 1 and abs(v) < Decimal("0.05"):
        d = 2
    q = v.quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:+}" if sign else f"{q}"
    if q == 0:
        s = s.lstrip("+-")
    return s.replace("-", "−")


def ci(e, lo, hi, d=1, sign=True):
    return f"{r(e, d, sign)} ({r(lo, d)}, {r(hi, d)})"


def main():
    reg = read("structure_soil_regression_naip.csv")
    reg = reg[reg["aws"] == "aws0_100"]
    stab = read("structure_soil_stable_naip.csv")
    strata = read("structure_soil_strata_naip.csv")
    summ = read("structure_soil_naip_summary.csv").set_index("region")

    def row(period, rule, resp, group):
        q = reg[(reg["period"] == period) & (reg["rule"] == rule) & (reg["response"] == resp)
                & (reg["group"] == group)]
        return q.iloc[0]

    out = []
    # Table 7: main window, per-cell threshold.
    out += ["| Group | Canopy index change (pp) | ET, no canopy change | K~c,act~, no canopy change "
            "| ET per +1 pp canopy | ET per +10 mm AWS |",
            "| ---------- | ------ | ---------- | ---------- | ---------- | ---------- |"]
    for g in GROUPS:
        e, k = row("recent", "cell", "dln_et", g), row("recent", "cell", "dln_kc", g)
        out.append(f"| {LABEL.get(g, g)} | {r(summ.loc[g, 'dnaip_pp'])} "
                   f"| {ci(e.intercept, e.int_lo, e.int_hi)} | {ci(k.intercept, k.int_lo, k.int_hi)} "
                   f"| {ci(e.per_pp_tcc, e.tcc_lo, e.tcc_hi, 2)} "
                   f"| {ci(e.per_10mm_aws, e.aws_lo, e.aws_hi, 2)} |")
    # Table 8: ET change not explained by the index, by ET window and threshold rule.
    out += ["", "<!-- Table 8 -->",
            "| ET window | Threshold rule | Ridge | Southwest Flatwoods |",
            "| ---------- | ---------- | ---------- | ---------- |"]
    for w, wl in WINDOWS.items():
        for ru, rl in RULES.items():
            q1, q2 = row(w, ru, "dln_et", "Ridge"), row(w, ru, "dln_et", "Southwest Flatwoods")
            out.append(f"| {wl} | {rl} | {ci(q1.intercept, q1.int_lo, q1.int_hi)} "
                       f"| {ci(q2.intercept, q2.int_lo, q2.int_hi)} |")
    # Numbers for the text (small-change limits) and the Figure 8c caption (tercile sizes).
    s_ = stab[stab["period"] == "recent"]
    for g in ("Ridge", "Southwest Flatwoods"):
        q = s_[s_["group"] == g]
        print(f"{g}: limits {list(q.limit_pp)} cells {list(q.cells)} area share "
              f"{[r(100 * float(x), 0) for x in q.share_area]} ET {[r(x, sign=True) for x in q.dln_et]}")
    st = strata[strata["period"] == "recent"]
    print("Figure 8c cells:", [(x.group, x.aws_tercile, x.cells) for x in st.itertuples()])
    (ROOT / "manuscript" / "tables_canopy.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))

    # Numbers quoted in the text of Section 3.7.
    print("\nText of Section 3.7:")
    for w in WINDOWS:
        for g in ("Ridge", "Southwest Flatwoods"):
            vals = [r(row(w, ru, "dln_et", g).intercept, sign=True) for ru in RULES]
            q = row(w, "cell", "dln_et", g)
            print(f"{w:8} {g:20} cell {ci(q.intercept, q.int_lo, q.int_hi)}; rules {vals}")


if __name__ == "__main__":
    main()
