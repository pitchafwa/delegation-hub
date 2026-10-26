"""Short-term program, Phase 5 (RESEARCH_short_term_projection_plan.md): horizon, fade and uncertainty for model A.
  1  fade by DAYS ahead (the plan works by date): how much of each signal persists at 1, 2, 3, 4-5, 6-7, 8-10, 11-14 days; is two weeks useful
  2  spread of outcomes by player level and horizon (single game and 7-day total)
  3  7-day calibration of A x chance model (re-centred), by player level
Run from ingest/:  .venv/Scripts/python research/st_phase5.py   (needs data/st_phase4_avail_recentred.parquet from st_avail_recentre.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_harness as H                                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
D, P, T = H.D, H.P, H.T
OUT = {}
A = H.a_feats()
DB = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 10), (11, 14)]
TP = H.TP.copy()
TP["db"] = TP["days_ahead"].map(lambda d: next((i for i, (lo, hi) in enumerate(DB) if lo <= d <= hi), np.nan))


def fit_rows(rows, feats):
    X = np.column_stack([np.ones(len(rows))] + [rows["info_id"].map(f).to_numpy() for f in feats.values()])
    y = rows["fp"].to_numpy() - rows["info_id"].map(P["delco"]).to_numpy()
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]


# ---------------------------------------------------------------- 1. fade by days ahead
print("1. how much of each signal persists, by days ahead (fit on train seasons)")
names = ["const", *A]
rows = []
co_d = {}
for i, (lo, hi) in enumerate(DB):
    t = TP[TP["season"].isin(H.TRAIN) & (TP["db"] == i)]
    c = fit_rows(t, A)
    co_d[i] = c
    rows.append({"days ahead": f"{lo}" if lo == hi else f"{lo}-{hi}", "n": len(t), "const": c[0], "points gap": c[names.index("fp_gap")],
                 "minutes gap": c[names.index("min_gap")], "DELCO slope": c[names.index("delco_lvl")]})
fade = pd.DataFrame(rows).set_index("days ahead")
fade["minutes, share of day 1"] = fade["minutes gap"] / fade["minutes gap"].iloc[0]
fade["points, share of day 1"] = fade["points gap"] / fade["points gap"].iloc[0]
print(fade.round(3).to_string())
OUT["fade_days"] = fade.round(4).reset_index().to_dict("records")
OUT["coef_days"] = {f"{lo}-{hi}": dict(zip(names, [round(float(x), 4) for x in co_d[i]])) for i, (lo, hi) in enumerate(DB)}

# level by days ahead: A fitted by days vs season DELCO vs v1 (is there still a gain at 11-14 days?)
te = TP[TP["season"].isin(H.TEST)].copy()
predA = np.full(len(te), np.nan)
for i in range(len(DB)):
    m = (te["db"] == i).to_numpy()
    c = co_d[i]
    v = P["delco"] + c[0] + sum(c[j + 1] * f.fillna(0.0) for j, f in enumerate(A.values()))
    predA[m] = te.loc[m, "info_id"].map(v).to_numpy()
te["A"] = predA
te["delco"] = te["info_id"].map(P["delco"])
te["v1"] = te["info_id"].map(H.baselines()["v1 (live since 10-26)"])
rm = lambda e: float(np.sqrt(np.nanmean(e ** 2)))
tab = te.groupby("db").apply(lambda d: pd.Series({"season DELCO": rm(d["fp"] - d["delco"]), "v1": rm(d["fp"] - d["v1"]), "A by days": rm(d["fp"] - d["A"]),
                                                   "A lean": float(np.nanmean(d["fp"] - d["A"]))}))
tab.index = [fade.index[int(i)] for i in tab.index]
tab["A gain vs DELCO"] = tab["A by days"] - tab["season DELCO"]
print("\ntypical miss by days ahead (test seasons):")
print(tab.round(3).to_string())
OUT["miss_days"] = tab.round(4).reset_index().to_dict("records")

# ---------------------------------------------------------------- 2. spread of outcomes
print("\n2. spread of a single game around the forecast (test seasons), by player level and days ahead")
te["res"] = te["fp"] - te["A"]
te["lvl"] = pd.cut(te["delco"], [0, 15, 25, 35, 45, 200], labels=["<15", "15-25", "25-35", "35-45", "45+"])
te["horizon"] = pd.cut(te["days_ahead"], [0, 3, 7, 14], labels=["1-3 days", "4-7 days", "8-14 days"])
sd = te.pivot_table(index="lvl", columns="horizon", values="res", aggfunc="std", observed=True)
cv = sd.div(te.groupby("lvl", observed=True)["A"].mean(), axis=0)
print("standard deviation (points):"); print(sd.round(2).to_string())
print("as a share of his level:"); print(cv.round(2).to_string())
q = te.groupby("lvl", observed=True)["res"].quantile([0.1, 0.9]).unstack()
print("10th / 90th percentile of (actual - forecast):"); print(q.round(1).to_string())
OUT["spread_sd"] = sd.round(3).to_dict()
OUT["spread_cv"] = cv.round(3).to_dict()

# ---------------------------------------------------------------- 3. 7-day total, A x chance model re-centred, by level
print("\n3. 7-day totals (A by days x chance model re-centred on last season)")
pR = pd.read_parquet(D / "st_phase4_avail_recentred.parquet")["p"]
t7 = T[T["season"].isin(H.TEST) & (T["days_ahead"] <= 7)].copy()
t7["db"] = t7["days_ahead"].map(lambda d: next((i for i, (lo, hi) in enumerate(DB) if lo <= d <= hi), np.nan))
lv = np.full(len(t7), np.nan)
for i in range(len(DB)):
    m = (t7["db"] == i).to_numpy()
    c = co_d[i]
    v = P["delco"] + c[0] + sum(c[j + 1] * f.fillna(0.0) for j, f in enumerate(A.values()))
    lv[m] = t7.loc[m, "info_id"].map(v).to_numpy()
t7["ev"] = lv * pR.reindex(t7.index).to_numpy()
t7["ev_flat"] = t7["info_id"].map(H.baselines()["v1 (live since 10-26)"]).to_numpy() * H.flat_avail().reindex(t7.index).to_numpy()
t7["act"] = t7["fp"].fillna(0.0)
w = t7.groupby("info_id").agg(act=("act", "sum"), ev=("ev", "sum"), ev_flat=("ev_flat", "sum"), n=("k", "size")).dropna()
w["lvl"] = pd.cut(w.index.map(P["delco"]), [0, 15, 25, 35, 45, 200], labels=["<15", "15-25", "25-35", "35-45", "45+"])
s7 = w.groupby("lvl", observed=True).apply(lambda d: pd.Series({"n": len(d), "actual avg": d["act"].mean(), "live today lean": (d["act"] - d["ev_flat"]).mean(),
                                                               "new lean": (d["act"] - d["ev"]).mean(), "new miss": rm(d["act"] - d["ev"]),
                                                               "spread (sd)": (d["act"] - d["ev"]).std()}))
print(s7.round(2).to_string())
print(f"all: live today miss {rm(w['act'] - w['ev_flat']):.2f} lean {(w['act'] - w['ev_flat']).mean():+.2f} | new miss {rm(w['act'] - w['ev']):.2f} lean {(w['act'] - w['ev']).mean():+.2f}")
OUT["week_by_level"] = s7.round(3).reset_index().to_dict("records")
json.dump(OUT, open(D / "st_phase5.json", "w"), indent=1, default=str)
