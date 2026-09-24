"""Ceiling calibration for incoming prospects (top picks).

Measured (ceiling_calibration_backtest.py): for past top-10 picks the hub-style trajectory (Output B rookie
prior aged forward) undershoots what they actually produced, growing with time (top-3 picks: +3 pts/g in
the rookie year, +8 by year 3). Prospects have no NBA data for the Kalman filter to correct the level, so
this is a genuine prior bias -- unlike players with real data, where the original curve held up.

Fit: bias_k(pick) = a_k + b_k * ln(pick) for k = 0..4 seasons after the draft, from past picks 1-15 who
played a >=20 GP season that year (presence is 88-98% for these picks, so survivorship is small).
Beyond k=4 the k=4 bias is held. Taper to zero for picks 11-16+ (late picks: presence drops below 70%,
so conditional-on-playing growth would overstate value). Leave-one-draft-class-out validation reported.

Writes data/prospect_calibration.json used by build_hub_data.py.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
MAXK, PICKS = 4, 15

pt = pd.read_csv(D / "prospect_trajectories.csv")
pt["trajectory"] = pt["trajectory"].apply(ast.literal_eval)
u = pd.read_csv(D / "rookie_model_dataset_unified.csv")[["PLAYER_ID", "real_draft_number"]].drop_duplicates("PLAYER_ID")
pt = pt.merge(u, on="PLAYER_ID", how="left")
pt["pick"] = pt["real_draft_number"].fillna(61)
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
             - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
act = sb[sb["GP"] >= 20].set_index(["PLAYER_ID", "yr"])["fpg"]

rows = []
for r in pt.itertuples():
    if not (2010 <= r.real_draft_year <= 2021) or r.pick > PICKS:
        continue
    for k in range(MAXK + 1):
        a = act.get((r.PLAYER_ID, int(r.real_draft_year) + k), np.nan)
        if np.isfinite(a):
            rows.append(dict(cls=int(r.real_draft_year), pick=r.pick, k=k, err=a - r.trajectory[k], pred=r.trajectory[k], actual=a))
df = pd.DataFrame(rows)
df["lp"] = np.log(df["pick"])
print(f"training rows: {len(df)} (past picks 1-{PICKS}, classes 2010-2021)")


def design(sub):
    # the #1 overall pick is its own tier (held-out: without this term #1 picks are under-projected by ~5-7 pts/g
    # and #2-3 slightly over-projected)
    return np.column_stack([np.ones(len(sub)), sub["lp"], (sub["pick"] == 1).astype(float)])


def fit_k(sub):
    return np.linalg.lstsq(design(sub), sub["err"].to_numpy(), rcond=None)[0]


coef = {k: fit_k(df[df["k"] == k]) for k in range(MAXK + 1)}
print("bias(k, pick) = a + b*ln(pick) + c*[pick==1]:")
for k, (a, b, c) in coef.items():
    print(f"  k={k}: a={a:+.2f} b={b:+.2f} c={c:+.2f}   -> pick 1: {a + c:+.1f}, pick 2: {a + b * np.log(2):+.1f}, pick 3: {a + b * np.log(3):+.1f}, pick 8: {a + b * np.log(8):+.1f}, pick 15: {a + b * np.log(15):+.1f}")

# leave-one-class-out validation
print("\nleave-one-class-out (fit on other classes, correct the held-out class): RMSE of actual - forecast")
print(f"{'k':>2s} {'n':>4s} {'uncalibrated':>13s} {'calibrated':>11s} {'bias before':>12s} {'bias after':>11s}")
val = {}
for k in range(MAXK + 1):
    sub = df[df["k"] == k]
    err_new = np.full(len(sub), np.nan)
    for c in sub["cls"].unique():
        tr, te = sub[sub["cls"] != c], sub[sub["cls"] == c]
        err_new[(sub["cls"] == c).to_numpy()] = te["err"].to_numpy() - design(te) @ fit_k(tr)
    r0, r1 = float(np.sqrt((sub["err"] ** 2).mean())), float(np.sqrt(np.nanmean(err_new ** 2)))
    val[k] = {"n": int(len(sub)), "rmse_before": r0, "rmse_after": r1, "bias_before": float(sub["err"].mean()), "bias_after": float(np.nanmean(err_new))}
    print(f"{k:2d} {len(sub):4d} {r0:13.2f} {r1:11.2f} {val[k]['bias_before']:+12.2f} {val[k]['bias_after']:+11.2f}")

out = {"max_k": MAXK, "full_strength_through_pick": 10, "zero_at_pick": 16, "coef": {str(k): [float(x) for x in v] for k, v in coef.items()},
       "loco_validation": {str(k): v for k, v in val.items()}, "training_rows": int(len(df))}
(D / "prospect_calibration.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
