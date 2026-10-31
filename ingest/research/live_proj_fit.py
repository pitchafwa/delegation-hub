"""Live in-game final projection: fit and grade (2026-10-31). Reads the replayed snapshots from live_proj_study.py (data/_live_proj_rows.pkl).
Projected final = points so far + expected remaining minutes x scoring rate, where
  rate = w x (tonight's points per minute, capped at CAP x his pregame rate) + (1 - w) x pregame rate,  w = minutes so far / (minutes so far + K)
  remaining minutes = a regression fitted on the training games (pregame minutes left, his minutes pace tonight, score margin x time, fouls)
Graded on games held out of every fit (30% of games), against "freeze" (points so far + pregame x share of game left: what "still to come" does now)
and naive pace. Writes research/live_proj_model.json (the fitted constants) for live_matchup.py.
Run from ingest/:  .venv/Scripts/python research/live_proj_fit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
R = pd.read_pickle(D / "_live_proj_rows.pkl")
R = R[(R.preMIN > 0) & (R.pre > 0) & (R.t < 48)].copy()
import os
if os.environ.get("LP_MINUTES") == "m10":            # test the live tracker's version: pregame minutes = average of his last 10 games, rate = pregame points / those minutes
    R = R[R.m10 > 0].copy()
    R["preMIN"] = R["m10"]
    print("pregame minutes = last-10-game average")
R["e"] = R.t / 48.0
R["rem_true"] = (R.ymin - R.mn).clip(lower=0)
games = np.array(sorted(R.game.unique()))
rng = np.random.default_rng(11)
test_games = set(rng.choice(games, size=int(0.3 * len(games)), replace=False))
R["test"] = R.game.isin(test_games)


def feats(df):
    e = df.e.to_numpy()
    pace_min = df.mn.to_numpy() / np.maximum(e, 1 / 48)
    am = np.abs(df.marg.to_numpy())
    pf = df.pf.to_numpy()
    return np.column_stack([df.preMIN * (1 - e), pace_min * (1 - e), am * e, am * e * e, np.maximum(am - 15, 0) * e,
                            pf * (1 - e), (pf >= 5) * (1 - e), (pf >= 4) * (1 - e) * (e < 0.75), df.mn * (1 - e)])


tr, te = R[~R.test], R[R.test]
# remaining minutes, fitted separately by game phase (the weights change through the game)
PH = [0, 12, 24, 36, 48]
mins_models = {}
for a, b in zip(PH[:-1], PH[1:]):
    s = tr[(tr.t >= a) & (tr.t < b)]
    mins_models[(a, b)] = LinearRegression().fit(feats(s), s.rem_true)


def rem_min(df):
    out = np.zeros(len(df))
    for (a, b), m in mins_models.items():
        k = ((df.t >= a) & (df.t < b)).to_numpy()
        if k.any():
            out[k] = m.predict(feats(df[k]))
    return np.clip(out, 0, (48 - df.t.to_numpy()) + 0.0)


def model(df, K, CAP, rem=None):
    rp = df.pre / df.preMIN
    rg = np.where(df.mn > 0, df.fp / np.maximum(df.mn, 1e-9), rp)
    rg = np.clip(rg, rp / CAP if CAP else -np.inf, rp * CAP if CAP else np.inf)
    w = df.mn / (df.mn + K)
    rate = w * rg + (1 - w) * rp
    r = rem_min(df) if rem is None else rem
    return df.fp + r * rate


mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
best = None
rm_tr = rem_min(tr)
for K in (4, 8, 12, 16, 20, 30, 45, 70, 120, 250, 1e9):
    for CAP in (1.5, 2.0, 3.0, None):
        m = mae(tr.y, model(tr, K, CAP, rm_tr))
        if best is None or m < best[0]:
            best = (m, K, CAP)
_, K, CAP = best
print(f"fitted on {tr.game.nunique()} games: K = {K} minutes (tonight's rate gets half the weight after {K} minutes), cap = {CAP}x his pregame rate")

te = te.copy()
te["freeze"] = te.fp + te.pre * (1 - te.e)
te["pace"] = te.fp / np.maximum(te.e, 1 / 48)
te["model"] = model(te, K, CAP)
te["model_premin"] = model(te, K, CAP, rem=np.clip(te.preMIN * (1 - te.e), 0, None))
print(f"\nheld-out {te.game.nunique()} games, {len(te)} snapshots. Mean absolute error of the projected final (fantasy points):")
print(f"{'game time':>12} {'n':>6} {'freeze (now)':>13} {'naive pace':>11} {'model':>7} {'model, pregame minutes':>23}")
for a, b in ((0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 42), (42, 48)):
    s = te[(te.t >= a) & (te.t < b)]
    print(f"{f'{a}-{b} min':>12} {len(s):6d} {mae(s.y, s.freeze):13.2f} {mae(s.y, s.pace):11.2f} {mae(s.y, s.model):7.2f} {mae(s.y, s.model_premin):23.2f}")
print(f"{'all':>12} {len(te):6d} {mae(te.y, te.freeze):13.2f} {mae(te.y, te.pace):11.2f} {mae(te.y, te.model):7.2f} {mae(te.y, te.model_premin):23.2f}")
print("\nbias (actual - projected), model:", round(float((te.y - te.model).mean()), 2), "| freeze:", round(float((te.y - te.freeze).mean()), 2))
hot = te[(te.t < 12) & (te.fp > 2.5 * te.pre * te.e) & (te.fp >= 15)]
print(f"early hot starts (first quarter, 2.5x+ his pregame pace, 15+ points; n={len(hot)}): actual final {hot.y.mean():.1f}, naive pace {hot.pace.mean():.1f}, model {hot.model.mean():.1f}, freeze {hot.freeze.mean():.1f}")
cold = te[(te.t >= 18) & (te.t < 30) & (te.fp < 0.4 * te.pre * te.e)]
print(f"cold halves (18-30 min, under 0.4x pace; n={len(cold)}): actual final {cold.y.mean():.1f}, model {cold.model.mean():.1f}, freeze {cold.freeze.mean():.1f}")

out = {"K": K, "CAP": CAP, "phases": [[a, b] for (a, b) in mins_models],
       "rem_min": {f"{a}-{b}": {"coef": m.coef_.round(5).tolist(), "intercept": round(float(m.intercept_), 4)} for (a, b), m in mins_models.items()},
       "features": ["preMIN*(1-e)", "minutes pace*(1-e)", "|margin|*e", "|margin|*e^2", "max(|margin|-15,0)*e", "fouls*(1-e)", "5+ fouls*(1-e)", "4+ fouls before Q4*(1-e)", "minutes so far*(1-e)"],
       "graded": {"games": int(te.game.nunique()), "mae_model": round(mae(te.y, te.model), 2), "mae_freeze": round(mae(te.y, te.freeze), 2), "mae_pace": round(mae(te.y, te.pace), 2)}}
(Path(__file__).resolve().parent / "live_proj_model.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
