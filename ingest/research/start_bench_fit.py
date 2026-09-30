"""Fit the production start/bench adjustment (Tommy, 2026-09-30): on a HEALTHY-roster night (no rotation teammate out -- see
start_bench_study_v2.py for why this half is kept separate from usage-flow), how many fantasy points should a player's projection move if
he's starting or benched relative to his OWN normal rate?

Model: dev = fp - b_fp (his actual production minus his own trailing "full-strength" baseline -- a rolling mean of his last 40 full-strength
games' fp, already computed by usage_flow_panel.py; NOT fp_full, which is just that same game's own raw fp on full-strength rows only and
identically equal to fp there -- using it as "the baseline" would make dev exactly 0 by construction, a real bug caught while building this)
regressed on `surprise = start - st15` (start is 1/0 tonight, st15 is his trailing share of games started -- so surprise is positive when he
starts MORE than his own normal rate, negative when he starts LESS). This naturally reproduces Part 1's finding that the swing is bigger for
a player who rarely starts (surprise close to +1) than for a mostly-starter who sits (surprise close to -1 is rare for him anyway) -- one
coefficient, scaled by how surprising tonight's role is for THIS player, rather than a flat "+8.5 if starting" applied to everyone alike.

Held out by season (leave-one-season-out), same discipline as matchup_fit_final.py.
Usage: uv run python research/start_bench_fit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"

d = pd.read_pickle(D / "usage_flow_panel.pkl")
d["start"] = d["start"].astype(bool)
d = d[(d.V_min == 0) & d.st15.notna() & d.b_fp.notna() & (d.full_n >= 5)].copy()
d["surprise"] = d.start.astype(float) - d.st15
d["dev"] = d.fp - d.b_fp
print(f"{len(d):,} healthy-roster player-games with a real baseline")
print(f"surprise range: min {d.surprise.min():.2f} max {d.surprise.max():.2f} mean {d.surprise.mean():.3f}")

X = d[["surprise"]].values
y = d.dev.values
m = LinearRegression().fit(X, y)
print(f"\nfull-sample fit: dev = {m.intercept_:.3f} + {m.coef_[0]:.3f} * surprise  (R^2 {m.score(X,y):.4f})")


def loso():
    errs_base, errs_fit = [], []
    for s in sorted(d.season_year.unique()):
        tr, te = d[d.season_year != s], d[d.season_year == s]
        if len(te) < 200:
            continue
        mm = LinearRegression().fit(tr[["surprise"]].values, tr.dev.values)
        pred = mm.predict(te[["surprise"]].values)
        errs_base.append(te.dev.values)
        errs_fit.append(te.dev.values - pred)
    b = np.concatenate(errs_base)
    f = np.concatenate(errs_fit)
    return np.sqrt(np.mean(b ** 2)), np.sqrt(np.mean(f ** 2))


rb, rf = loso()
print(f"held-out (leave-one-season-out) RMSE of fp deviation: baseline (no adjustment) {rb:.4f}  with surprise-adjustment {rf:.4f}")

# sanity check against Part 1's bucketed numbers: mean dev by rounded surprise bucket
d["bucket"] = pd.cut(d.surprise, [-1.01, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.01])
print("\nmean deviation by surprise bucket (should rise smoothly with surprise, ~0 near 0):")
print(d.groupby("bucket", observed=True).dev.agg(["mean", "count"]))

coef = {"intercept": float(m.intercept_), "surprise_coef": float(m.coef_[0]), "cap": 12.0}
import json
MODEL_PATH = Path(__file__).resolve().parent / "start_bench_model.json"   # lives beside matchup_model.json/usage_flow_model.json, NOT in data/
json.dump(coef, open(MODEL_PATH, "w"), indent=1)                          # (data/ is gitignored -- regenerable pulls only, not fitted models)
print(f"\nwrote {MODEL_PATH.name}:", coef)
