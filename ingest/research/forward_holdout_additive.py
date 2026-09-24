"""Closes the last gap in tonight's search: the ORIGINAL additive-model
linear-target winner (talent_pctile + rec_filled + breakout_age_filled +
draft_age_filled, seed-stable across 4 seeds in LOCO-CV) was never tested
against a genuine forward holdout. Train it once on 2008-2019, fit its
params via the same differential_evolution search used throughout this
project, then score it on the untouched 2020-2023 classes.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

sys.argv = [sys.argv[0], "vor_linear"]
from fit_rookie_model_v4 import df, build_bounds, score, capital_curve, CAPITAL_BOUNDS, ndcg
import numpy as np
from scipy.optimize import differential_evolution

FEATURES = ["rec_filled", "breakout_age_filled", "exp_numeric", "draft_age_filled"]
SELECTION_YEARS = list(range(2008, 2020))
HOLDOUT_YEARS = list(range(2020, 2024))

train = df[df["real_draft_year"].isin(SELECTION_YEARS)]
test = df[df["real_draft_year"].isin(HOLDOUT_YEARS)]
print(f"train rows: {len(train)}  holdout rows: {len(test)}", flush=True)

pre_bounds = build_bounds(FEATURES)
n_pre = len(pre_bounds)
bounds = pre_bounds + CAPITAL_BOUNDS


def objective(params, data):
    pre_part = score(params[:n_pre], data, FEATURES)
    cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
    pred = pre_part + cap_part
    rel = data["vor_linear"].to_numpy(dtype=float)
    g = ndcg(pred, rel)
    return -g if g is not None else 0.0


result = differential_evolution(objective, bounds, args=(train,), maxiter=150, popsize=30, seed=42, workers=1, tol=1e-5)

pre_part = score(result.x[:n_pre], test, FEATURES)
cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
pred = pre_part + cap_part

scores, base_scores = [], []
for yr in HOLDOUT_YEARS:
    sub = test[test["real_draft_year"] == yr]
    if len(sub) < 5:
        continue
    idx = sub.index
    p = pred[test.index.get_indexer(idx)]
    rel = sub["vor_linear"].to_numpy(dtype=float)
    g = ndcg(p, rel)
    bg = ndcg(-sub["pick_filled"].to_numpy(dtype=float), rel)
    if g is not None:
        scores.append(g)
    if bg is not None:
        base_scores.append(bg)

print(f"DRAFT-ALONE baseline on holdout: {np.mean(base_scores):.4f}")
print(f"ADDITIVE MODEL (original linear winner) on holdout: {np.mean(scores):.4f}")
