"""Final Output A fit: winning pre-draft feature combo (bpm, rec, exp_inv,
agility, fg_pct) + draft-capital curve for the real post-draft comparison.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df, ALL_FEATURE_BOUNDS, build_bounds, score

ROOT = Path(__file__).resolve().parent
WINNING_FEATURES = ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "fg_pct_filled", "draft_age_filled"]
PRE_BOUNDS = build_bounds(WINNING_FEATURES)
CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def objective_post(params, data):
    n_pre = len(PRE_BOUNDS)
    pre_part = score(params[:n_pre], data, WINNING_FEATURES)
    cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
    pred = pre_part + cap_part
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


if __name__ == "__main__":
    bounds = PRE_BOUNDS + CAPITAL_BOUNDS
    classes = sorted(df["real_draft_year"].unique())
    fold_rhos = []
    for held_out in classes:
        train = df[df["real_draft_year"] != held_out]
        test = df[df["real_draft_year"] == held_out]
        result = differential_evolution(
            objective_post, bounds, args=(train,), maxiter=60, popsize=15, seed=42, workers=1, tol=1e-5,
        )
        n_pre = len(PRE_BOUNDS)
        pre_part = score(result.x[:n_pre], test, WINNING_FEATURES)
        cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
        pred = pre_part + cap_part
        rho, _ = spearmanr(pred, test["age_22_29_best3"])
        if np.isfinite(rho):
            fold_rhos.append(rho)
        print(f"  held out {held_out}: rho={rho:.4f} (n={len(test)})")

    print(f"\nPOST-DRAFT (winning combo + capital curve) LOCO-CV: {np.mean(fold_rhos):.4f}")

    baseline_folds = []
    for held_out in classes:
        test = df[df["real_draft_year"] == held_out]
        rho, _ = spearmanr(-test["pick_filled"], test["age_22_29_best3"])
        baseline_folds.append(rho)
    print(f"DRAFT-ALONE baseline: {np.mean(baseline_folds):.4f}")
