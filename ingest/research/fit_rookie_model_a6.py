"""Confirmatory LOCO-CV DE fits for the top interaction candidates surfaced
by screen_interactions.py -- does adding ONE interaction term to the
current champion (bpm, draft_age, fg_pct + capital) actually beat 0.5440,
or was the residual-screen correlation just noise?
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df, build_bounds, score, ALL_FEATURE_BOUNDS

BASE_FEATURES = ["bpm", "draft_age_filled", "fg_pct_filled"]
CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]

INTERACTION_PAIRS = [
    ("power_conf", "apg_c"),
    ("rec_filled", "apg_c"),
    ("power_conf", "ppg_c"),
    ("porpag", "power_conf"),
    ("LANE_AGILITY_TIME_PCTILE", "power_conf"),
    ("bpm", "three_pct_filled"),
    ("ortg", "draft_age_filled"),
]

for a, b in INTERACTION_PAIRS:
    xa = ALL_FEATURE_BOUNDS[a][0].to_numpy(dtype=float)
    xb = ALL_FEATURE_BOUNDS[b][0].to_numpy(dtype=float)
    za = (xa - np.nanmean(xa)) / (np.nanstd(xa) + 1e-9)
    zb = (xb - np.nanmean(xb)) / (np.nanstd(xb) + 1e-9)
    col = f"inter_{a}_x_{b}"
    df[col] = za * zb
    ALL_FEATURE_BOUNDS[col] = (df[col], ((-2, 2), (1, 6), (-15, 15)))


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def make_objective(features, n_pre):
    def objective_post(params, data):
        pre_part = score(params[:n_pre], data, features)
        cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
        pred = pre_part + cap_part
        if np.std(pred) < 1e-9:
            return 0.0
        rho, _ = spearmanr(pred, data["age_22_29_best3"])
        return -rho if np.isfinite(rho) else 0.0
    return objective_post


if __name__ == "__main__":
    classes = sorted(df["real_draft_year"].unique())
    baseline_folds = []
    for held_out in classes:
        test = df[df["real_draft_year"] == held_out]
        rho, _ = spearmanr(-test["pick_filled"], test["age_22_29_best3"])
        baseline_folds.append(rho)
    print(f"DRAFT-ALONE baseline: {np.mean(baseline_folds):.4f}")
    print(f"Current champion (bpm, age, fg_pct + capital): 0.5440\n")

    for a, b in INTERACTION_PAIRS:
        col = f"inter_{a}_x_{b}"
        features = BASE_FEATURES + [col]
        pre_bounds = build_bounds(features)
        n_pre = len(pre_bounds)
        bounds = pre_bounds + CAPITAL_BOUNDS
        objective_post = make_objective(features, n_pre)
        fold_rhos = []
        for held_out in classes:
            train = df[df["real_draft_year"] != held_out]
            test = df[df["real_draft_year"] == held_out]
            result = differential_evolution(
                objective_post, bounds, args=(train,), maxiter=80, popsize=20, seed=42, workers=1, tol=1e-6,
            )
            pre_part = score(result.x[:n_pre], test, features)
            cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
            pred = pre_part + cap_part
            rho, _ = spearmanr(pred, test["age_22_29_best3"])
            if np.isfinite(rho):
                fold_rhos.append(rho)
        print(f"base + inter({a} x {b}): POST-DRAFT LOCO-CV = {np.mean(fold_rhos):.4f}")
