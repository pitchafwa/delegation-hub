"""Post-bugfix retest: breakout-age ensemble component, with and without
the (now-corrected, previously-inert) FG% feature.
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df, build_bounds, score

CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]
FEATURE_SETS = {
    "bpm_breakout_fg": ["bpm", "breakout_age_filled", "fg_pct_filled"],
    "bpm_breakout": ["bpm", "breakout_age_filled"],
}


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
    for name, features in FEATURE_SETS.items():
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
        print(f"{name}: POST-DRAFT LOCO-CV = {np.mean(fold_rhos):.4f}")
