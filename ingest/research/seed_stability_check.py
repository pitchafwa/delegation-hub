"""Is the 0.544 post-draft result (bpm, draft_age, fg_pct + capital) real,
or an artifact of a single DE random seed? Refit across several seeds.
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df, build_bounds, score

FEATURES = ["bpm", "draft_age_filled", "fg_pct_filled"]
CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def objective_post(params, data, n_pre):
    pre_part = score(params[:n_pre], data, FEATURES)
    cap_part = capital_curve(params[n_pre:], data["real_draft_number"].to_numpy(dtype=float))
    pred = pre_part + cap_part
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


if __name__ == "__main__":
    pre_bounds = build_bounds(FEATURES)
    n_pre = len(pre_bounds)
    bounds = pre_bounds + CAPITAL_BOUNDS
    classes = sorted(df["real_draft_year"].unique())

    for seed in [7, 42, 123, 999, 2024]:
        fold_rhos = []
        for held_out in classes:
            train = df[df["real_draft_year"] != held_out]
            test = df[df["real_draft_year"] == held_out]
            result = differential_evolution(
                objective_post, bounds, args=(train, n_pre), maxiter=80, popsize=20, seed=seed, workers=1, tol=1e-6,
            )
            pre_part = score(result.x[:n_pre], test, FEATURES)
            cap_part = capital_curve(result.x[n_pre:], test["real_draft_number"].to_numpy(dtype=float))
            pred = pre_part + cap_part
            rho, _ = spearmanr(pred, test["age_22_29_best3"])
            if np.isfinite(rho):
                fold_rhos.append(rho)
        print(f"seed={seed}: POST-DRAFT LOCO-CV = {np.mean(fold_rhos):.4f}")
