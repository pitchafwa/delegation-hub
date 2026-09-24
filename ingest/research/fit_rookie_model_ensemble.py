"""Last lever: ensemble the champion with the next-best distinct combos.
Averaging predictions from models that made different real mistakes can
beat any single model, even when none of the alternates beat champion solo.
"""
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df, build_bounds, score

CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]
MODELS = {
    "champion": ["bpm", "draft_age_filled"],
    "rec_variant": ["bpm", "draft_age_filled", "rec_filled"],
    "breakout_variant": ["bpm", "breakout_age_filled"],
}


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def fit_and_predict(features, train, test, seed):
    pre_bounds = build_bounds(features)
    n_pre = len(pre_bounds)
    bounds = pre_bounds + CAPITAL_BOUNDS

    def objective(params, data):
        pre_part = score(params[:n_pre], data, features)
        cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
        pred = pre_part + cap_part
        if np.std(pred) < 1e-9:
            return 0.0
        rho, _ = spearmanr(pred, data["age_22_29_best3"])
        return -rho if np.isfinite(rho) else 0.0

    result = differential_evolution(objective, bounds, args=(train,), maxiter=80, popsize=20, seed=seed, workers=1, tol=1e-6)
    pre_part = score(result.x[:n_pre], test, features)
    cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
    return pre_part + cap_part


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    classes = sorted(df["real_draft_year"].unique())
    single_fold_rhos = {name: [] for name in MODELS}
    ensemble_fold_rhos = []

    for held_out in classes:
        train = df[df["real_draft_year"] != held_out]
        test = df[df["real_draft_year"] == held_out]
        preds = {}
        for name, features in MODELS.items():
            pred = fit_and_predict(features, train, test, seed)
            preds[name] = pred
            rho, _ = spearmanr(pred, test["age_22_29_best3"])
            if np.isfinite(rho):
                single_fold_rhos[name].append(rho)

        # ensemble: average rank-normalized predictions
        from scipy.stats import rankdata
        ranks = np.mean([rankdata(preds[name]) for name in MODELS], axis=0)
        erho, _ = spearmanr(ranks, test["age_22_29_best3"])
        if np.isfinite(erho):
            ensemble_fold_rhos.append(erho)

    print(f"seed={seed}")
    for name in MODELS:
        print(f"  {name} solo: {np.mean(single_fold_rhos[name]):.4f}")
    print(f"  ENSEMBLE (rank-averaged): {np.mean(ensemble_fold_rhos):.4f}")
