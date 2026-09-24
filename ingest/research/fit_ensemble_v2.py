import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import rankdata, spearmanr

from fit_rookie_model_v2 import df, build_bounds, score, capital_curve, CAPITAL_BOUNDS, TARGET_COL

MODELS = {
    "champion": ["talent_pctile", "draft_age_filled"],
    "breakout_variant": ["talent_pctile", "breakout_age_filled"],
    "full_combo": ["talent_pctile", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "draft_age_filled"],
    "rec_variant": ["talent_pctile", "draft_age_filled", "rec_filled"],
    "ts_variant": ["talent_pctile", "draft_age_filled", "ts"],
}


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
        rho, _ = spearmanr(pred, data[TARGET_COL])
        return -rho if np.isfinite(rho) else 0.0

    result = differential_evolution(objective, bounds, args=(train,), maxiter=80, popsize=20, seed=seed, workers=1, tol=1e-6)
    pre_part = score(result.x[:n_pre], test, features)
    cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
    return pre_part + cap_part


if __name__ == "__main__":
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
            rho, _ = spearmanr(pred, test[TARGET_COL])
            if np.isfinite(rho):
                single_fold_rhos[name].append(rho)
        ranks = np.mean([rankdata(preds[name]) for name in MODELS], axis=0)
        erho, _ = spearmanr(ranks, test[TARGET_COL])
        if np.isfinite(erho):
            ensemble_fold_rhos.append(erho)

    print(f"seed={seed}")
    for name in MODELS:
        print(f"  {name} solo: {np.mean(single_fold_rhos[name]):.4f}")
    print(f"  ENSEMBLE (rank-averaged): {np.mean(ensemble_fold_rhos):.4f}")
