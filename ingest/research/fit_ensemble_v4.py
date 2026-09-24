"""NDCG-based ensemble: rank-average several distinct post-draft feature
combos (each fit independently per LOCO-CV fold), evaluate the blended
ranking's own NDCG. Ensembling has been the strongest single lever
throughout this whole project (Spearman-era and NDCG-era alike) -- this is
the NDCG-era version of that same test.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import rankdata

TARGET_COL = sys.argv[1] if len(sys.argv) > 1 else "vor_convex"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

# fit_rookie_model_v4 reads its TARGET_COL from sys.argv[1] at import time --
# spoof argv before importing so the shared df/score/capital_curve/ndcg
# helpers pick up the right target column (its own __main__ block won't run
# on import, so this is safe).
sys.argv = [sys.argv[0], TARGET_COL]
from fit_rookie_model_v4 import df, build_bounds, score, capital_curve, CAPITAL_BOUNDS, ndcg

MODELS = {
    "champion": ["talent_pctile", "draft_age_filled"],
    "convex_combo": ["talent_pctile", "rec_filled", "breakout_age_filled", "draft_age_filled"],
    "linear_combo": ["rec_filled", "breakout_age_filled", "exp_numeric", "draft_age_filled"],
    "breakout_variant": ["talent_pctile", "breakout_age_filled"],
    "full_combo": ["talent_pctile", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "draft_age_filled"],
}


def fit_and_predict(features, train, test, seed):
    pre_bounds = build_bounds(features)
    n_pre = len(pre_bounds)
    bounds = pre_bounds + CAPITAL_BOUNDS

    def objective(params, data):
        pre_part = score(params[:n_pre], data, features)
        cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
        pred = pre_part + cap_part
        rel = data[TARGET_COL].to_numpy(dtype=float)
        g = ndcg(pred, rel)
        return -g if g is not None else 0.0

    result = differential_evolution(
        objective, bounds, args=(train,), maxiter=80, popsize=20, seed=seed, workers=1, tol=1e-5,
    )
    pre_part = score(result.x[:n_pre], test, features)
    cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
    return pre_part + cap_part


if __name__ == "__main__":
    classes = sorted(df["real_draft_year"].unique())
    single_fold = {name: [] for name in MODELS}
    ensemble_fold = []
    for held_out in classes:
        train = df[df["real_draft_year"] != held_out]
        test = df[df["real_draft_year"] == held_out]
        if len(test) < 10:
            continue
        rel = test[TARGET_COL].to_numpy(dtype=float)
        preds = {}
        for name, features in MODELS.items():
            pred = fit_and_predict(features, train, test, seed)
            preds[name] = pred
            g = ndcg(pred, rel)
            if g is not None:
                single_fold[name].append(g)
        ranks = np.mean([rankdata(preds[name]) for name in MODELS], axis=0)
        eg = ndcg(ranks, rel)
        if eg is not None:
            ensemble_fold.append(eg)

    print(f"target={TARGET_COL} seed={seed}")
    for name in MODELS:
        print(f"  {name} solo: {np.mean(single_fold[name]):.4f}")
    print(f"  ENSEMBLE (rank-averaged): {np.mean(ensemble_fold):.4f}")
