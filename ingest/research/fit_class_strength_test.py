"""Does adding real, leave-one-out class-strength (lottery talent density,
or overall class talent density) to the post-draft model actually improve
LOCO-CV beyond pick number + individual talent alone?
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import build_bounds, score
import fit_rookie_model_a2 as a2

# merge the leave-one-out class-strength columns onto a2's own (already
# filtered/featured) working dataframe by real_draft_year -- LOO values are
# per-class, so a left-merge on year is exact and safe.
cs = pd.read_csv("data/rookie_model_dataset_with_class_strength.csv")
cs_by_year = cs.groupby("real_draft_year")[["class_strength_lottery_loo", "class_strength_overall_loo"]].mean().reset_index()
df = a2.df.merge(cs_by_year, on="real_draft_year", how="left")

a2.ALL_FEATURE_BOUNDS["class_strength_lottery_loo"] = (
    df["class_strength_lottery_loo"], ((35, 42), (1, 6), (-15, 15))
)
a2.ALL_FEATURE_BOUNDS["class_strength_overall_loo"] = (
    df["class_strength_overall_loo"], ((29, 38), (1, 8), (-15, 15))
)
a2.df = df  # so build_bounds/score's closures see the merged columns too

CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]
BASE = ["bpm", "draft_age_filled"]
FEATURE_SETS = {
    "champion_baseline": BASE,
    "champion_+lottery_strength": BASE + ["class_strength_lottery_loo"],
    "champion_+overall_strength": BASE + ["class_strength_overall_loo"],
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
