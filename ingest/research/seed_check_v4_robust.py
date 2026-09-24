"""Same seed-stability check as seed_check_v4.py, but with a more thorough
differential_evolution search (more iterations, bigger population) to test
whether the convex target's seed-to-seed instability is an optimizer
convergence problem rather than a real property of the model.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

TARGET_COL = sys.argv[1] if len(sys.argv) > 1 else "vor_convex"
sys.argv = [sys.argv[0], TARGET_COL]
from fit_rookie_model_v4 import loco_cv_post, baseline_ndcg

FEATURES = ["talent_pctile", "rec_filled", "breakout_age_filled", "draft_age_filled"]

if __name__ == "__main__":
    print(f"target={TARGET_COL} (robust: maxiter=150, popsize=30)")
    print(f"DRAFT-ALONE baseline NDCG: {baseline_ndcg():.4f}")
    for seed in [42, 7, 123, 999]:
        g, folds = loco_cv_post(FEATURES, maxiter=150, popsize=30, seed=seed)
        print(f"  seed={seed:4d}  POST-DRAFT NDCG: {g:.4f}")
