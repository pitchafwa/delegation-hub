"""Seed-stability check (robust settings) for the new best convex-target
combo found in combo_search_convex.py: rec_filled + porpag +
breakout_age_filled + draft_age_filled. Same discipline as before -- a
single-seed win means nothing until it holds across multiple seeds.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

sys.argv = [sys.argv[0], "vor_convex"]
from fit_rookie_model_v4 import loco_cv_post, baseline_ndcg

FEATURES = ["rec_filled", "porpag", "breakout_age_filled", "draft_age_filled"]

if __name__ == "__main__":
    print("target=vor_convex (robust: maxiter=150, popsize=30)")
    print(f"features={FEATURES}")
    print(f"DRAFT-ALONE baseline NDCG: {baseline_ndcg():.4f}")
    for seed in [42, 7, 123, 999]:
        g, folds = loco_cv_post(FEATURES, maxiter=150, popsize=30, seed=seed)
        print(f"  seed={seed:4d}  POST-DRAFT NDCG: {g:.4f}")
