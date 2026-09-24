"""Seed-stability check for the winning combo (talent_pctile, rec_filled,
breakout_age_filled, draft_age_filled) + draft capital curve, under NDCG,
for a given target. A real edge over baseline should hold up across
multiple random seeds, not just the one (42) used to find it.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

TARGET_COL = sys.argv[1] if len(sys.argv) > 1 else "vor_convex"
sys.argv = [sys.argv[0], TARGET_COL]
from fit_rookie_model_v4 import loco_cv_post, baseline_ndcg

FEATURES = ["talent_pctile", "rec_filled", "breakout_age_filled", "draft_age_filled"]

if __name__ == "__main__":
    print(f"target={TARGET_COL}")
    print(f"DRAFT-ALONE baseline NDCG: {baseline_ndcg():.4f}")
    for seed in [42, 7, 123, 999]:
        g, folds = loco_cv_post(FEATURES, seed=seed)
        print(f"  seed={seed:4d}  POST-DRAFT NDCG: {g:.4f}")
