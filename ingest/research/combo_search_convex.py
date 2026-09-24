"""Wider combo search specifically for vor_convex, since the draft-order
baseline beat every combo tried so far under thorough optimization.
Tests several 3-5 feature combos not yet tried (not just the two built
from individual-feature rankings), including pairwise-interaction-style
combos mixing production and physical/timing features.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

sys.argv = [sys.argv[0], "vor_convex"]
from fit_rookie_model_v4 import loco_cv_post, baseline_ndcg

CANDIDATES = {
    "talent_rec_breakout": ["talent_pctile", "rec_filled", "breakout_age_filled"],
    "talent_breakout_draftage": ["talent_pctile", "breakout_age_filled", "draft_age_filled"],
    "talent_rec_draftage": ["talent_pctile", "rec_filled", "draft_age_filled"],
    "talent_porpag_breakout": ["talent_pctile", "porpag", "breakout_age_filled"],
    "talent_usg_breakout_draftage": ["talent_pctile", "usg", "breakout_age_filled", "draft_age_filled"],
    "talent_rec_breakout_usg_draftage": ["talent_pctile", "rec_filled", "breakout_age_filled", "usg", "draft_age_filled"],
    "talent_ts_breakout_draftage": ["talent_pctile", "ts", "breakout_age_filled", "draft_age_filled"],
    "rec_porpag_breakout_draftage": ["rec_filled", "porpag", "breakout_age_filled", "draft_age_filled"],
}

if __name__ == "__main__":
    print(f"DRAFT-ALONE baseline NDCG vs vor_convex: {baseline_ndcg():.4f}\n")
    for name, feats in CANDIDATES.items():
        g, folds = loco_cv_post(feats, maxiter=100, popsize=25, seed=42)
        print(f"{name:35s} NDCG: {g:.4f}  features={feats}")
