"""Does applying the fitted availability discount to the composite's
trajectory actually improve real predictive accuracy on the FULL outcome
(including real washout seasons, not filtered to GP>=20)? Test against the
same pooled multi-season data used to discover the effect, comparing
with-discount vs without-discount total-season-value predictions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent

# Reuse pooled_residuals.py's per-game-rate predictions (already computed,
# season-ahead per-game projections) and injury_availability_test.py's real
# outcome data (total season value including washouts).
per_game_pred = pd.read_csv(ROOT / "data" / "pooled_residuals.csv")  # predicted, actual (per-game rate), pred_source_year
injury = pd.read_csv(ROOT / "data" / "injury_availability_test.csv")  # injury_history, next_gp, next_fantasy_total

AVAIL_A, AVAIL_B, AVAIL_C = 0.88420392, 0.71142792, 0.68520518


def availability_fraction(injury_history):
    return np.clip(AVAIL_A - AVAIL_B * injury_history ** AVAIL_C, 0.05, 1.0)


# per_game_pred's "predicted" is a per-game rate; convert to a total-season
# prediction ASSUMING a full healthy season (82 games) -- matching what the
# current model implicitly assumes with no availability adjustment.
merged = per_game_pred.merge(
    injury[["PLAYER_ID", "pred_source_year", "injury_history", "next_gp", "next_fantasy_total"]],
    on=["PLAYER_ID", "pred_source_year"], how="inner",
)
print(f"Matched rows: {len(merged)}")

merged["pred_total_no_discount"] = merged["predicted"] * 82
merged["availability"] = availability_fraction(merged["injury_history"])
merged["pred_total_with_discount"] = merged["pred_total_no_discount"] * merged["availability"]
actual_total = merged["next_fantasy_total"]


def report(col, label):
    rho, _ = spearmanr(merged[col], actual_total)
    r, _ = pearsonr(merged[col], actual_total)
    mae = float(np.mean(np.abs(merged[col] - actual_total)))
    print(f"{label:40s} Spearman={rho:.4f}  Pearson r={r:.4f}  MAE={mae:.1f}")


print("\n=== Predicting TOTAL season value (including real washout seasons) ===")
report("pred_total_no_discount", "No availability discount (current model)")
report("pred_total_with_discount", "WITH injury-history availability discount")

merged.to_csv(ROOT / "data" / "availability_discount_validation.csv", index=False)
