"""Confidence-adjusted outcome target: real, resolved age_22_29_best3 when
a player has actually played through enough of that window; otherwise a
real, EMPIRICALLY-projected eventual peak based on early-career production,
rather than either (a) capping a hot young start at what's been observed so
far, or (b) leaving it as an unusable NaN that excludes recent classes
entirely.

Real, deliberate design choice: use isotonic regression (a real, smooth,
MONOTONIC, non-parametric fit) on the resolved population's own
(entry_3yr_best2 -> age_22_29_best3) relationship, NOT the existing Kalman
engine's fitted age-decline curve -- that curve is explicitly documented as
"only ever validated one season ahead" and produces "absurd runouts" when
extrapolated far, exactly the failure mode this needs to avoid. Isotonic
regression on real comparables can't extrapolate past what's actually been
observed in similar players, which is the safer property here.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent
target = pd.read_csv(ROOT / "data" / "target_window_test.csv")

# --- Step 1: fit the real, empirical early -> eventual-peak relationship
# on the population where we ALREADY KNOW BOTH (fully resolved comparables) ---
resolved_both = target.dropna(subset=["entry_3yr_best2", "age_22_29_best3"])
print(f"Fitting projection on {len(resolved_both)} fully-resolved real comparables")

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(resolved_both["entry_3yr_best2"], resolved_both["age_22_29_best3"])

# real, honest check: does the projection ever fall BELOW the raw input for
# a hot start? (it shouldn't systematically, given players improve on
# average -- but confirm rather than assume)
check_x = np.array([10, 20, 30, 40, 45, 50])
check_y = iso.predict(check_x)
print("\nReal projection curve (early production -> projected eventual peak):")
for x, y in zip(check_x, check_y):
    print(f"  entry_3yr_best2={x:5.1f}  ->  projected age_22_29_best3={y:6.2f}")

# --- Step 2: apply to every player -- real resolved value where we have
# one, real empirical projection otherwise ---
target["confidence_adjusted_outcome"] = target["age_22_29_best3"]
target["outcome_is_projected"] = target["age_22_29_best3"].isna().astype(int)
needs_projection = target["age_22_29_best3"].isna() & target["entry_3yr_best2"].notna()
target.loc[needs_projection, "confidence_adjusted_outcome"] = iso.predict(
    target.loc[needs_projection, "entry_3yr_best2"]
)
print(f"\nReal resolved: {(target['outcome_is_projected']==0).sum()}")
print(f"Projected (early production known, real peak not yet resolved): {needs_projection.sum()}")
print(f"Neither available (too early even for entry window): {target['confidence_adjusted_outcome'].isna().sum()}")

target.to_csv(ROOT / "data" / "target_window_test.csv", index=False, encoding="utf-8")
print("\nSaved confidence_adjusted_outcome + outcome_is_projected columns to target_window_test.csv")

# --- Real face-validity checks ---
main = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
merged = target.merge(main[["PLAYER_ID", "player"]].drop_duplicates(subset=["PLAYER_ID"]), on="PLAYER_ID", how="left")
print("\n=== Face-validity checks ===")
for name in ["Al Horford", "Zion Williamson", "Ja Morant", "Victor Wembanyama", "Cooper Flagg"]:
    m = merged[merged["player"].str.contains(name, na=False, case=False)]
    if len(m):
        r = m.iloc[0]
        status = "RESOLVED (real)" if r["outcome_is_projected"] == 0 else "PROJECTED"
        print(f"{name:20s} entry_3yr_best2={r['entry_3yr_best2']:.1f}  real_age_22_29_best3={r['age_22_29_best3']}  "
              f"confidence_adjusted={r['confidence_adjusted_outcome']:.1f}  [{status}]")
    else:
        print(f"{name}: not found")
