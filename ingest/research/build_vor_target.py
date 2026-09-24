"""The real success metric: value over opportunity cost, not a flat
percentile. Below the real replacement bar, contributes ~zero (matches
"don't care about 20th vs 60th percentile" -- both are non-rosterable).
Above it, value counts, and a convex variant is tested to see if
"93rd-vs-97th matters way more than 75th-vs-79th" is something the data
itself supports, rather than assumed.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# real, already-fit opportunity cost curve from keeper_value_over_replacement.py
OPP_FLOOR, OPP_AMP, OPP_TAU = 14.84, 20.37, 19.96


def opportunity_cost(keepers_per_team):
    return OPP_FLOOR + OPP_AMP * np.exp(-keepers_per_team / OPP_TAU)


OPP_COST_K5 = opportunity_cost(5)  # the real, forward-looking keeper plan
print(f"Real opportunity cost at K=5 (Tommy's planned future keeper count): {OPP_COST_K5:.2f} fantasy PPG")

target = pd.read_csv(ROOT / "data" / "target_window_test.csv")
target["vor_linear"] = (target["confidence_adjusted_outcome"] - OPP_COST_K5).clip(lower=0)
target["vor_convex"] = target["vor_linear"] ** 1.5

print(f"\nvor_linear: base rate above 0 = {(target['vor_linear']>0).mean():.3f}")
print(target["vor_linear"].describe())

target.to_csv(ROOT / "data" / "target_window_test.csv", index=False, encoding="utf-8")
print("\nSaved vor_linear + vor_convex columns to target_window_test.csv")

main = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
merged = target.merge(main[["PLAYER_ID", "player"]].drop_duplicates(subset=["PLAYER_ID"]), on="PLAYER_ID", how="left")
print("\n=== Real face-validity: known tiers ===")
for name in ["Victor Wembanyama", "Anthony Davis", "DeJuan Blair", "Fab Melo", "Patty Mills"]:
    m = merged[merged["player"].str.contains(name, na=False, case=False)]
    if len(m):
        r = m.iloc[0]
        print(f"{name:20s} adjusted_outcome={r['confidence_adjusted_outcome']:.1f}  vor_linear={r['vor_linear']:.1f}  vor_convex={r['vor_convex']:.1f}")
