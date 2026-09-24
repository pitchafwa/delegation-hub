"""Leave-one-SEASON-out CV for the Layer A talent model -- the basketball
analog of WRPI/RUPI's leave-one-class-out. For each season, fit on every
OTHER season, score the held-out season, average across all folds. This is
the real, honest validation number -- never a random row split.

Also computes and freezes the pick-alone-equivalent baseline the model must
beat: predicting the multi-year target from current FANTASY_PPG alone.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fit_talent_model import fit, unpack_and_score, TARGET, ALL_BOUNDS, N_PARAMS

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset_v2.csv")
for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
    base_col = col.replace("_PREV", "")
    df[col] = df[col].fillna(df[base_col])

seasons = sorted(df["SEASON_YEAR"].unique())
print(f"Seasons (folds): {seasons}")

fold_scores = []
baseline_fold_scores = []
t0 = time.time()
for held_out in seasons:
    train = df[df["SEASON_YEAR"] != held_out]
    test = df[df["SEASON_YEAR"] == held_out]
    if len(test) < 20:
        print(f"  season {held_out}: too few rows ({len(test)}), skipping")
        continue

    params, _ = fit(train, maxiter=60, popsize=12, seed=42)
    pred = unpack_and_score(params, test)
    rho, _ = spearmanr(pred, test[TARGET])
    fold_scores.append(rho)

    baseline_rho, _ = spearmanr(test["FANTASY_PPG"], test[TARGET])
    baseline_fold_scores.append(baseline_rho)

    elapsed = time.time() - t0
    print(f"  season {held_out}: model rho={rho:.4f}  baseline rho={baseline_rho:.4f}  "
          f"(n={len(test)}, elapsed {elapsed:.0f}s)")

model_loso = float(np.mean(fold_scores))
baseline_loso = float(np.mean(baseline_fold_scores))
print(f"\n=== FROZEN RESULT ===")
print(f"TALENT_MODEL_LOSO_SPEARMAN = {model_loso:.4f}")
print(f"BASELINE_LOSO_SPEARMAN (current PPG alone) = {baseline_loso:.4f}")
print(f"Lift: {model_loso - baseline_loso:+.4f}")

with open(ROOT / "data" / "loso_results.json", "w") as f:
    json.dump({
        "fold_seasons": [int(s) for s in seasons],
        "model_fold_scores": [float(s) for s in fold_scores],
        "baseline_fold_scores": [float(s) for s in baseline_fold_scores],
        "TALENT_MODEL_LOSO_SPEARMAN": model_loso,
        "BASELINE_LOSO_SPEARMAN": baseline_loso,
    }, f, indent=2)
print("Saved research/data/loso_results.json")
