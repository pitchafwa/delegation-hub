"""Freeze the production Layer A ('Talent/Production Value') model.

Per WRPI/RUPI methodology §4.4: parameters are fit once, then committed as a
versioned JSON artifact -- deterministic, auditable, decoupled from the
fitting code. Refitting is a separate, rare, deliberate event, not something
that happens automatically on every data pull.

Uses a higher optimization budget than the per-fold LOSO fits (which only
need to be fast/comparable across 13 folds) since this is the one-time
artifact that actually ships.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fit_talent_model import fit, unpack_and_score, TARGET, RAMP_FEATURES

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset_v2.csv")
for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
    base_col = col.replace("_PREV", "")
    df[col] = df[col].fillna(df[base_col])

t0 = time.time()
params, spearman_insample = fit(df, maxiter=150, popsize=20, seed=42)
print(f"Final fit done in {time.time()-t0:.0f}s. In-sample Spearman: {spearman_insample:.4f}")

with open(ROOT / "data" / "loso_results.json") as f:
    loso = json.load(f)

component_names = [name for name, _ in RAMP_FEATURES] + ["age_peak", "age_slope_up", "age_slope_down",
                                                          "team_changed_penalty", "traded_penalty", "intercept"]
artifact = {
    "version": "1.0.0",
    "fit_date": "2026-09-18",
    "target_definition": "best-2-of-next-3-seasons fantasy PPG (league's real validated scoring formula)",
    "params_raw": params.tolist(),
    "component_order_note": "see fit_talent_model.py unpack_and_score() for exact param layout",
    "TALENT_MODEL_LOSO_SPEARMAN": loso["TALENT_MODEL_LOSO_SPEARMAN"],
    "BASELINE_LOSO_SPEARMAN": loso["BASELINE_LOSO_SPEARMAN"],
    "loso_fold_scores": loso["model_fold_scores"],
    "loso_fold_seasons": loso["fold_seasons"],
    "in_sample_spearman_full_fit": spearman_insample,
}
with open(ROOT / "data" / "layer_a_model.frozen.json", "w") as f:
    json.dump(artifact, f, indent=2)
print("Saved research/data/layer_a_model.frozen.json")
print(f"\nHardcoded baseline any future change must beat: "
      f"TALENT_MODEL_LOSO_SPEARMAN={loso['TALENT_MODEL_LOSO_SPEARMAN']:.4f} "
      f"vs BASELINE_LOSO_SPEARMAN={loso['BASELINE_LOSO_SPEARMAN']:.4f}")
