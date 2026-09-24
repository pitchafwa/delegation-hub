"""Layer A: 'Talent/Production Value' -- context-free multi-year projection.

Mirrors WRPI/RUPI's actual architecture (methodology doc §4, §5):
  - additive bounded/clipped components per feature, not a raw linear model
  - fit via global black-box optimization (differential evolution) directly
    against the real validation objective (Spearman rank correlation),
    not a proxy loss like MSE
  - validated via leave-one-season-out CV (the basketball analog of
    leave-one-class-out -- holding out an entire season, never a random
    row split, because rows in the same season share era/rule effects)
  - a hardcoded pick-alone-equivalent baseline every version must beat
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset_v2.csv")
TARGET = "TARGET_BEST2OF3"

# Fill trend features that don't exist for true rookies (HAS_PREV==0) with
# the current-season value -- "assume no trend info available" rather than
# feeding the optimizer a NaN.
for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
    base_col = col.replace("_PREV", "")
    df[col] = df[col].fillna(df[base_col])

# --- Component definitions: (name, column, param-count) ---
RAMP_FEATURES = [
    ("current_prod", "FANTASY_PPG"),
    ("prev_prod", "FANTASY_PPG_PREV"),
    ("usage", "USG_PCT"),
    ("efficiency", "TS_PCT"),
    ("health", "GP"),
    ("experience", "EXPERIENCE"),
]
# bounds per ramp: (lo_x_min, lo_x_max), (span_min, span_max), (weight_min, weight_max)
RAMP_BOUNDS = {
    "current_prod": ((0, 40), (5, 60), (0, 50)),
    "prev_prod": ((0, 40), (5, 60), (0, 30)),
    "usage": ((0.05, 0.30), (0.02, 0.30), (0, 20)),
    "efficiency": ((0.40, 0.65), (0.02, 0.25), (0, 20)),
    "health": ((0, 60), (5, 82), (0, 15)),
    "experience": ((0, 10), (1, 20), (-15, 15)),
}
N_RAMP_PARAMS = len(RAMP_FEATURES) * 3
# Age "tent" curve: peak_age, slope_up (>=0), slope_down (<=0)
AGE_BOUNDS = [(22, 32), (0, 4), (-4, 0)]
# Binary penalty terms: team changed in offseason, traded mid-season
PENALTY_BOUNDS = [(-15, 5), (-15, 5)]
INTERCEPT_BOUNDS = [(0, 30)]

ALL_BOUNDS = []
for name in RAMP_FEATURES:
    ALL_BOUNDS.extend(RAMP_BOUNDS[name[0]])
ALL_BOUNDS.extend(AGE_BOUNDS)
ALL_BOUNDS.extend(PENALTY_BOUNDS)
ALL_BOUNDS.extend(INTERCEPT_BOUNDS)
N_PARAMS = len(ALL_BOUNDS)


def unpack_and_score(params, data: pd.DataFrame) -> np.ndarray:
    idx = 0
    score = np.zeros(len(data))
    for name, col in RAMP_FEATURES:
        lo_x, span, weight = params[idx], params[idx + 1], params[idx + 2]
        idx += 3
        hi_x = lo_x + span
        x = data[col].to_numpy(dtype=float)
        frac = np.clip((x - lo_x) / max(hi_x - lo_x, 1e-6), 0.0, 1.0)
        score += weight * frac
    peak_age, slope_up, slope_down = params[idx], params[idx + 1], params[idx + 2]
    idx += 3
    age = data["AGE"].to_numpy(dtype=float)
    diff = age - peak_age
    age_contrib = np.where(diff <= 0, slope_up * diff, slope_down * diff)
    score += age_contrib
    team_changed_w, traded_w = params[idx], params[idx + 1]
    idx += 2
    score += team_changed_w * data["TEAM_CHANGED_OFFSEASON"].to_numpy(dtype=float)
    score += traded_w * data["TRADED_MIDSEASON"].to_numpy(dtype=float)
    intercept = params[idx]
    score += intercept
    return score


def neg_spearman(params, data: pd.DataFrame) -> float:
    pred = unpack_and_score(params, data)
    if np.std(pred) < 1e-9:
        return 0.0  # degenerate (constant prediction) -> worst-ish, avoid NaN
    rho, _ = spearmanr(pred, data[TARGET].to_numpy())
    return -rho if np.isfinite(rho) else 0.0


def fit(data: pd.DataFrame, maxiter=80, popsize=15, seed=42):
    result = differential_evolution(
        neg_spearman, ALL_BOUNDS, args=(data,), maxiter=maxiter, popsize=popsize,
        seed=seed, workers=1, polish=True, tol=1e-6,
    )
    return result.x, -result.fun


if __name__ == "__main__":
    t0 = time.time()
    print(f"Params to fit: {N_PARAMS}")
    print("=== Quick single fit on ALL data (sanity check before LOSO) ===")
    params, score = fit(df, maxiter=60, popsize=12)
    print(f"In-sample Spearman: {score:.4f}  (elapsed {time.time()-t0:.1f}s)")

    baseline_rho, _ = spearmanr(df["FANTASY_PPG"], df[TARGET])
    print(f"Baseline (current FANTASY_PPG alone) in-sample Spearman: {baseline_rho:.4f}")

    np.save(ROOT / "data" / "_quick_fit_params.npy", params)
