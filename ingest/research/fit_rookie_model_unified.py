"""Pre-draft and post-draft model fitting on the UNIFIED (college +
international) rookie dataset. Mirrors fit_rookie_model_a2.py's structure
exactly, with "talent_pctile" (percentile-ranked BPM for college,
percentile-ranked Game Score for international -- see merge_international.py
for why raw units can't be mixed) standing in for "bpm".
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
df = df.dropna(subset=["age_22_29_best3"]).copy()
df = df[df["real_draft_year"] <= 2018].copy()
print(f"Unified trainable rows: {len(df)} (college: {(df['data_source']=='college').sum()}, "
      f"international: {(df['data_source']=='international').sum()})")

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["LANE_AGILITY_TIME_PCTILE"] = df["LANE_AGILITY_TIME_PCTILE"].fillna(df["LANE_AGILITY_TIME_PCTILE"].median())
df["breakout_age_filled"] = df["breakout_age"].fillna(df["breakout_age"].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["talent_pctile"] = df["talent_pctile"].fillna(df["talent_pctile"].median())

ALL_FEATURE_BOUNDS = {
    "talent_pctile": (df["talent_pctile"], ((0, 0.6), (0.1, 1.0), (-10, 20))),
    "rec_filled": (df["rec_filled"], ((0, 60), (10, 100), (0, 20))),
    "exp_numeric": (df["exp_numeric"], ((1, 3), (0.5, 3), (-20, 5))),
    "LANE_AGILITY_TIME_PCTILE": (df["LANE_AGILITY_TIME_PCTILE"], ((0, 0.6), (0.1, 1.0), (-10, 15))),
    "draft_age_filled": (df["draft_age_filled"], ((18, 23), (0.5, 5), (-20, 5))),
    "breakout_age_filled": (df["breakout_age_filled"], ((18, 22), (0.5, 8), (-20, 5))),
}


def build_bounds(feature_list):
    bounds = []
    for f in feature_list:
        bounds.extend(ALL_FEATURE_BOUNDS[f][1])
    bounds.append((0, 40))
    return bounds


def score(params, data, feature_list):
    idx = 0
    total = np.zeros(len(data))
    for f in feature_list:
        lo_x, span, weight = params[idx], params[idx + 1], params[idx + 2]
        idx += 3
        hi_x = lo_x + span
        x = data[f].to_numpy(dtype=float)
        frac = np.clip((x - lo_x) / max(hi_x - lo_x, 1e-6), 0.0, 1.0)
        total += weight * frac
    total += params[idx]
    return total


def objective(params, data, feature_list):
    pred = score(params, data, feature_list)
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


def loco_cv(feature_list, maxiter=80, popsize=20, seed=42):
    bounds = build_bounds(feature_list)
    classes = sorted(df["real_draft_year"].unique())
    fold_rhos = []
    for held_out in classes:
        train = df[df["real_draft_year"] != held_out]
        test = df[df["real_draft_year"] == held_out]
        result = differential_evolution(
            objective, bounds, args=(train, feature_list), maxiter=maxiter, popsize=popsize,
            seed=seed, workers=1, tol=1e-5,
        )
        pred = score(result.x, test, feature_list)
        rho, _ = spearmanr(pred, test["age_22_29_best3"])
        if np.isfinite(rho):
            fold_rhos.append(rho)
    return np.mean(fold_rhos), fold_rhos


CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def objective_post(params, data, feature_list, n_pre):
    pre_part = score(params[:n_pre], data, feature_list)
    cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
    pred = pre_part + cap_part
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


def loco_cv_post(feature_list, maxiter=80, popsize=20, seed=42):
    pre_bounds = build_bounds(feature_list)
    n_pre = len(pre_bounds)
    bounds = pre_bounds + CAPITAL_BOUNDS
    classes = sorted(df["real_draft_year"].unique())
    fold_rhos = []
    for held_out in classes:
        train = df[df["real_draft_year"] != held_out]
        test = df[df["real_draft_year"] == held_out]
        result = differential_evolution(
            objective_post, bounds, args=(train, feature_list, n_pre), maxiter=maxiter, popsize=popsize,
            seed=seed, workers=1, tol=1e-5,
        )
        pre_part = score(result.x[:n_pre], test, feature_list)
        cap_part = capital_curve(result.x[n_pre:], test["pick_filled"].to_numpy(dtype=float))
        pred = pre_part + cap_part
        rho, _ = spearmanr(pred, test["age_22_29_best3"])
        if np.isfinite(rho):
            fold_rhos.append(rho)
    return np.mean(fold_rhos), fold_rhos


if __name__ == "__main__":
    import sys as _sys
    mode = _sys.argv[1] if len(_sys.argv) > 1 else "pre"
    feature_list = _sys.argv[2].split(",") if len(_sys.argv) > 2 else ["talent_pctile", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "draft_age_filled"]
    print(f"Testing ({mode}): {feature_list}")
    if mode == "pre":
        rho, folds = loco_cv(feature_list)
        print(f"PRE-DRAFT unified LOCO-CV: {rho:.4f}")
    else:
        rho, folds = loco_cv_post(feature_list)
        print(f"POST-DRAFT unified LOCO-CV: {rho:.4f}")
        baseline_folds = []
        for y in sorted(df["real_draft_year"].unique()):
            test = df[df["real_draft_year"] == y]
            r, _ = spearmanr(-test["pick_filled"], test["age_22_29_best3"])
            baseline_folds.append(r)
        print(f"DRAFT-ALONE baseline (unified pool): {np.mean(baseline_folds):.4f}")
