"""Pre-draft / post-draft model fitting against the NEAR-TERM outcome
(best 2-of-first-3-seasons fantasy PPG right after draft) instead of the
long-term age-anchored target. Real motivation: the long-term target only
resolves for classes through ~2018, and a real, direct test showed our
pre-draft talent signal has degraded specifically for recent classes
(0.575 -> 0.402 rho, old era to recent) even though draft capital itself
barely moved -- switching to a target that resolves through 2023 (14 real
classes instead of 9-11, 725 real players instead of 503) lets us actually
validate against the modern game instead of guessing it still applies.
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
target = pd.read_csv(ROOT / "data" / "target_window_test.csv")
df = df.merge(target[["PLAYER_ID", "entry_3yr_best2"]], on="PLAYER_ID", how="left")
df = df.dropna(subset=["entry_3yr_best2"]).copy()
df = df[df["real_draft_year"] >= 2008].copy()
TARGET_COL = "entry_3yr_best2"
print(f"Trainable rows (entry_3yr_best2 resolved): {len(df)} (college: {(df['data_source']=='college').sum()}, "
      f"international: {(df['data_source']=='international').sum()}), classes "
      f"{int(df['real_draft_year'].min())}-{int(df['real_draft_year'].max())}")

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["LANE_AGILITY_TIME_PCTILE"] = df["LANE_AGILITY_TIME_PCTILE"].fillna(df["LANE_AGILITY_TIME_PCTILE"].median())
df["breakout_age_filled"] = df["breakout_age"].fillna(df["breakout_age"].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["talent_pctile"] = df["talent_pctile"].fillna(df["talent_pctile"].median())
for c in ["bpm", "porpag", "usg", "ts", "ortg"]:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())
df["fg_pct_filled"] = df["fg_pct"].fillna(df["fg_pct"].median())
df["three_pct_filled"] = df["three_pct"].fillna(df["three_pct"].median()) if "three_pct" in df.columns else np.nan
df["ft_pct_filled"] = df["ft_pct"].fillna(df["ft_pct"].median()) if "ft_pct" in df.columns else np.nan

ALL_FEATURE_BOUNDS = {
    "talent_pctile": (df["talent_pctile"], ((0, 0.6), (0.1, 1.0), (-10, 20))),
    "rec_filled": (df["rec_filled"], ((0, 60), (10, 100), (0, 20))),
    "exp_numeric": (df["exp_numeric"], ((1, 3), (0.5, 3), (-20, 5))),
    "LANE_AGILITY_TIME_PCTILE": (df["LANE_AGILITY_TIME_PCTILE"], ((0, 0.6), (0.1, 1.0), (-10, 15))),
    "draft_age_filled": (df["draft_age_filled"], ((18, 23), (0.5, 5), (-20, 5))),
    "breakout_age_filled": (df["breakout_age_filled"], ((18, 22), (0.5, 8), (-20, 5))),
    "fg_pct_filled": (df["fg_pct_filled"], ((0.35, 0.55), (0.10, 0.30), (-10, 20))),
    "porpag": (df["porpag"], ((-3, 6), (2, 12), (-10, 20))) if "porpag" in df.columns else None,
    "usg": (df["usg"], ((10, 25), (5, 25), (-10, 20))) if "usg" in df.columns else None,
    "ts": (df["ts"], ((45, 60), (5, 20), (-10, 20))) if "ts" in df.columns else None,
}
ALL_FEATURE_BOUNDS = {k: v for k, v in ALL_FEATURE_BOUNDS.items() if v is not None}


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
    rho, _ = spearmanr(pred, data[TARGET_COL])
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
        rho, _ = spearmanr(pred, test[TARGET_COL])
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
    rho, _ = spearmanr(pred, data[TARGET_COL])
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
        rho, _ = spearmanr(pred, test[TARGET_COL])
        if np.isfinite(rho):
            fold_rhos.append(rho)
    return np.mean(fold_rhos), fold_rhos


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
    feature_list = sys.argv[2].split(",") if len(sys.argv) > 2 else ["talent_pctile", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "draft_age_filled"]
    print(f"Testing ({mode}) vs {TARGET_COL}: {feature_list}")
    if mode == "pre":
        rho, folds = loco_cv(feature_list)
        print(f"PRE-DRAFT LOCO-CV: {rho:.4f}")
    else:
        rho, folds = loco_cv_post(feature_list)
        print(f"POST-DRAFT LOCO-CV: {rho:.4f}")
        baseline_folds = []
        for y in sorted(df["real_draft_year"].unique()):
            test = df[df["real_draft_year"] == y]
            r, _ = spearmanr(-test["pick_filled"], test[TARGET_COL])
            baseline_folds.append(r)
        print(f"DRAFT-ALONE baseline: {np.mean(baseline_folds):.4f}")
