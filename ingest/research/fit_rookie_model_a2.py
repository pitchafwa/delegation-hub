"""Flexible feature-combination tester for Output A, run via CLI with a
comma-separated feature list so multiple combinations can be tested in
parallel background processes rather than guessed at sequentially.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")
df = df.dropna(subset=["age_22_29_best3"]).copy()
df = df[df["real_draft_year"] <= 2018].copy()

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["rec_filled"] = df["rec"].fillna(0)
df["n_college_seasons_c"] = df["n_college_seasons"]
POWER_CONFS = {"ACC", "B10", "B12", "BE", "P10", "P12", "SEC", "Amer"}
df["power_conf"] = df["conf"].isin(POWER_CONFS).astype(int)
df["fg_pct_filled"] = df["fg_pct"].fillna(df["fg_pct"].median())
df["three_pct_filled"] = df["three_pct"].fillna(df["three_pct"].median())
df["ft_pct_filled"] = df["ft_pct"].fillna(df["ft_pct"].median())
df["ppg_c"] = df["ppg"]
df["apg_c"] = df["apg"]
for c in ["WINGSPAN_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE"]:
    df[c] = df[c].fillna(df[c].median())
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
# undrafted players (real NaN real_draft_number) must NEVER reach the
# capital curve as a raw NaN -- a single NaN row in a training fold makes
# spearmanr return NaN for the WHOLE fold, which every downstream script's
# objective function silently maps to a flat 0.0 for every candidate
# parameter set. That makes the optimizer's objective completely flat
# (no signal to climb), so differential_evolution just wanders and lands
# on effectively random parameters -- confirmed as the cause of wild,
# non-reproducible seed-to-seed LOCO-CV swings after undrafted players'
# previously-fake pick=0 was correctly nulled. Same 61-just-past-the-draft
# convention already used in build_dashboard_data.py.
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["breakout_age_filled"] = df["breakout_age_filled"].fillna(df["breakout_age_filled"].max())
df["never_broke_out"] = df["never_broke_out"].fillna(1).astype(int)

# All candidate features and their fitting bounds (lo_x, span, weight -- sign free)
ALL_FEATURE_BOUNDS = {
    "bpm": (df["bpm"], ((-5, 5), (3, 20), (-10, 20))),
    "porpag": (df["porpag"], ((-3, 6), (2, 12), (-10, 20))),
    "usg": (df["usg"], ((10, 25), (5, 25), (-10, 20))),
    "ts": (df["ts"], ((45, 60), (5, 20), (-10, 20))),
    "ortg": (df["ortg"], ((90, 115), (5, 30), (-10, 20))),
    # weight was originally free-signed (-10,20) and converged to a NEGATIVE
    # value in the champion post-draft fit -- but the real raw correlation
    # between rec_filled and outcome is +0.14 (whole reference pool), and
    # +0.17 even controlling for bpm. A negative weight contradicts the real
    # data and is an overfit artifact (same failure mode as the earlier
    # negative-pre-peak-slope bug) -- caught via a real user example
    # (Cameron Boozer's own elite recruit ranking actively hurting his
    # score). Constrained to non-negative, matching the real relationship.
    "rec_filled": (df["rec_filled"], ((0, 60), (10, 100), (0, 20))),
    "exp_numeric": (df["exp_numeric"], ((1, 3), (0.5, 3), (-20, 5))),
    "LANE_AGILITY_TIME_PCTILE": (df["LANE_AGILITY_TIME_PCTILE"], ((0, 0.6), (0.1, 1.0), (-10, 15))),
    "THREE_QUARTER_SPRINT_PCTILE": (df["THREE_QUARTER_SPRINT_PCTILE"], ((0, 0.6), (0.1, 1.0), (-10, 15))),
    "WINGSPAN_PCTILE": (df["WINGSPAN_PCTILE"], ((0, 0.6), (0.1, 1.0), (-10, 15))),
    "n_college_seasons_c": (df["n_college_seasons_c"], ((1, 3), (0.5, 3), (-10, 10))),
    "power_conf": (df["power_conf"], ((0, 0.5), (0.3, 1.0), (-10, 10))),
    # NOTE: fg_pct/three_pct/ft_pct are real 0-1 FRACTIONS in this dataset
    # (e.g. Blake Griffin's real college fg_pct = 0.654), not percentage
    # points. Bounds must match that scale or the ramp's lo_x is
    # unreachable and the feature is silently inert (weight*0 for every
    # player, every fit) -- a real bug caught via the dashboard build when
    # "FG% 0.7" showed for a real 65%-shooter. Confirmed via df['fg_pct'].describe().
    "fg_pct_filled": (df["fg_pct_filled"], ((0.35, 0.55), (0.10, 0.30), (-10, 20))),
    "three_pct_filled": (df["three_pct_filled"], ((0.25, 0.40), (0.05, 0.25), (-10, 20))),
    "ft_pct_filled": (df["ft_pct_filled"], ((0.55, 0.85), (0.10, 0.35), (-10, 20))),
    "ppg_c": (df["ppg_c"], ((5, 20), (5, 20), (-10, 20))),
    "apg_c": (df["apg_c"], ((0.5, 5), (1, 8), (-10, 20))),
    "draft_age_filled": (df["draft_age_filled"], ((18, 23), (0.5, 5), (-20, 5))),
    "breakout_age_filled": (df["breakout_age_filled"], ((18, 22), (0.5, 8), (-20, 5))),
    "never_broke_out": (df["never_broke_out"], ((0, 0.5), (0.3, 1.0), (-20, 10))),
    "porpag": (df["porpag"].fillna(df["porpag"].median()), ((-3, 6), (2, 12), (-10, 20))),
    "ts": (df["ts"].fillna(df["ts"].median()), ((45, 60), (5, 20), (-10, 20))),
    "ortg": (df["ortg"].fillna(df["ortg"].median()), ((90, 115), (5, 30), (-10, 20))),
}


def build_bounds(feature_list):
    bounds = []
    for f in feature_list:
        bounds.extend(ALL_FEATURE_BOUNDS[f][1])
    bounds.append((0, 40))  # intercept
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


def loco_cv(feature_list, maxiter=50, popsize=15, seed=42):
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


if __name__ == "__main__":
    feature_list = sys.argv[1].split(",")
    print(f"Testing features: {feature_list}")
    rho, folds = loco_cv(feature_list)
    print(f"LOCO-CV mean rho: {rho:.4f}")
    print(f"Fold rhos: {[round(r,3) for r in folds]}")
