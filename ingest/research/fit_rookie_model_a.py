"""Output A: the long-term prospect grade. Pre-draft (talent only) and
post-draft (talent + draft capital) fit SEPARATELY with the same additive
bounded-component functional form, via differential evolution directly
against leave-one-DRAFT-CLASS-out Spearman correlation -- mirroring RUPI's
own architecture exactly, applied to real college data instead of guessed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")
df = df.dropna(subset=["age_22_29_best3"]).copy()
# thin recent classes (2019+) can't be evaluated fairly in LOCO-CV -- too few players per fold.
df = df[df["real_draft_year"] <= 2018].copy()
print(f"Output A trainable rows: {len(df)}, draft classes: {sorted(df['real_draft_year'].unique())}")

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["rec_filled"] = df["rec"].fillna(0)  # unranked recruit -- real, informative zero, not missing
df["n_college_seasons_c"] = df["n_college_seasons"]

# Real, non-circular program-strength proxy: power-conference membership
# (an external categorization, not derived from outcome data -- avoids the
# leakage risk of a data-driven "program prestige" score).
POWER_CONFS = {"ACC", "B10", "B12", "BE", "P10", "P12", "SEC", "Amer"}
df["power_conf"] = df["conf"].isin(POWER_CONFS).astype(int)

# Basic box stats -- a real academic finding (Edwards et al., Stanford CS229)
# is that BASIC college stats (FG%, 3P%, AST) predicted NBA win shares about
# as well as advanced composites like BPM, contrary to the assumption that
# advanced stats should dominate. Worth testing directly, not assuming BPM
# already captures everything they do.
df["fg_pct_filled"] = df["fg_pct"].fillna(df["fg_pct"].median())
df["three_pct_filled"] = df["three_pct"].fillna(df["three_pct"].median())
df["ft_pct_filled"] = df["ft_pct"].fillna(df["ft_pct"].median())
df["ppg_c"] = df["ppg"]
df["apg_c"] = df["apg"]

# Fill missing combine percentiles with the population median (no combine
# invite is itself mildly informative -- lower profile prospect -- but
# shouldn't be treated as a hard zero).
for c in ["WINGSPAN_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE"]:
    df[c] = df[c].fillna(df[c].median())

FEATURES = [
    ("bpm", "bpm"), ("rec_filled", "rec"), ("exp_numeric", "exp_inv"),
    ("LANE_AGILITY_TIME_PCTILE", "agility"),
]
for col, _ in FEATURES:
    df[col] = df[col].fillna(df[col].median())

# Bounds per ramp: (lo_x, span, weight). Weight sign left FREE (not forced
# positive) so the optimizer can tell us the real direction -- if something
# comes back negative where we expected positive, that's a real finding to
# report, not silently flip.
RAMP_BOUNDS = {
    "porpag": ((-3, 6), (2, 12), (-10, 20)),
    "bpm": ((-5, 5), (3, 20), (-10, 20)),
    "usg": ((10, 25), (5, 25), (-10, 20)),
    "ts": ((45, 60), (5, 20), (-10, 20)),
    "ortg": ((90, 115), (5, 30), (-10, 20)),
    "rec": ((0, 60), (10, 100), (-10, 20)),
    "exp_inv": ((1, 3), (0.5, 3), (-20, 5)),  # expect negative: more experience for same output = worse
    "agility": ((0, 0.6), (0.1, 1.0), (-10, 15)),
    "sprint": ((0, 0.6), (0.1, 1.0), (-10, 15)),
    "wingspan": ((0, 0.6), (0.1, 1.0), (-10, 15)),
    "seasons": ((1, 3), (0.5, 3), (-10, 10)),
}
PRE_BOUNDS = []
for _, name in FEATURES:
    PRE_BOUNDS.extend(RAMP_BOUNDS[name])
PRE_BOUNDS.append((0, 40))  # intercept

CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]  # k, c, p, floor for k*(pick+c)^-p + floor


def pre_draft_score(params, data):
    idx = 0
    score = np.zeros(len(data))
    for col, name in FEATURES:
        lo_x, span, weight = params[idx], params[idx + 1], params[idx + 2]
        idx += 3
        hi_x = lo_x + span
        x = data[col].to_numpy(dtype=float)
        frac = np.clip((x - lo_x) / max(hi_x - lo_x, 1e-6), 0.0, 1.0)
        score += weight * frac
    score += params[idx]  # intercept
    return score


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def objective_pre(params, data):
    pred = pre_draft_score(params, data)
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


def objective_post(params, data, pre_params):
    n_pre = len(PRE_BOUNDS)
    pre_part = pre_draft_score(params[:n_pre], data)
    cap_part = capital_curve(params[n_pre:], data["real_draft_number"].to_numpy(dtype=float))
    pred = pre_part + cap_part
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


def loco_cv(objective_fn, bounds, extra_args=(), maxiter=40, popsize=12, seed=42):
    classes = sorted(df["real_draft_year"].unique())
    fold_rhos = []
    for held_out in classes:
        train = df[df["real_draft_year"] != held_out]
        test = df[df["real_draft_year"] == held_out]
        result = differential_evolution(
            objective_fn, bounds, args=(train,) + extra_args, maxiter=maxiter, popsize=popsize,
            seed=seed, workers=1, tol=1e-5,
        )
        if extra_args:
            n_pre = len(PRE_BOUNDS)
            pre_part = pre_draft_score(result.x[:n_pre], test)
            cap_part = capital_curve(result.x[n_pre:], test["real_draft_number"].to_numpy(dtype=float))
            pred = pre_part + cap_part
        else:
            pred = pre_draft_score(result.x, test)
        rho, _ = spearmanr(pred, test["age_22_29_best3"])
        if np.isfinite(rho):
            fold_rhos.append(rho)
        print(f"  held out {held_out}: test rho={rho:.4f}  (n={len(test)})")
    return np.mean(fold_rhos), fold_rhos


if __name__ == "__main__":
    baseline_rho, _ = spearmanr(-df["real_draft_number"], df["age_22_29_best3"])
    print(f"\nBASELINE (draft position alone, in-sample, not even LOCO): {baseline_rho:.4f}\n")

    print("=== PRE-DRAFT model (talent only, no draft capital) LOCO-CV ===")
    pre_rho, pre_folds = loco_cv(objective_pre, PRE_BOUNDS, maxiter=60, popsize=15)
    print(f"PRE-DRAFT LOCO-CV mean rho: {pre_rho:.4f}\n")

    print("=== POST-DRAFT model (talent + capital curve) LOCO-CV ===")
    post_bounds = PRE_BOUNDS + CAPITAL_BOUNDS
    post_rho, post_folds = loco_cv(objective_post, post_bounds, extra_args=(None,), maxiter=60, popsize=15)
    print(f"POST-DRAFT LOCO-CV mean rho: {post_rho:.4f}")

    baseline_folds = []
    for held_out in sorted(df["real_draft_year"].unique()):
        test = df[df["real_draft_year"] == held_out]
        rho, _ = spearmanr(-test["real_draft_number"], test["age_22_29_best3"])
        baseline_folds.append(rho)
    print(f"\nDRAFT-ALONE baseline, fold-averaged (fair comparison): {np.mean(baseline_folds):.4f}")
    print(f"PRE-DRAFT LOCO-CV:  {pre_rho:.4f}")
    print(f"POST-DRAFT LOCO-CV: {post_rho:.4f}")
