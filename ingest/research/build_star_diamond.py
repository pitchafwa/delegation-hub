"""Star and diamond calibration for Output A, mirroring WRPI/RUPI's exact
methodology (confirmed from the real code, not assumed):

STAR: a binary flag (not a 1-5 tier), computed SEPARATELY for pre-draft and
post-draft scores. Convert each score to a percentile against a frozen,
fully-matured reference pool (real_draft_year <= 2018, matching what LOCO-CV
was validated on). Build a decile hit-rate table against a real, fixed
"hit" bar (a genuinely NBA-fantasy-relevant player for a sustained stretch
in their prime -- NOT itself an elite bar), then set the star threshold at
whichever percentile shows a real jump in hit-rate -- exactly RUPI's method
(90th for RB, 95th for WR; basketball gets its own empirical answer).

DIAMOND: a SEPARATE, purpose-built index -- NOT derived from the main
Output A score. Restrict to a late-pick sub-pool, fit a heavily-regularized
logistic regression (C=0.05, class_weight="balanced") of the same "hit"
label on a curated, sign-expected indicator set, clip coefficients >= 0 and
renormalize to sum to 1, giving index weights. diamond_score = weighted sum
of signed z-scores. Flag threshold = 80th percentile of diamond_score
WITHIN the late pool. Sweep candidate cut-picks and pick the one with the
best LOCO-CV lift@5, mirroring RUPI's exact sweep-and-select approach.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df as full_df, build_bounds, score

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. Real, fixed "hit" bar in FANTASY_PPG units (same formula used
#    league-wide throughout this whole project): a genuinely fantasy-
#    relevant sustained-stretch player. Full-league real (40+ GP) season
#    distribution: median 23.8, 70th 31.1, 75th 33.5, 90th 44.3, 95th 51.0.
#    28.0 sits just under the 70th percentile of that full real-player
#    distribution -- "was a legitimate, startable contributor for a real
#    stretch in their prime," the same moderate spirit as WRPI's 12-PPG /
#    RUPI's 13-PPG bar (neither of those was itself an "elite" bar either --
#    the STAR designation comes from the model-score threshold, not the hit
#    bar itself).
# ---------------------------------------------------------------------------
HIT_BAR = 28.0
df = full_df.copy()
df["hit"] = (df["age_22_29_best3"] >= HIT_BAR).astype(int)
print(f"Hit bar: age_22_29_best3 >= {HIT_BAR}  -> base hit rate = {df['hit'].mean():.3f} ({df['hit'].sum()}/{len(df)})")

# ---------------------------------------------------------------------------
# 2. Load frozen Output A model, score every player pre-draft AND post-draft
# ---------------------------------------------------------------------------
frozen = json.loads((ROOT / "data" / "output_a_model.frozen.json").read_text())
PRE = frozen["pre_draft_model"]
POST = frozen["post_draft_model"]


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


pre_score = score(PRE["params"], df, PRE["features"])

from scipy.stats import rankdata
component_ranks = []
for name, cfg in POST["components"].items():
    n_pre = cfg["n_pre_draft_params"]
    p = cfg["params"]
    pre_part = score(p[:n_pre], df, cfg["features"])
    cap_part = capital_curve(p[n_pre:], df["pick_filled"].to_numpy(dtype=float))
    component_ranks.append(rankdata(pre_part + cap_part))
post_score = np.mean(component_ranks, axis=0)  # already an ensemble rank-average

df["pre_draft_score"] = pre_score
df["post_draft_score"] = post_score


def to_percentile(x, ref):
    ref_sorted = np.sort(ref)
    return np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)


df["pre_draft_pctl"] = to_percentile(df["pre_draft_score"].to_numpy(), df["pre_draft_score"].to_numpy())
df["post_draft_pctl"] = to_percentile(df["post_draft_score"].to_numpy(), df["post_draft_score"].to_numpy())

# ---------------------------------------------------------------------------
# 3. Decile hit-rate calibration tables -- find where hit-rate jumps
# ---------------------------------------------------------------------------
def decile_table(pctl_col):
    d = df.copy()
    d["_t"] = np.ceil(d[pctl_col] * 10).clip(1, 10).astype(int)
    g = d.groupby("_t").agg(n=("hit", "size"), hit_rate=("hit", "mean")).reset_index()
    return g


print("\n=== PRE-DRAFT decile hit-rate table ===")
pre_table = decile_table("pre_draft_pctl")
print(pre_table.to_string(index=False))

print("\n=== POST-DRAFT decile hit-rate table ===")
post_table = decile_table("post_draft_pctl")
print(post_table.to_string(index=False))

# decile 10 clearly stands out from decile 9 in BOTH tables (0.757 vs 0.568
# pre-draft; 0.730 vs 0.595 post-draft) -- top-decile = star, matching
# RUPI's own 90th-percentile RB threshold.
STAR_PCTL = 0.90
df["is_star_pre"] = (df["pre_draft_pctl"] >= STAR_PCTL).astype(int)
df["is_star_post"] = (df["post_draft_pctl"] >= STAR_PCTL).astype(int)
print(f"\nSTAR_PCTL = {STAR_PCTL}")
print(f"is_star_pre hit rate: {df[df['is_star_pre']==1]['hit'].mean():.3f} (n={df['is_star_pre'].sum()}) vs base {df['hit'].mean():.3f}")
print(f"is_star_post hit rate: {df[df['is_star_post']==1]['hit'].mean():.3f} (n={df['is_star_post'].sum()}) vs base {df['hit'].mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Diamond index -- a SEPARATE, purpose-built logistic-regression index
#    over a late-pick sub-pool, mirroring RUPI's fit_diamond_rb.py exactly.
# ---------------------------------------------------------------------------
IND = {
    "bpm": +1, "porpag": +1, "fg_pct_filled": +1, "three_pct_filled": +1,
    "rec_filled": +1, "usg": +1, "ts": +1, "LANE_AGILITY_TIME_PCTILE": +1,
    "power_conf": +1,
    "draft_age_filled": -1, "breakout_age_filled": -1, "never_broke_out": -1,
    "exp_numeric": -1,
}
# several of these are raw torvik columns not yet filled at the dataframe
# level (only inside ALL_FEATURE_BOUNDS's own stored series) -- fill for real
# use here so the logistic regression never sees NaN.
for f in IND:
    if df[f].isna().any():
        df[f] = df[f].fillna(df[f].median())

print(f"\n=== Diamond index: cut-pick sweep ===")
print(f"real_draft_number range: {df['real_draft_number'].min():.0f}-{df['real_draft_number'].max():.0f}")

CUT_CANDIDATES = [14, 20, 30, 40, 50, 60]


def fit_diamond(late, ind):
    z_cols = {}
    means, stds = {}, {}
    for f, sgn in ind.items():
        x = late[f].to_numpy(dtype=float)
        m, s = np.nanmean(x), np.nanstd(x) + 1e-9
        means[f], stds[f] = m, s
        z_cols[f] = sgn * (x - m) / s
    Z = np.column_stack([z_cols[f] for f in ind])
    y = late["hit"].to_numpy()
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    lr = LogisticRegression(C=0.05, max_iter=6000, class_weight="balanced").fit(Z, y)
    w = np.clip(lr.coef_[0], 0, None)
    if w.sum() < 1e-9:
        return None
    w = w / w.sum()
    dscore = (Z * w).sum(axis=1)
    return {"weights": dict(zip(ind.keys(), w.tolist())), "means": means, "stds": stds, "dscore": dscore}


def loco_lift_at_5(cut_pick, ind):
    """Leave-one-draft-class-out: fit on other classes, score held-out class,
    measure precision@5 among the held-out late pool vs base hit rate."""
    classes = sorted(df["real_draft_year"].unique())
    precisions = []
    for held_out in classes:
        train_late = df[(df["real_draft_year"] != held_out) & (df["pick_filled"] >= cut_pick)]
        test_late = df[(df["real_draft_year"] == held_out) & (df["pick_filled"] >= cut_pick)]
        if len(test_late) < 5 or len(train_late) < 30:
            continue
        fit = fit_diamond(train_late, ind)
        if fit is None:
            continue
        z_cols = []
        for f in ind:
            x = test_late[f].to_numpy(dtype=float)
            z_cols.append(ind[f] * (x - fit["means"][f]) / fit["stds"][f])
        Z_test = np.column_stack(z_cols)
        w = np.array([fit["weights"][f] for f in ind])
        test_dscore = (Z_test * w).sum(axis=1)
        top5_idx = np.argsort(-test_dscore)[:5]
        precisions.append(test_late["hit"].to_numpy()[top5_idx].mean())
    return np.mean(precisions) if precisions else np.nan


results = []
for cut in CUT_CANDIDATES:
    late_pool = df[df["pick_filled"] >= cut]
    base_rate = late_pool["hit"].mean()
    lift5 = loco_lift_at_5(cut, IND)
    results.append((cut, len(late_pool), base_rate, lift5, lift5 / base_rate if base_rate > 0 else np.nan))
    print(f"cut_pick={cut:3d}  late_pool_n={len(late_pool):4d}  base_rate={base_rate:.3f}  LOCO precision@5={lift5:.3f}  lift={lift5/base_rate if base_rate>0 else float('nan'):.2f}x")

# require a real, robust-sized late pool -- a small pool's LOCO lift@5 can
# look great by pure noise (confirmed: cut=40 with only 110 players showed
# a sweep-estimated 1.71x lift that collapsed to 1.11x on the final full-pool
# fit, while cut=14's far more data-backed 1.48x held up). Floor chosen so
# the late pool still has enough real players for LOCO folds to be meaningful.
MIN_POOL_N = 200
eligible = [r for r in results if r[1] >= MIN_POOL_N and np.isfinite(r[4])]
best_cut = max(eligible, key=lambda r: r[4])
print(f"\nBest cut_pick by LOCO lift@5 among pools with n>={MIN_POOL_N} (precision relative to that pool's own base rate, not raw precision -- a higher-base-rate pool gets higher raw precision for free, and a small pool's lift estimate is too noisy to trust): {best_cut[0]}")

# final fit on the FULL late pool at the chosen cut
CUT_PICK = best_cut[0]
late_final = df[df["pick_filled"] >= CUT_PICK]
final_fit = fit_diamond(late_final, IND)
late_final = late_final.copy()
late_final["dscore"] = final_fit["dscore"]
FLAG_THRESHOLD = float(late_final["dscore"].quantile(0.80))
print(f"\nFinal diamond fit: cut_pick={CUT_PICK}, flag_threshold (80th pctl of late pool)={FLAG_THRESHOLD:.4f}")
print("Weights:")
for f, w in sorted(final_fit["weights"].items(), key=lambda kv: -kv[1]):
    print(f"  {f:28s} {w:.4f}")

df["diamond_score"] = np.nan
df.loc[late_final.index, "diamond_score"] = late_final["dscore"]
df["is_diamond"] = ((df["pick_filled"] >= CUT_PICK) & (df["diamond_score"] >= FLAG_THRESHOLD)).astype(int)
diamond_hit_rate = df[df["is_diamond"] == 1]["hit"].mean()
late_base_rate = late_final["hit"].mean()
print(f"\nis_diamond hit rate: {diamond_hit_rate:.3f} (n={df['is_diamond'].sum()}) vs late-pool base {late_base_rate:.3f}  (lift={diamond_hit_rate/late_base_rate:.2f}x)")

# ---------------------------------------------------------------------------
# 5. Save calibration to the frozen model artifact
# ---------------------------------------------------------------------------
frozen["star_diamond_calibration"] = {
    "hit_bar_fantasy_ppg": HIT_BAR,
    "hit_bar_definition": "age_22_29_best3 >= 28.0 -- a genuinely fantasy-relevant, startable contributor for a sustained stretch in their prime (not itself an elite bar; full real 40+GP-season league distribution: median 23.8, 70th pctl 31.1, 90th pctl 44.3)",
    "base_hit_rate": float(df["hit"].mean()),
    "star": {
        "star_pctl": STAR_PCTL,
        "method": "percentile of pre-draft / post-draft score against the fully-matured reference pool (real_draft_year <= 2018); decile hit-rate table showed a clean jump at the top decile for both pre-draft (0.757 vs 0.568 in decile 9) and post-draft (0.730 vs 0.595 in decile 9)",
        "pre_draft_decile_table": pre_table.to_dict("records"),
        "post_draft_decile_table": post_table.to_dict("records"),
        "is_star_pre_hit_rate": float(df[df["is_star_pre"] == 1]["hit"].mean()),
        "is_star_post_hit_rate": float(df[df["is_star_post"] == 1]["hit"].mean()),
    },
    "diamond": {
        "cut_pick_candidates_swept": CUT_CANDIDATES,
        "cut_pick_sweep_results": [
            {"cut_pick": c, "late_pool_n": n, "base_rate": br, "loco_precision_at_5": l5, "lift": lf}
            for c, n, br, l5, lf in results
        ],
        "chosen_cut_pick": CUT_PICK,
        "flag_threshold": FLAG_THRESHOLD,
        "weights": final_fit["weights"],
        "z_means": {k: float(v) for k, v in final_fit["means"].items()},
        "z_stds": {k: float(v) for k, v in final_fit["stds"].items()},
        "indicator_signs": IND,
        "late_pool_base_rate": float(late_base_rate),
        "is_diamond_hit_rate": float(diamond_hit_rate),
        "lift": float(diamond_hit_rate / late_base_rate),
    },
}
out_path = ROOT / "data" / "output_a_model.frozen.json"
out_path.write_text(json.dumps(frozen, indent=2))
print(f"\nSaved star + diamond calibration to {out_path}")
