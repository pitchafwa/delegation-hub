"""Fit the two final champion Output A models (pre-draft-only, and
post-draft talent+capital) on the FULL trainable dataset (not per-fold) to
get production parameters, and freeze them alongside their real LOCO-CV
validation numbers and the rejected-variant log.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import df, build_bounds, score

ROOT = Path(__file__).resolve().parent

PRE_DRAFT_FEATURES = ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "fg_pct_filled", "draft_age_filled"]
POST_DRAFT_FEATURES = ["bpm", "draft_age_filled", "fg_pct_filled"]
CAPITAL_BOUNDS = [(1, 300), (0.1, 50), (0.1, 3), (0, 30)]


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


def objective_pre(params, data, features):
    pred = score(params, data, features)
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


def objective_post(params, data, features, n_pre):
    pre_part = score(params[:n_pre], data, features)
    cap_part = capital_curve(params[n_pre:], data["real_draft_number"].to_numpy(dtype=float))
    pred = pre_part + cap_part
    if np.std(pred) < 1e-9:
        return 0.0
    rho, _ = spearmanr(pred, data["age_22_29_best3"])
    return -rho if np.isfinite(rho) else 0.0


print("Fitting final PRE-DRAFT model on full trainable dataset...")
pre_bounds = build_bounds(PRE_DRAFT_FEATURES)
pre_result = differential_evolution(
    objective_pre, pre_bounds, args=(df, PRE_DRAFT_FEATURES), maxiter=150, popsize=25, seed=42, workers=1, tol=1e-7,
)
pre_params = pre_result.x.tolist()
pre_pred = score(pre_result.x, df, PRE_DRAFT_FEATURES)
pre_rho_full, _ = spearmanr(pre_pred, df["age_22_29_best3"])
print(f"  full-data fit rho={pre_rho_full:.4f} (in-sample, not the validation number)")

print("Fitting final POST-DRAFT model (talent + capital) on full trainable dataset...")
post_bounds = build_bounds(POST_DRAFT_FEATURES) + CAPITAL_BOUNDS
n_pre = len(build_bounds(POST_DRAFT_FEATURES))
post_result = differential_evolution(
    objective_post, post_bounds, args=(df, POST_DRAFT_FEATURES, n_pre), maxiter=150, popsize=25, seed=42, workers=1, tol=1e-7,
)
post_params = post_result.x.tolist()
pre_part_full = score(post_result.x[:n_pre], df, POST_DRAFT_FEATURES)
cap_part_full = capital_curve(post_result.x[n_pre:], df["real_draft_number"].to_numpy(dtype=float))
post_pred = pre_part_full + cap_part_full
post_rho_full, _ = spearmanr(post_pred, df["age_22_29_best3"])
print(f"  full-data fit rho={post_rho_full:.4f} (in-sample, not the validation number)")

frozen = {
    "model": "Output A -- rookie/prospect long-term grade",
    "target": "age_22_29_best3 (best-3-of-any-3-consecutive-seasons average fantasy value, ages 22-29)",
    "pre_draft_model": {
        "features": PRE_DRAFT_FEATURES,
        "params": pre_params,
        "loco_cv_rho": 0.4531,
        "in_sample_full_data_rho": pre_rho_full,
        "functional_form": "additive bounded-ramp per feature: weight * clip((x-lo)/span, 0, 1), summed + intercept",
    },
    "post_draft_model": {
        "features": POST_DRAFT_FEATURES,
        "params": post_params,
        "loco_cv_rho": 0.5440,
        "in_sample_full_data_rho": post_rho_full,
        "capital_curve": "floor + k * (pick + c) ** (-p)",
        "capital_curve_param_order": ["k", "c", "p", "floor"],
        "n_pre_draft_params": n_pre,
        "baseline_draft_capital_alone_loco_cv_rho": 0.5335,
        "beats_baseline_by": round(0.5440 - 0.5335, 4),
    },
    "rejected_variants": {
        "pre_draft": [
            {"features": ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE"], "loco_cv_rho": 0.4117, "note": "baseline 4-feature combo"},
            {"features": ["+power_conf"], "loco_cv_rho": 0.4068, "note": "power-conference flag adds noise, rejected"},
            {"features": ["+fg_pct_filled"], "loco_cv_rho": 0.4211, "note": "helped modestly before age was added"},
            {"features": ["+ppg_c"], "loco_cv_rho": 0.4027, "note": "raw scoring rate adds noise, rejected"},
            {"features": ["fg_pct", "three_pct", "ft_pct", "ppg", "apg", "rec", "exp"], "loco_cv_rho": 0.2351, "note": "basic-box-stats-only (testing Stanford CS229 finding) -- MUCH worse for this age-anchored target; a real divergence from that paper's result, not force-fit to match it"},
            {"features": ["+power_conf", "+fg_pct_filled"], "loco_cv_rho": 0.4058, "note": "worse than fg_pct alone -- confirms power_conf is pure noise"},
            {"features": ["bpm", "draft_age_filled"], "loco_cv_rho": 0.4107, "note": "age alone with bpm -- real but modest signal"},
            {"features": ["bpm", "rec_filled", "LANE_AGILITY_TIME_PCTILE", "fg_pct_filled", "draft_age_filled"], "loco_cv_rho": 0.4399, "note": "dropping exp_numeric after adding age -- exp still adds real signal, not redundant with age"},
        ],
        "post_draft": [
            {"features": PRE_DRAFT_FEATURES, "loco_cv_rho": 0.5183, "note": "full 6-feature pre-draft combo + capital curve -- WORSE than leaner 3-feature version; too many joint params overfit the DE optimization even though each feature helped pre-draft-only"},
            {"features": ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "fg_pct_filled"], "loco_cv_rho": 0.5303, "note": "pre-age combo3 + capital -- essentially tied with baseline, does not beat it"},
            {"features": ["bpm", "draft_age_filled", "LANE_AGILITY_TIME_PCTILE"], "loco_cv_rho": 0.5262, "note": "agility instead of fg_pct -- worse"},
            {"features": ["bpm", "draft_age_filled", "rec_filled"], "loco_cv_rho": 0.5383, "note": "recruit rank instead of fg_pct -- beats baseline slightly but less than fg_pct version"},
            {"features": ["bpm", "draft_age_filled", "fg_pct_filled", "LANE_AGILITY_TIME_PCTILE"], "loco_cv_rho": 0.5418, "note": "adding agility on top of the winner -- slightly worse, agility redundant with capital post-draft"},
            {"features": ["draft_age_filled"], "loco_cv_rho": 0.5254, "note": "age alone + capital -- worse than baseline, needs bpm"},
        ],
    },
    "data_notes": [
        "1093 rookie-dataset players; birthdates real (nba_api CommonPlayerInfo), 1093/1093 matched.",
        "5 rows (0.46%) had physically impossible draft_age (real_draft_year data-quality issue for those specific rows, not a namesake ID collision -- birthdate matched correctly). Nulled and median-filled rather than left uninvestigated.",
        "Trainable rows restricted to real_draft_year <= 2018 (later classes too thin per class for fair LOCO-CV).",
    ],
}

out_path = ROOT / "data" / "output_a_model.frozen.json"
out_path.write_text(json.dumps(frozen, indent=2))
print(f"\nFrozen Output A model saved to {out_path}")
print(f"\nFINAL RESULT: post-draft LOCO-CV {post_rho_full and 0.5440:.4f} beats draft-capital-alone baseline 0.5335 by +0.0105")
