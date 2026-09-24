"""Freeze the FINAL Output A models: pre-draft-only champion (unchanged),
and the post-draft ENSEMBLE (champion + rec_variant + breakout_variant,
rank-averaged) which beat every single-model post-draft candidate across
all 5 seed trials -- the real, seed-stable winner of the exhaustive search.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr, rankdata

from fit_rookie_model_a2 import df, build_bounds, score

ROOT = Path(__file__).resolve().parent

PRE_DRAFT_FEATURES = ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "draft_age_filled"]
ENSEMBLE_MODELS = {
    "champion": ["bpm", "draft_age_filled"],
    "rec_variant": ["bpm", "draft_age_filled", "rec_filled"],
    "breakout_variant": ["bpm", "breakout_age_filled"],
}
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
    cap_part = capital_curve(params[n_pre:], data["pick_filled"].to_numpy(dtype=float))
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

print("Fitting the 3 ensemble component models on full trainable dataset...")
ensemble_component_params = {}
for name, features in ENSEMBLE_MODELS.items():
    bounds = build_bounds(features) + CAPITAL_BOUNDS
    n_pre = len(build_bounds(features))
    result = differential_evolution(
        objective_post, bounds, args=(df, features, n_pre), maxiter=150, popsize=25, seed=42, workers=1, tol=1e-7,
    )
    ensemble_component_params[name] = {
        "features": features,
        "params": result.x.tolist(),
        "n_pre_draft_params": n_pre,
    }
    print(f"  {name} fit done")

# full-data in-sample ensemble check (not the validation number, just a sanity check)
preds = []
for name, cfg in ensemble_component_params.items():
    n_pre = cfg["n_pre_draft_params"]
    p = cfg["params"]
    pre_part = score(p[:n_pre], df, cfg["features"])
    cap_part = capital_curve(p[n_pre:], df["pick_filled"].to_numpy(dtype=float))
    preds.append(rankdata(pre_part + cap_part))
ensemble_pred = np.mean(preds, axis=0)
ensemble_rho_full, _ = spearmanr(ensemble_pred, df["age_22_29_best3"])
print(f"Full-data in-sample ensemble rho: {ensemble_rho_full:.4f} (not the validation number)")

frozen = {
    "model": "Output A -- rookie/prospect long-term grade (FINAL, post-exhaustive-search, post-bugfix)",
    "target": "age_22_29_best3 (best-3-of-any-3-consecutive-seasons average fantasy value, ages 22-29)",
    "bugfix_note": "fg_pct_filled/three_pct_filled/ft_pct_filled were originally given ramp bounds "
        "assuming a 0-100 percentage-point scale (e.g. (35,55)), but the real underlying data is a "
        "0-1 fraction (Blake Griffin's real college fg_pct = 0.654). Since no real player's value ever "
        "reached the ramp's lo_x, these three features were SILENTLY INERT (weight*0 for every player, "
        "every fit) in every result reported before this freeze -- caught by eyeballing a real player's "
        "value in the dashboard build ('FG% 0.7' for a 65%-shooter), not by the LOCO-CV numbers alone. "
        "Bounds fixed to the real 0-1 scale; FG% was then retested everywhere it had been used and found "
        "to genuinely HURT once able to actually contribute (not just neutral) -- dropped from both the "
        "pre-draft model and the post-draft ensemble's champion/breakout-age components. The ensemble's "
        "overall result barely moved (0.5480 -> 0.5490) because its real strength was always the "
        "diversity across draft-age/breakout-age/recruit-rank, not FG%. Diamond index was NOT affected "
        "(it z-scores raw feature values directly, no ramp bounds involved).",
    "second_audit_note": "Three more real bugs found via a user data-quality audit AFTER the first freeze: "
        "(1) a namesake-collision bug in link_college_to_nba.py -- a unique-name match was accepted with "
        "NO year-proximity check (unlike the multi-candidate branch), silently linking real historical NBA "
        "players (e.g. Alex English, a 1970s Hall of Famer; Amir Johnson, who never played college at all) "
        "to unrelated modern college seasons. Fixed by requiring <=2yr gap on unique-name matches too; "
        "removed ~200 bad rows. (2) the 2025-26 college season (year=2026 in torvik's convention) was never "
        "pulled at all, so real 2026 draftees (Cameron Boozer, AJ Dybantsa, Darryn Peterson) were missing or "
        "falsely matched to old namesakes -- fixed by extending the live-endpoint pull. (3) the age-anchored "
        "target's bio-year floor (>=2010) was copied from the entry-anchored target's floor even though "
        "age_window_best() doesn't share that function's false-zero vulnerability -- unnecessarily zeroed "
        "out real, resolvable 2008-2009 draftees' actual outcomes; loosened to >=2008 for age-anchored "
        "targets only. (4) rec_filled's weight was free-signed and converged to NEGATIVE despite a real "
        "+0.14 raw / +0.17 bpm-partial correlation with outcome -- an overfit artifact (caught via Cameron "
        "Boozer's own elite recruit ranking actively hurting his score); constrained to non-negative, "
        "costing ~0.008 pre-draft LOCO-CV for a more defensible, generalizable fit. (5) nba_api's DRAFT_NUMBER=0 "
        "sentinel for undrafted players was treated as a literal pick 0 (better than the #1 overall pick), "
        "inflating those players' capital-curve contribution -- fixed by treating it as null. Fixing (5) also "
        "surfaced a previously-masked issue: raw NaN picks reaching the capital curve made the DE objective "
        "function flat (spearmanr of any NaN-containing vector returns NaN, silently mapped to a neutral 0.0), "
        "causing wildly unstable, effectively-random LOCO-CV results across seeds until every fitting script "
        "was switched to a pick_filled column (NaN filled to 61, just past the real draft) before the capital "
        "curve. Net effect of the whole audit: reference pool grew 368->414 players and got meaningfully "
        "cleaner; BOTH the model AND the draft-alone baseline improved substantially (ensemble 0.549->0.592, "
        "baseline 0.534->0.581) because removing ~200 garbage rows raised predictability across the board -- "
        "the model's real edge over draft capital alone (+0.011) is similar in magnitude to before (+0.015), "
        "just recalibrated onto a cleaner scale, not a windfall from the bugfixes themselves.",
    "pre_draft_model": {
        "features": PRE_DRAFT_FEATURES,
        "params": pre_params,
        "loco_cv_rho": 0.4527,
        "functional_form": "additive bounded-ramp per feature: weight * clip((x-lo)/span, 0, 1), summed + intercept",
    },
    "post_draft_model": {
        "type": "ensemble -- rank-average of 3 component models' predictions",
        "components": ensemble_component_params,
        "capital_curve": "floor + k * (pick + c) ** (-p)",
        "capital_curve_param_order": ["k", "c", "p", "floor"],
        "combine_method": "average the rank (scipy.stats.rankdata) of each component's raw prediction, across the 3 components",
        "loco_cv_rho_mean_across_5_seeds": 0.5921,
        "loco_cv_rho_per_seed": {"42": 0.5903, "7": 0.5908, "123": 0.5932, "999": 0.5928, "2024": 0.5935},
        "single_best_component_loco_cv_rho_mean_across_5_seeds": 0.5908,
        "baseline_draft_capital_alone_loco_cv_rho": 0.5809,
        "beats_baseline_by": round(0.5921 - 0.5809, 4),
        "beats_best_single_model_by": round(0.5921 - 0.5908, 4),
        "why_ensemble_over_single_model": "each component model makes different real mistakes (draft_age vs breakout_age; recruit rank as an extra signal) -- rank-averaging reduces variance and won in 5/5 seed trials, a materially more robust margin over baseline than any single-model champion.",
    },
    "rejected_variants": {
        "note": "see git history for the pre-bugfix version of this file with the full original search log. FG%-inclusive results below are RE-VERIFIED post-bugfix; other entries are unaffected by the bug and unchanged.",
        "fg_pct_retested_post_bugfix": [
            {"features": ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "fg_pct_filled"], "loco_cv": 0.3942, "note": "pre-draft combo3 with REAL (non-inert) fg_pct -- now WORSE than the 4-feature baseline (0.4117) it was once reported to beat. The bounds bug fully explains the original spurious +0.0094."},
            {"features": ["bpm", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "fg_pct_filled", "draft_age_filled"], "loco_cv": 0.4296, "note": "pre-draft combo7 with real fg_pct -- worse than the same combo without it (0.4346), confirmed FG% net-negative once real"},
            {"features": ["bpm", "draft_age_filled", "fg_pct_filled"], "loco_cv": 0.5261, "note": "post-draft champion+capital with real fg_pct -- much worse than bpm+age alone (0.5425)"},
            {"features": ["bpm", "breakout_age_filled", "fg_pct_filled"], "loco_cv": 0.5379, "note": "post-draft breakout-variant+capital with real fg_pct -- worse than bpm+breakout_age alone (0.5421)"},
        ],
        "pairwise_interaction_screen": {
            "method": "screened residual (target minus champion-model rank-prediction) against every pairwise z-scored feature product (190 pairs) via Spearman; top candidates confirmed with full LOCO-CV DE fits. Predates the FG% bugfix but doesn't depend on the buggy bounds (raw z-scores), so still valid.",
            "top_screened_candidate": {"pair": ["power_conf", "apg_c"], "screen_rho_vs_residual": 0.1527, "single_seed_loco_cv": 0.5466, "5_seed_mean_loco_cv": 0.5428, "verdict": "REJECTED -- looked real on one seed (+0.0026 over champion) but washed out to a statistical tie once averaged across 5 seeds. A real example of why every promising result needs a seed-stability check before being trusted, not just single-seed confirmation."},
            "other_confirmed_candidates": [
                {"pair": ["rec_filled", "apg_c"], "loco_cv": 0.5324, "verdict": "worse than champion"},
                {"pair": ["power_conf", "ppg_c"], "loco_cv": 0.5173, "verdict": "worse than champion"},
                {"pair": ["porpag", "power_conf"], "loco_cv": 0.5220, "verdict": "worse than champion"},
                {"pair": ["LANE_AGILITY_TIME_PCTILE", "power_conf"], "loco_cv": 0.5416, "verdict": "worse than champion"},
                {"pair": ["bpm", "three_pct_filled"], "loco_cv": 0.5320, "verdict": "worse than champion"},
                {"pair": ["ortg", "draft_age_filled"], "loco_cv": 0.5370, "verdict": "worse than champion"}
            ]
        },
        "breakout_age_and_advanced_stat_swaps": [
            {"features": ["porpag", "draft_age_filled", "fg_pct_filled"], "loco_cv": 0.5333, "note": "porpag instead of bpm (pre-bugfix fg_pct, still directionally valid since porpag itself was on a correct scale) -- essentially ties baseline, worse than champion"},
            {"features": ["ts", "draft_age_filled", "fg_pct_filled"], "loco_cv": 0.5076, "note": "true shooting instead of bpm -- much worse"},
            {"features": ["ortg", "draft_age_filled", "fg_pct_filled"], "loco_cv": 0.4984, "note": "offensive rating instead of bpm -- much worse"},
            {"features": ["age_bpm_interaction", "fg_pct_filled"], "loco_cv": 0.5241, "note": "explicit age*bpm product term used INSTEAD of separate age+bpm -- worse than additive champion"},
            {"features": ["bpm", "draft_age_filled", "fg_pct_filled", "age_bpm_interaction"], "loco_cv": 0.4984, "note": "age*bpm interaction ADDED on top of champion -- much worse, extra joint DE parameter overfits"}
        ],
        "seed_stability_of_champion_alone_post_bugfix": {"42": 0.5425, "7": 0.5410, "123": 0.5409, "999": 0.5435, "2024": 0.5435, "mean": 0.5423}
    },
    "data_notes": [
        "1093 rookie-dataset players; birthdates real (nba_api CommonPlayerInfo), 1093/1093 matched.",
        "5 rows (0.46%) had physically impossible draft_age (real_draft_year data-quality issue for those specific rows, not a namesake ID collision). Nulled and median-filled.",
        "breakout_age: age at first real college season with BPM>=6.0 AND mpg>=20 (excludes garbage-time flukes), computed from full multi-year college trajectories via torvik's stable per-player id, real nba_api birthdates. 593/1086 players had a real breakout by this definition; the rest flagged never_broke_out=1 (informative, not missing).",
        "Trainable rows restricted to real_draft_year <= 2018 (later classes too thin per class for fair LOCO-CV).",
        "Every reported LOCO-CV number in this file that mattered for a go/no-go decision was checked across 5 random DE seeds before being trusted -- single-seed results are noted as such and were NOT used to make final decisions.",
        "fg_pct/three_pct/ft_pct scale bug found and fixed after this freeze was first built (see bugfix_note) -- this version reflects the corrected numbers.",
    ],
}

out_path = ROOT / "data" / "output_a_model.frozen.json"
out_path.write_text(json.dumps(frozen, indent=2))
print(f"\nFrozen FINAL Output A model saved to {out_path}")
print(f"\nFINAL RESULT (post-bugfix): post-draft ENSEMBLE 5-seed-mean LOCO-CV 0.5490 beats draft-capital-alone baseline 0.5335 by +0.0155")
