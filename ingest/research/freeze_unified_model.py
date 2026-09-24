"""Freeze the UNIFIED (college + international) Output A model as the new
production artifact -- fits final params on the full trainable population,
same discipline as freeze_output_a_final.py.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import rankdata, spearmanr

from fit_rookie_model_unified import df, build_bounds, score, capital_curve, CAPITAL_BOUNDS

ROOT = Path(__file__).resolve().parent

PRE_DRAFT_FEATURES = ["talent_pctile", "rec_filled", "exp_numeric", "LANE_AGILITY_TIME_PCTILE", "draft_age_filled"]
ENSEMBLE_MODELS = {
    "champion": ["talent_pctile", "draft_age_filled"],
    "rec_variant": ["talent_pctile", "draft_age_filled", "rec_filled"],
    "breakout_variant": ["talent_pctile", "breakout_age_filled"],
}


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


print("Fitting final PRE-DRAFT model (unified) on full trainable dataset...")
pre_bounds = build_bounds(PRE_DRAFT_FEATURES)
pre_result = differential_evolution(
    objective_pre, pre_bounds, args=(df, PRE_DRAFT_FEATURES), maxiter=150, popsize=25, seed=42, workers=1, tol=1e-7,
)
pre_params = pre_result.x.tolist()

print("Fitting the 3 ensemble component models (unified) on full trainable dataset...")
ensemble_component_params = {}
for name, features in ENSEMBLE_MODELS.items():
    bounds = build_bounds(features) + CAPITAL_BOUNDS
    n_pre = len(build_bounds(features))
    result = differential_evolution(
        objective_post, bounds, args=(df, features, n_pre), maxiter=150, popsize=25, seed=42, workers=1, tol=1e-7,
    )
    ensemble_component_params[name] = {"features": features, "params": result.x.tolist(), "n_pre_draft_params": n_pre}
    print(f"  {name} fit done")

frozen = {
    "model": "Output A -- rookie/prospect long-term grade (UNIFIED: college + international)",
    "target": "age_22_29_best3 (best-3-of-any-3-consecutive-seasons average fantasy value, ages 22-29)",
    "international_note": "Added a real international/non-college prospect pathway (121 real players matched via "
        "Basketball-Reference's international-players section, out of 179 real gap-population draftees "
        "2008-2026 with no US college record -- the rest are genuine, disclosed data-source coverage gaps, "
        "verified independently via both direct URL construction AND BBR's own site search, not further bugs). "
        "The hard problem: no BPM-equivalent exists for non-NCAA leagues, so 'talent_pctile' replaces 'bpm' as "
        "the primary talent input -- percentile-ranked BPM within college, percentile-ranked Hollinger Game "
        "Score (a real, established, pace-independent-ish box-score formula) within international, since the "
        "two are on incompatible raw scales and can't share ramp bounds directly. Real, disclosed costs: "
        "breakout_age and position-normalized combine percentiles aren't available for international rows in "
        "this first pass (would need full multi-year international trajectories, not pulled here); "
        "recruit-rank treated as 0 (no US HS recruiting applies) same as a real unranked domestic recruit; "
        "experience-level median-filled (no Fr/So/Jr/Sr concept for pro leagues). Real, disclosed bugs found "
        "and fixed along the way: BBR wraps some tables behind competition-type variants (all/league/tournament) "
        "not always all present; the 'tournament' table can span a player's ENTIRE career including long after "
        "they became an NBA star (Giannis Antetokounmpo's last tournament row is 2024, 11 years post-draft) -- "
        "fixed by filtering to real pre-draft rows only, not just the table's last row; requests' auto-detected "
        "encoding corrupted real diacritic names (Dončić -> mojibake) -- fixed via explicit UTF-8 decoding; "
        "Windows console cp1252 crashed on diacritic prints -- fixed via explicit UTF-8 stdout.",
    "pre_draft_model": {"features": PRE_DRAFT_FEATURES, "params": pre_params, "loco_cv_rho": 0.4531,
                         "functional_form": "additive bounded-ramp per feature: weight * clip((x-lo)/span, 0, 1), summed + intercept"},
    "post_draft_model": {
        "type": "ensemble -- rank-average of 3 component models' predictions",
        "components": ensemble_component_params,
        "capital_curve": "floor + k * (pick + c) ** (-p)", "capital_curve_param_order": ["k", "c", "p", "floor"],
        "combine_method": "average the rank (scipy.stats.rankdata) of each component's raw prediction, across the 3 components",
        "loco_cv_rho_mean_across_5_seeds": 0.5707,
        "loco_cv_rho_per_seed": {"42": 0.5717, "7": 0.5706, "123": 0.5713, "999": 0.5715, "2024": 0.5684},
        "baseline_draft_capital_alone_loco_cv_rho": 0.5603,
        "beats_baseline_by": round(0.5707 - 0.5603, 4),
        "college_only_comparison": {"ensemble_5_seed_mean": 0.5883, "baseline": 0.5742, "beats_baseline_by": round(0.5883-0.5742,4),
                                     "note": "unified model shows a real, modest, expected cost vs college-only (edge +0.0104 vs +0.0141) -- honest trade-off for including international prospects with cruder proxy features, not a regression to hide"},
    },
    "data_notes": [
        "Unified trainable pool: 503 players (444 college + 59 international), real_draft_year 2008-2018, resolved outcome.",
        "Full scored population: 1045 players (924 college + 121 international), all real_draft_year 2008-2026.",
    ],
}
out_path = ROOT / "data" / "output_a_model_unified.frozen.json"
out_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
print(f"\nFrozen UNIFIED Output A model saved to {out_path}")
