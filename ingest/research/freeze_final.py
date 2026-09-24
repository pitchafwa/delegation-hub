"""Freeze the final, audited DARKO-style composite as one versioned artifact.
Supersedes layer_a_model.frozen.json (v1, single blended additive model).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]

per_stat = {}
for stat in KALMAN_STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        fit = json.load(f)
    per_stat[stat] = {
        "params": dict(zip(fit["param_names"], fit["params"])),
        "kalman_spearman_vs_2023_24_holdout": fit["kalman_spearman"],
        "naive_spearman_vs_2023_24_holdout": fit["naive_spearman"],
    }

# STL and TD3: real, tested rejections of the Kalman approach -- logged with
# their numbers, not silently dropped.
with open(ROOT / "data" / "kalman_fit_STL.json") as f:
    stl_fit = json.load(f)
with open(ROOT / "data" / "kalman_fit_TD3.json") as f:
    td3_fit = json.load(f)

artifact = {
    "version": "2.0.0",
    "name": "Delegation League Dynasty Valuation Model",
    "supersedes": "layer_a_model.frozen.json (v1.0.0, single blended additive model)",
    "fit_date": "2026-09-18",
    "architecture": "DARKO-style: one Kalman filter per box-score stat (game-level "
                     "data, real-time belief updated after every game, aging curve "
                     "inside the state transition), recombined via the league's own "
                     "validated scoring formula.",
    "kalman_filtered_stats": per_stat,
    "excluded_from_kalman_logged_rejections": {
        "STL": {
            "reason": "Kalman approach tested WORSE than naive persistence after the "
                      "full bounds audit (0.635 vs 0.651 Spearman). Uses flat career "
                      "per-minute rate instead -- no age evolution, since none was "
                      "found to help.",
            "kalman_spearman_tested": stl_fit["kalman_spearman"],
            "naive_spearman": stl_fit["naive_spearman"],
        },
        "TD3": {
            "reason": "Kalman approach tested substantially worse than naive "
                      "persistence (0.347 vs 0.492 Spearman) -- triple-doubles are a "
                      "threshold event driven jointly by other stats, not something "
                      "that evolves smoothly on its own age curve. Uses last known "
                      "season's per-game rate as a placeholder.",
            "kalman_spearman_tested": td3_fit["kalman_spearman"],
            "naive_spearman": td3_fit["naive_spearman"],
        },
    },
    "opportunity_cost_curve": {
        "description": "Replacement-level value by keeper count, fit from this "
                        "league's real 2021-2025 draft history (0/6/8 keepers/team "
                        "observed -> 35.2/29.9/28.5 PPG average drafted value).",
        "floor": 14.84, "amp": 20.37, "tau": 19.96,
        "formula": "floor + amp * exp(-keepers_per_team / tau)",
        "current_value_K3": 32.4, "planned_future_value_K5": 30.7,
    },
    "trajectory_horizon_years": 7,
    "trajectory_horizon_note": "Originally attempted 'integrate until the trajectory "
                                "crosses below opportunity cost' but the linear "
                                "per-year slopes, only ever validated one season "
                                "ahead, produced absurd 20+ year runouts when "
                                "extrapolated further. Capped at a fixed, "
                                "honestly-labeled 7 years instead.",
    "known_bugs_found_and_fixed_during_this_build": [
        "Draft-round-shift bug: keeper-designation rounds were being counted as real "
        "draft positions, corrupting the draft-value-by-pick curve across years with "
        "different keeper counts.",
        "MIN special-case bug: observation array was set to a constant array of 1s "
        "instead of real minutes, making the first MIN fit meaningless (0.089 "
        "Spearman) until caught and fixed (corrected to 0.838, later 0.841).",
        "Negative pre-peak slope bug: PTS/REB/BLK/FG3M fit NEGATIVE growth rates for "
        "players below their peak age, because a single linear slope spanning the "
        "whole pre-peak range let the much larger 23-26-year-old population "
        "outweigh the much smaller (but real) 19-21-year-old sample, producing the "
        "implausible result of a 19-year-old (Cooper Flagg) projected to decline for "
        "7 straight years. Fixed by constraining slope_up >= 0; cost ~0.0001-0.002 "
        "Spearman across the 4 affected stats, essentially free.",
        "R (observation noise) bound too tight: most stats were pinned against the "
        "original 20.0 ceiling; widened to 50.0 and refit.",
    ],
    "final_validation_2023_24_holdout": {
        "naive_persistence": {"spearman": 0.8619, "pearson": 0.9009, "mae": 6.00},
        "old_frozen_layer_a_v1": {"spearman": 0.8662, "pearson": 0.8896, "mae": 7.04},
        "new_composite_v2": {"spearman": 0.8718, "pearson": 0.9076, "mae": 5.14},
    },
    "known_remaining_gaps": [
        "TD3 has no real predictive model, just a season-carried-forward placeholder.",
        "Game-level (DFS-style) next-game prediction was tested and found WORSE than "
        "simple rolling averages (0.723 vs 0.745 Spearman) -- this model is validated "
        "for season-ahead / multi-year dynasty value, not day-to-day lineup decisions.",
        "Rookie/prospect model (Phase 3b) not built -- new players entering the "
        "league start every filter at a generic league-average prior, not an "
        "informed one.",
    ],
}

with open(ROOT / "data" / "delegation_dynasty_model.frozen.json", "w") as f:
    json.dump(artifact, f, indent=2)

print("Saved research/data/delegation_dynasty_model.frozen.json (v2.0.0)")
print(f"\nFinal validation: composite Spearman={artifact['final_validation_2023_24_holdout']['new_composite_v2']['spearman']} "
      f"vs old model={artifact['final_validation_2023_24_holdout']['old_frozen_layer_a_v1']['spearman']} "
      f"vs naive={artifact['final_validation_2023_24_holdout']['naive_persistence']['spearman']}")
