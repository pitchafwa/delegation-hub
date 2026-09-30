"""Follow-up to test_kalman_possession_pts.py (2026-09-27): that script only checked SEASON-level rank correlation (the same metric DELCO's real
params were validated on). Tommy's actual question was about GAME-level accuracy, which that test doesn't measure. This reuses the two already-fit
parameter sets (from test_kalman_possession_PTS.json -- no re-optimization, so this is fast) and asks: for a REAL, SPECIFIC game, is the per-minute
or per-possession filter's honest pre-game prediction (return_pre=True -- never peeks at that game's own result) closer to what the player actually
scored?

Two exposure modes, both evaluated with the SAME state trajectory (the filter's belief is always updated using REAL exposure -- that's what
genuinely happened and is needed to correctly track the player's true rate; only how that belief gets turned into a predicted COUNT for a game
changes below):
  - "oracle": multiply the pre-game rate belief by that game's REAL minutes / REAL team pace (known only in hindsight). Isolates the rate-basis
    question alone, same convention as gamelevel_study.py's Part 1.
  - "forecast": multiply by what would genuinely be knowable BEFORE the game -- the player's trailing average minutes for both bases, and for the
    possession basis, that SPECIFIC matchup's expected pace. This is the scenario Tommy is actually asking about: if the two fastest teams in the
    league play each other, a per-possession projection has a real, forecastable way to use that; a per-minute one has no equivalent lever. The
    proxy version of this (Part 1's "real-forecast" row) already found a real, if modest, ~0.4% RMSE gain (12.792 -> 12.739) using a SIMPLE AVERAGE
    of the two teams' trailing pace (gamelevel_study.py's `exp_pace = (pace_td + pace_td_opp) / 2`) -- but Tommy correctly flagged that as the wrong
    combination rule (a known issue in tempo prediction, e.g. KenPom's college-basketball methodology): a team's raw to-date pace is already dragged
    toward the league mean by whatever mix of fast/slow opponents it happened to face, so two fast teams meeting should compound faster than a
    simple average implies, not just split the difference. Fixed here to the standard multiplicative form: each team's pace expressed as a ratio to
    league average, then those ratios multiplied together and rescaled by league average --
    `exp_pace = pace_td_own * pace_td_opp / league_avg_pace` -- so two teams each ~10% above average now compound to ~19% above average instead of
    averaging out to ~10%. (This still isn't a full opponent-strength-of-schedule adjustment like KenPom's iterative AdjTempo -- pace_td_own/opp are
    still each team's raw, unadjusted trailing average -- just the correct way to COMBINE two raw pace numbers into a matchup expectation.)

Evaluated on the 2023-24 holdout season plus every season after (2023-24 onward), each filter run continuously forward in time.

Writes research/data/test_kalman_possession_gamelevel.json. Touches no production file.
Usage: uv run python research/test_kalman_possession_gamelevel.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalman_engine import run_filter_all_players

ROOT = Path(__file__).resolve().parent
STAT = "PTS"

fit_result = json.load(open(ROOT / "data" / "test_kalman_possession_PTS.json"))

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

tb = pd.read_pickle(ROOT / "data" / "gamelevel" / "team_box_features.pkl")
from team_abbr import canon
tb["team"] = tb.TEAM_ABBREVIATION.map(canon)
tb_idx = tb.set_index(["team", "GAME_DATE"])
df["team_poss"] = tb_idx["poss"].reindex(pd.MultiIndex.from_arrays([df.TEAM, df.GAME_DATE])).to_numpy()
df["opp"] = tb_idx["opp"].reindex(pd.MultiIndex.from_arrays([df.TEAM, df.GAME_DATE])).to_numpy()
df["pace_td_own"] = tb_idx["pace_td"].reindex(pd.MultiIndex.from_arrays([df.TEAM, df.GAME_DATE])).to_numpy()
df["pace_td_opp"] = tb_idx["pace_td"].reindex(pd.MultiIndex.from_arrays([df.opp, df.GAME_DATE])).to_numpy()
df = df.dropna(subset=["team_poss"]).reset_index(drop=True)
df["player_poss"] = df["team_poss"] * (df["MIN"] / 48.0)

# expected pace for THIS matchup -- multiplicative combination of each team's pace ratio to league average, not a simple average (see module
# docstring): two teams each faster than average should compound, not split the difference back toward the mean.
league_avg_pace = df.groupby("SEASON")["team_poss"].transform("mean")
df["exp_pace"] = df["pace_td_own"] * df["pace_td_opp"] / league_avg_pace

# trailing average minutes as of BEFORE this game (genuinely forecastable, unlike real same-day minutes)
df["min_td"] = df.groupby("PLAYER_ID")["MIN"].transform(lambda s: s.expanding().mean().shift(1))
df["n_prior"] = df.groupby("PLAYER_ID").cumcount()
df["exp_player_poss"] = df["exp_pace"] * df["min_td"] / 48.0

HOLDOUT_SEASON = "2023-24"
holdout_seasons = sorted(s for s in df["SEASON"].unique() if s >= HOLDOUT_SEASON)
print("evaluating game-level accuracy on seasons:", holdout_seasons)

player_ids = df["PLAYER_ID"].to_numpy()
days = df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = df["MIN"].to_numpy(dtype=float)
poss = df["player_poss"].to_numpy(dtype=float)
min_td = df["min_td"].to_numpy(dtype=float)
exp_player_poss = df["exp_player_poss"].to_numpy(dtype=float)
obs_arr = df[STAT].to_numpy(dtype=float)
is_holdout = df["SEASON"].isin(holdout_seasons).to_numpy()
rotation = minutes >= 15  # rotation-minutes games only, matching the earlier gamelevel_study.py convention
have_forecast = (df["n_prior"] >= 10).to_numpy() & np.isfinite(min_td) & (min_td >= 15) & np.isfinite(exp_player_poss)


def rmse_mae(mask, predicted_count):
    err = obs_arr[mask] - predicted_count[mask]
    return {"n_games": int(mask.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}


def evaluate(basis):
    params = fit_result[basis]["params"]
    Q, R, peak_age, slope_up, slope_down = params
    real_gain = minutes if basis == "minute" else poss
    x0 = float(obs_arr.sum()) / max(float(real_gain.sum()), 1.0)
    # state update always uses REAL exposure -- that's what actually happened and is needed to track the true rate correctly
    _, x_pre = run_filter_all_players(player_ids, days, age, real_gain, obs_arr, Q, R, peak_age, slope_up, slope_down, x0, return_pre=True)

    oracle_mask = is_holdout & rotation
    oracle = rmse_mae(oracle_mask, x_pre * real_gain)

    forecast_gain = min_td if basis == "minute" else exp_player_poss
    forecast_mask = is_holdout & rotation & have_forecast
    forecast = rmse_mae(forecast_mask, x_pre * forecast_gain)

    return {"basis": basis, "oracle": oracle, "forecast": forecast}


if __name__ == "__main__":
    out = {
        "holdout_seasons": holdout_seasons,
        "oracle_mode": "real minutes / real team pace (hindsight) -- isolates the rate-basis question alone",
        "forecast_mode": "trailing avg minutes (both bases) / this matchup's exp_pace = avg(own trailing pace, opp trailing pace) for possession "
                          "-- genuinely knowable before the game, e.g. two fast teams meeting",
    }
    out["minute"] = evaluate("minute")
    out["possession"] = evaluate("possession")

    print("\n=== GAME-LEVEL ACCURACY (real, out-of-sample games) ===")
    for basis in ("minute", "possession"):
        r = out[basis]
        print(f"[{basis}]  oracle:   n={r['oracle']['n_games']}  RMSE={r['oracle']['rmse']:.4f}  MAE={r['oracle']['mae']:.4f}")
        print(f"[{basis}]  forecast: n={r['forecast']['n_games']}  RMSE={r['forecast']['rmse']:.4f}  MAE={r['forecast']['mae']:.4f}")

    with open(ROOT / "data" / "test_kalman_possession_gamelevel.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote research/data/test_kalman_possession_gamelevel.json")
