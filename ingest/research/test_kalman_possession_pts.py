"""Non-destructive test (2026-09-27, at Tommy's request): does DELCO's real Kalman engine improve if PTS is tracked as a per-POSSESSION rate
instead of per-MINUTE? Exact same fitting methodology as fit_stat_kalman.py (differential_evolution maximizing Spearman correlation between the
filter's end-of-training belief, projected forward, and real 2023-24 rates -- a genuine forward holdout), same BOUNDS, same default maxiter/popsize,
so the result is directly comparable to the real production number in data/kalman_fit_PTS.json (multiyear_r2 aside -- this reproduces the ORIGINAL
single-holdout objective that file's "kalman_spearman"/"naive_spearman" fields... actually those fields were superseded; compare against
fitted_multiyear_spearman=0.6960 / naive_multiyear_spearman=0.5555 is NOT apples-to-apples since that used a different multiyear backtest script.
The fair comparison is: re-run THIS SAME single-holdout methodology for per-minute PTS (using the unmodified kalman_engine.py, unmodified bounds)
right alongside the per-possession version, in the same run, so both numbers come from identical code and effort.

Per-game team possessions come from the REAL historical team box scores (team_box_features.pkl's proper NBA pace formula, real OREB/DREB), joined
onto each player-game by (team, date) -- analogous to how MIN is that player's own real minutes for that specific game (not a projection).
Player possession-equivalent for a game = team_poss_that_game * (player_minutes / 48) -- his fair share of his team's actual pace that night.

Writes NOTHING to any production file. Output: research/data/test_kalman_possession_PTS.json (a new file, not kalman_fit_PTS.json).
Usage: uv run python research/test_kalman_possession_pts.py [maxiter] [popsize]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from kalman_engine import run_filter_all_players

ROOT = Path(__file__).resolve().parent
STAT = "PTS"

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

tb = pd.read_pickle(ROOT / "data" / "gamelevel" / "team_box_features.pkl")
FIX = {"NOH": "NOP", "NJN": "BKN", "CHO": "CHA", "SEA": "OKC"}
tb["team"] = tb.TEAM_ABBREVIATION.map(lambda t: FIX.get(t, t))
poss_lookup = tb.set_index(["team", "GAME_DATE"])["poss"]

df["team_poss"] = poss_lookup.reindex(pd.MultiIndex.from_arrays([df.TEAM, df.GAME_DATE])).to_numpy()
before = len(df)
df = df.dropna(subset=["team_poss"]).reset_index(drop=True)
print(f"matched team possessions for {len(df)}/{before} rows ({len(df)/before:.1%})")
df["player_poss"] = df["team_poss"] * (df["MIN"] / 48.0)

HOLDOUT_SEASON = "2023-24"
cutoff_date = df.loc[df["SEASON"] == HOLDOUT_SEASON, "GAME_DATE"].min()
fit_df = df[df["GAME_DATE"] < cutoff_date].copy()
holdout_df = df[df["SEASON"] == HOLDOUT_SEASON].copy()

player_ids = fit_df["PLAYER_ID"].to_numpy()
days = fit_df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = fit_df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = fit_df["MIN"].to_numpy(dtype=float)
poss = fit_df["player_poss"].to_numpy(dtype=float)

last_game_idx = fit_df.groupby("PLAYER_ID").tail(1).index
last_game_date = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["GAME_DATE"]
last_game_age = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
days_to_holdout = (cutoff_date - last_game_date).dt.days

BOUNDS = [
    (1e-6, 0.01),
    (0.1, 50.0),
    (20, 30),
    (0.0, 0.08),
    (-0.5, 0.0),
]


def build_actual(basis):
    if basis == "minute":
        actual = holdout_df.groupby("PLAYER_ID").agg(stat_sum=(STAT, "sum"), gain_sum=("MIN", "sum"), gp=(STAT, "size"))
    else:
        actual = holdout_df.groupby("PLAYER_ID").agg(stat_sum=(STAT, "sum"), gain_sum=("player_poss", "sum"), gp=(STAT, "size"))
    actual["actual_rate"] = actual["stat_sum"] / actual["gain_sum"].replace(0, np.nan)
    return actual[actual["gp"] >= 20]


def objective(params, gain_arr, obs_arr, actual, x0):
    Q, R, peak_age, slope_up, slope_down = params
    posterior = run_filter_all_players(player_ids, days, age, gain_arr, obs_arr, Q, R, peak_age, slope_up, slope_down, x0)
    fit_df["_posterior"] = posterior
    end_state = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["_posterior"]
    diff = last_game_age - peak_age
    daily_slope = np.where(diff <= 0, slope_up, slope_down) / 365.0
    projected = end_state + daily_slope * days_to_holdout
    merged = pd.DataFrame({"predicted": projected}).join(actual["actual_rate"], how="inner").dropna()
    if len(merged) < 30:
        return 0.0
    rho, _ = spearmanr(merged["predicted"], merged["actual_rate"])
    return -rho if np.isfinite(rho) else 0.0


def run_basis(basis, maxiter, popsize):
    obs_arr = fit_df[STAT].to_numpy(dtype=float)
    gain_arr = minutes if basis == "minute" else poss
    x0 = float(obs_arr.sum()) / max(float(gain_arr.sum()), 1.0)
    actual = build_actual(basis)

    t0 = time.time()
    result = differential_evolution(
        objective, BOUNDS, args=(gain_arr, obs_arr, actual, x0),
        maxiter=maxiter, popsize=popsize, seed=42, workers=1, tol=1e-5,
    )
    elapsed = time.time() - t0
    best_rho = -result.fun
    print(f"[{basis}] fit done in {elapsed:.0f}s. Best Spearman: {best_rho:.4f}")
    print(f"[{basis}] Params (Q, R, peak_age, slope_up, slope_down):", result.x.tolist())

    naive_col = "MIN" if basis == "minute" else "player_poss"
    naive = fit_df.groupby("PLAYER_ID").apply(
        lambda d: d[STAT].sum() / max(d[naive_col].sum(), 1.0), include_groups=False
    )
    naive_merged = pd.DataFrame({"predicted": naive}).join(actual["actual_rate"], how="inner").dropna()
    naive_rho, _ = spearmanr(naive_merged["predicted"], naive_merged["actual_rate"])
    print(f"[{basis}] Naive baseline Spearman: {naive_rho:.4f}")

    return {
        "stat": STAT, "basis": basis, "params": result.x.tolist(), "kalman_spearman": best_rho,
        "naive_spearman": float(naive_rho), "elapsed_sec": elapsed,
        "param_names": ["Q", "R", "peak_age", "slope_up", "slope_down"],
    }


if __name__ == "__main__":
    maxiter = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    popsize = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "maxiter": maxiter, "popsize": popsize}
    out["minute"] = run_basis("minute", maxiter, popsize)
    out["possession"] = run_basis("possession", maxiter, popsize)

    print("\n=== SUMMARY ===")
    print(f"per-minute:      Spearman {out['minute']['kalman_spearman']:.4f}  (naive {out['minute']['naive_spearman']:.4f})")
    print(f"per-possession:  Spearman {out['possession']['kalman_spearman']:.4f}  (naive {out['possession']['naive_spearman']:.4f})")
    print(f"production kalman_fit_PTS.json for reference: fitted_multiyear_spearman=0.6960, naive_multiyear_spearman=0.5555 (different, multiyear metric -- not directly comparable to the single-holdout numbers above; use the minute/possession pair in THIS run for the fair comparison)")

    with open(ROOT / "data" / "test_kalman_possession_PTS.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote research/data/test_kalman_possession_PTS.json")
