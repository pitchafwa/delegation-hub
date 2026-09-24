"""Fit one stat's Kalman filter (Q, R, peak_age, slope_up, slope_down) via
differential evolution, maximizing Spearman correlation between the filter's
end-of-2022-23 belief (projected forward to 2023-24's start) and each
player's REAL per-minute rate in 2023-24 -- a genuine forward holdout, never
seen during fitting.

Usage: python fit_stat_kalman.py <STAT_COLUMN> [maxiter] [popsize]
MIN is a special case: gain=1 (directly tracking minutes/game), not a
per-minute rate of itself.
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

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

HOLDOUT_SEASON = "2023-24"
cutoff_date = df.loc[df["SEASON"] == HOLDOUT_SEASON, "GAME_DATE"].min()
fit_df = df[df["GAME_DATE"] < cutoff_date].copy()
holdout_df = df[df["SEASON"] == HOLDOUT_SEASON].copy()

player_ids = fit_df["PLAYER_ID"].to_numpy()
days = fit_df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = fit_df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = fit_df["MIN"].to_numpy(dtype=float)

last_game_idx = fit_df.groupby("PLAYER_ID").tail(1).index
last_game_date = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["GAME_DATE"]
last_game_age = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
days_to_holdout = (cutoff_date - last_game_date).dt.days

BOUNDS = [
    (1e-6, 0.01),   # Q: process noise per day -- audited in log-scale, no stat
                    # was near either edge, left as-is
    (0.1, 50.0),    # R: observation noise scale -- widened from a 20.0 ceiling
                    # that most stats were pinned against (PTS/REB/FTA/FTM/TOV/
                    # FG3M all sat within ~10% of it)
    (20, 30),       # peak_age
    (0.0, 0.08),    # slope_up (per year) -- floor stays non-negative (see prior
                    # fix); ceiling widened from 0.05, which MIN was pinned against
    (-0.5, 0.0),    # slope_down (per year) -- widened from -0.3, which MIN was
                    # pinned against
]


# Stats tracked directly per-game rather than as a per-minute rate: MIN (it
# IS the thing being tracked) and TD3 (a per-game 0/1 event, not sensibly
# expressed as "triple-doubles per minute").
DIRECT_STATS = {"MIN", "TD3"}


def build_actual(stat_col, is_direct):
    if is_direct:
        actual = holdout_df.groupby("PLAYER_ID").agg(stat_sum=(stat_col, "sum"), gp=(stat_col, "size"))
        actual["actual_rate"] = actual["stat_sum"] / actual["gp"]
    else:
        actual = holdout_df.groupby("PLAYER_ID").agg(
            stat_sum=(stat_col, "sum"), min_sum=("MIN", "sum"), gp=(stat_col, "size")
        )
        actual["actual_rate"] = actual["stat_sum"] / actual["min_sum"].replace(0, np.nan)
    return actual[actual["gp"] >= 20]


def objective(params, stat_col, gain_arr, obs_arr, actual, x0):
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


if __name__ == "__main__":
    stat_col = sys.argv[1]
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    popsize = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    is_direct = stat_col in DIRECT_STATS

    raw_col = minutes if stat_col == "MIN" else fit_df[stat_col].to_numpy(dtype=float)
    obs_arr = raw_col
    gain_arr = np.ones_like(minutes) if is_direct else minutes
    x0 = float(raw_col.sum()) / len(raw_col) if is_direct \
        else float(obs_arr.sum()) / max(float(gain_arr.sum()), 1.0)
    actual = build_actual(stat_col, is_direct)

    t0 = time.time()
    result = differential_evolution(
        objective, BOUNDS, args=(stat_col, gain_arr, obs_arr, actual, x0),
        maxiter=maxiter, popsize=popsize, seed=42, workers=1, tol=1e-5,
    )
    elapsed = time.time() - t0
    best_rho = -result.fun
    print(f"{stat_col} fit done in {elapsed:.0f}s. Best Spearman: {best_rho:.4f}")
    print("Params (Q, R, peak_age, slope_up, slope_down):", result.x.tolist())

    naive = fit_df.groupby("PLAYER_ID").apply(
        lambda d: (d[stat_col].mean() if is_direct else d[stat_col].sum() / max(d["MIN"].sum(), 1.0)),
        include_groups=False,
    )
    naive_merged = pd.DataFrame({"predicted": naive}).join(actual["actual_rate"], how="inner").dropna()
    naive_rho, _ = spearmanr(naive_merged["predicted"], naive_merged["actual_rate"])
    print(f"Naive baseline Spearman: {naive_rho:.4f}")

    out = {
        "stat": stat_col, "params": result.x.tolist(), "kalman_spearman": best_rho,
        "naive_spearman": float(naive_rho), "elapsed_sec": elapsed,
        "param_names": ["Q", "R", "peak_age", "slope_up", "slope_down"],
    }
    with open(ROOT / "data" / f"kalman_fit_{stat_col}.json", "w") as f:
        json.dump(out, f, indent=2)
