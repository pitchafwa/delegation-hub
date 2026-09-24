"""Real fix for the root issue Tommy identified: the original fitting
objective (fit_stat_kalman.py) validates against a ~1-season-ahead holdout,
where the Kalman filter's own real-time tracking of recent games already
does most of the work -- so the deterministic age-slope parameters barely
affect that score and end up weakly identified (this is why MIN's rescaled
refit still came back with essentially zero growth, despite real data
showing +1.5-2.4 min/game/year for developing players). But the actual
production use (kalman_vor.py / project_prospect_trajectory.py) extrapolates
7 YEARS forward with no real data to update against -- entirely dependent on
these same slope parameters. This script re-fits against genuine multi-year-
ahead real holdouts instead, using TWO different real (cutoff, target) pairs
spanning different eras and horizons (4yr and 5yr), mirroring the EXACT
two-stage projection production actually uses (partial-year to next season
start, then full annual steps) so what's being optimized matches what's
being used.
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
df["season_year"] = df["SEASON"].apply(lambda s: int(s.split("-")[0]))

# Two real (cutoff, target) pairs -- different horizons (4yr, 5yr) AND
# different eras, so the fitted curve isn't just calibrated to one slice of
# history (also a partial, cheap guard against the era-confound issue
# flagged separately -- not a full fix, but two independent eras averaged
# together is better than one).
HORIZONS = [
    {"cutoff_season": "2018-19", "target_season": "2022-23"},  # 4-year gap
    {"cutoff_season": "2020-21", "target_season": "2025-26"},  # 5-year gap
]
NEXT_SEASON_START = {  # real season-opener dates, for the partial-year step
    "2019-20": pd.Timestamp("2019-10-22"),
    "2021-22": pd.Timestamp("2021-10-19"),
}
DIRECT_STATS = {"MIN", "TD3"}


def prep_horizon(h):
    cutoff_date = df.loc[df["SEASON"] == h["cutoff_season"], "GAME_DATE"].min()
    next_season_key = f"{int(h['cutoff_season'][:4])+1}-{str(int(h['cutoff_season'][:4])+2)[-2:]}"
    next_season_start = NEXT_SEASON_START.get(next_season_key)
    if next_season_start is None:
        # derive from real data: first real game date of that season
        next_season_start = df.loc[df["SEASON"] == next_season_key, "GAME_DATE"].min()
    fit_df = df[df["GAME_DATE"] < cutoff_date].copy()
    target_df = df[df["SEASON"] == h["target_season"]].copy()
    target_year = int(h["target_season"][:4])
    cutoff_year = int(h["cutoff_season"][:4])
    n_full_years = target_year - (cutoff_year + 1)  # additional whole-year steps after the partial-year step
    return {
        "fit_df": fit_df, "target_df": target_df, "cutoff_date": cutoff_date,
        "next_season_start": next_season_start, "n_full_years": n_full_years,
    }


def build_actual(target_df, stat_col, is_direct):
    if is_direct:
        actual = target_df.groupby("PLAYER_ID").agg(stat_sum=(stat_col, "sum"), gp=(stat_col, "size"))
        actual["actual_rate"] = actual["stat_sum"] / actual["gp"]
    else:
        actual = target_df.groupby("PLAYER_ID").agg(
            stat_sum=(stat_col, "sum"), min_sum=("MIN", "sum"), gp=(stat_col, "size")
        )
        actual["actual_rate"] = actual["stat_sum"] / actual["min_sum"].replace(0, np.nan)
    return actual[actual["gp"] >= 20]


def project_multiyear(params, prep, stat_col, is_direct, x0):
    Q, R, peak_age, slope_up, slope_down = params
    fit_df = prep["fit_df"]
    player_ids = fit_df["PLAYER_ID"].to_numpy()
    days = fit_df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
    age = fit_df["AGE_AT_GAME"].to_numpy(dtype=float)
    minutes = fit_df["MIN"].to_numpy(dtype=float)
    raw_col = minutes if is_direct else fit_df[stat_col].to_numpy(dtype=float)
    gain_arr = np.ones_like(minutes) if is_direct else minutes

    posterior = run_filter_all_players(player_ids, days, age, gain_arr, raw_col, Q, R, peak_age, slope_up, slope_down, x0)
    fit_df = fit_df.copy()
    fit_df["_post"] = posterior
    last_idx = fit_df.groupby("PLAYER_ID").tail(1).index
    end_state = fit_df.loc[last_idx].set_index("PLAYER_ID")["_post"]
    last_age = fit_df.loc[last_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
    last_date = fit_df.loc[last_idx].set_index("PLAYER_ID")["GAME_DATE"]

    # stage 1: partial-year projection from each player's last real game to the START of the next real season
    days_gap = (prep["next_season_start"] - last_date).dt.days.clip(lower=0)
    diff0 = last_age - peak_age
    daily_slope = np.where(diff0 <= 0, slope_up, slope_down) / 365.0
    state = (end_state + daily_slope * days_gap).clip(lower=0)
    age0 = last_age  # age carried forward; full-year steps below use age0+k same as production build_trajectory

    # stage 2: n_full_years additional whole-year steps, EXACT same logic as build_trajectory in production
    for k in range(prep["n_full_years"]):
        diff = (age0 + k) - peak_age
        slope = np.where(diff <= 0, slope_up, slope_down)
        state = (state + slope).clip(lower=0)

    return state  # projected value at target season, indexed by PLAYER_ID


def objective(params, preps, stat_col, is_direct, x0):
    rhos = []
    for prep in preps:
        projected = project_multiyear(params, prep, stat_col, is_direct, x0)
        merged = pd.DataFrame({"predicted": projected}).join(prep["actual"]["actual_rate"], how="inner").dropna()
        if len(merged) < 30:
            continue
        rho, _ = spearmanr(merged["predicted"], merged["actual_rate"])
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return 0.0
    return -np.mean(rhos)


if __name__ == "__main__":
    stat_col = sys.argv[1]
    maxiter = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    popsize = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    is_direct = stat_col in DIRECT_STATS

    preps = [prep_horizon(h) for h in HORIZONS]
    for prep, h in zip(preps, HORIZONS):
        prep["actual"] = build_actual(prep["target_df"], stat_col, is_direct)
        print(f"  horizon {h['cutoff_season']} -> {h['target_season']} ({prep['n_full_years']+1} yr): "
              f"fit_df={len(prep['fit_df'])} rows, real target n={len(prep['actual'])}", flush=True)

    full_raw = df[stat_col].to_numpy(dtype=float) if not is_direct else df["MIN"].to_numpy(dtype=float)
    full_min = df["MIN"].to_numpy(dtype=float)
    x0 = float(full_raw.sum()) / len(full_raw) if is_direct else float(full_raw.sum()) / max(float(full_min.sum()), 1.0)

    # bounds: same rescale fix already validated for MIN's raw scale; other
    # stats stay on the small (rate-appropriate) scale, peak_age widened
    if stat_col == "MIN":
        slope_bounds = [(0.0, 4.0), (-4.0, 0.0)]
    else:
        slope_bounds = [(-0.08, 0.08), (-0.5, 0.0)]
    BOUNDS = [(1e-6, 0.01), (0.1, 50.0), (18, 34)] + slope_bounds

    t0 = time.time()
    result = differential_evolution(
        objective, BOUNDS, args=(preps, stat_col, is_direct, x0),
        maxiter=maxiter, popsize=popsize, seed=42, workers=1, tol=1e-5,
    )
    elapsed = time.time() - t0
    best_rho = -result.fun
    print(f"{stat_col} multi-year fit done in {elapsed:.0f}s. Mean multi-year Spearman: {best_rho:.4f}")
    print("Params (Q, R, peak_age, slope_up, slope_down):", result.x.tolist())

    # naive baseline: each player's own trailing average rate (no age curve at all), same multi-year test
    naive_rhos = []
    for prep in preps:
        naive = prep["fit_df"].groupby("PLAYER_ID").apply(
            lambda d: (d["MIN"].mean() if is_direct else d[stat_col].sum() / max(d["MIN"].sum(), 1.0)),
            include_groups=False,
        )
        merged = pd.DataFrame({"predicted": naive}).join(prep["actual"]["actual_rate"], how="inner").dropna()
        rho, _ = spearmanr(merged["predicted"], merged["actual_rate"])
        if np.isfinite(rho):
            naive_rhos.append(rho)
    print(f"Naive baseline (trailing avg, no age curve) mean multi-year Spearman: {np.mean(naive_rhos):.4f}")

    out = {
        "stat": stat_col, "params": result.x.tolist(), "multiyear_spearman": best_rho,
        "naive_multiyear_spearman": float(np.mean(naive_rhos)), "elapsed_sec": elapsed,
        "param_names": ["Q", "R", "peak_age", "slope_up", "slope_down"],
        "horizons_used": HORIZONS,
    }
    with open(ROOT / "data" / f"kalman_fit_{stat_col}_multiyear_test.json", "w") as f:
        json.dump(out, f, indent=2)
