"""Same multi-year-ahead + quality-weighted design as
fit_stat_kalman_qualityweighted.py, with ONE corrected piece: the scoring
metric. That version used Spearman rank correlation, inherited unquestioned
from the original short-horizon script -- but Spearman is invariant to any
monotonic transform of the predictions, so it can't tell a well-calibrated
projection from one that shrinks every player's value by a similar relative
amount and still preserves rank order. A real sanity check caught exactly
this: PTS's fitted curve produced a blended points-per-game trajectory that
crashed from 10.1 to 1.7 by age 27 for a representative young quality
player -- nonsense in absolute terms, but apparently fine by Spearman since
relative ordering was barely disturbed. Production (VOR, opportunity-cost
comparisons, dashboard trajectories) runs entirely on ABSOLUTE values, not
ranks, so the objective now maximizes R² (coefficient of determination)
against real target-season values instead -- the same metric Output B used
successfully elsewhere in this project, with no calibration problems there.
Everything else (real multi-year horizons, quality-player population,
two-stage projection mechanics mirroring production) is unchanged.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from keeper_value_over_replacement import opportunity_cost

from kalman_engine import run_filter_all_players

ROOT = Path(__file__).resolve().parent
K_THIS_YEAR = 3
REPLACEMENT_BAR = opportunity_cost(K_THIS_YEAR)

# --- real "quality player" set: cleared the real keeper-replacement bar at
# their real career-best season, using the SAME league scoring formula
# already used for the dashboard's historical-context feature ---
season_base_q = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base_q = season_base_q[season_base_q["GP"] >= 20].copy()
season_base_q["fantasy_pg"] = (
    season_base_q["PTS"] + 1.5 * season_base_q["REB"] + 2 * season_base_q["AST"] + 3 * season_base_q["STL"]
    + 3 * season_base_q["BLK"] + season_base_q["FG3M"] + 2 * season_base_q["FTM"] - season_base_q["FTA"]
    - season_base_q["TOV"] + 3 * season_base_q["TD3"]
) / season_base_q["GP"]
career_best = season_base_q.groupby("PLAYER_ID")["fantasy_pg"].max()
QUALITY_PLAYER_IDS = set(career_best[career_best >= REPLACEMENT_BAR].index)
print(f"Quality players (real career-best season >= {REPLACEMENT_BAR:.1f} replacement PPG): "
      f"{len(QUALITY_PLAYER_IDS)} of {season_base_q['PLAYER_ID'].nunique()} real players with a qualifying season.", flush=True)

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
    actual = actual[actual["gp"] >= 20]
    # the fitting/tracking itself still uses every real player (see fit_df in
    # __main__) -- only the SCORED evaluation set is restricted to real
    # quality players, so the optimizer is graded on getting difference-
    # makers right, not on matching bench players' flat, unremarkable careers
    return actual[actual.index.isin(QUALITY_PLAYER_IDS)]


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


def r2_score(pred, actual):
    """Real coefficient of determination -- penalizes magnitude/calibration
    error, not just rank order (unlike Spearman, which is blind to it)."""
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot <= 1e-9:
        return None
    return 1 - ss_res / ss_tot


def objective(params, preps, stat_col, is_direct, x0):
    scores = []
    for prep in preps:
        projected = project_multiyear(params, prep, stat_col, is_direct, x0)
        merged = pd.DataFrame({"predicted": projected}).join(prep["actual"]["actual_rate"], how="inner").dropna()
        if len(merged) < 30:
            continue
        r2 = r2_score(merged["predicted"].to_numpy(), merged["actual_rate"].to_numpy())
        if r2 is not None and np.isfinite(r2):
            scores.append(r2)
    if not scores:
        return 1e6  # heavy real penalty, not a silent 0.0 -- R2 has no natural floor like Spearman's [-1,1]
    return -np.mean(scores)


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
    best_r2 = -result.fun
    print(f"{stat_col} calibrated multi-year fit done in {elapsed:.0f}s. Mean R2 (quality players only): {best_r2:.4f}")
    print("Params (Q, R, peak_age, slope_up, slope_down):", result.x.tolist())

    # naive baseline: each player's own trailing average rate (no age curve at all), same multi-year test
    naive_r2s, naive_rhos, fitted_rhos = [], [], []
    for prep in preps:
        naive = prep["fit_df"].groupby("PLAYER_ID").apply(
            lambda d: (d["MIN"].mean() if is_direct else d[stat_col].sum() / max(d["MIN"].sum(), 1.0)),
            include_groups=False,
        )
        merged = pd.DataFrame({"predicted": naive}).join(prep["actual"]["actual_rate"], how="inner").dropna()
        r2 = r2_score(merged["predicted"].to_numpy(), merged["actual_rate"].to_numpy())
        if r2 is not None and np.isfinite(r2):
            naive_r2s.append(r2)
        rho, _ = spearmanr(merged["predicted"], merged["actual_rate"])
        if np.isfinite(rho):
            naive_rhos.append(rho)
        # fitted model's rank quality too -- secondary diagnostic only, not optimized against
        fitted_pred = project_multiyear(result.x, prep, stat_col, is_direct, x0)
        fmerged = pd.DataFrame({"predicted": fitted_pred}).join(prep["actual"]["actual_rate"], how="inner").dropna()
        frho, _ = spearmanr(fmerged["predicted"], fmerged["actual_rate"])
        if np.isfinite(frho):
            fitted_rhos.append(frho)
    print(f"Naive baseline mean multi-year R2: {np.mean(naive_r2s):.4f}  (naive Spearman: {np.mean(naive_rhos):.4f})")
    print(f"Fitted model Spearman (diagnostic only, not optimized against): {np.mean(fitted_rhos):.4f}")

    out = {
        "stat": stat_col, "params": result.x.tolist(), "multiyear_r2": best_r2,
        "naive_multiyear_r2": float(np.mean(naive_r2s)), "naive_multiyear_spearman": float(np.mean(naive_rhos)),
        "fitted_multiyear_spearman": float(np.mean(fitted_rhos)), "elapsed_sec": elapsed,
        "param_names": ["Q", "R", "peak_age", "slope_up", "slope_down"],
        "horizons_used": HORIZONS,
    }
    with open(ROOT / "data" / f"kalman_fit_{stat_col}_calibrated.json", "w") as f:
        json.dump(out, f, indent=2)
