"""Build pooled residuals (actual - predicted) across MULTIPLE held-out
season transitions, not just one, for a more robust test of candidate
contextual variables. Applying already-frozen parameters is cheap (no
fitting) -- this just runs the composite projection at several different
point-in-time cutoffs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalman_engine import run_filter_all_players

ROOT = Path(__file__).resolve().parent
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_base["FANTASY_PTS"] = (
    season_base["PTS"] + 1.5 * season_base["REB"] + 2 * season_base["AST"] + 3 * season_base["STL"]
    + 3 * season_base["BLK"] + season_base["FG3M"] + 2 * season_base["FTM"] - season_base["FTA"]
    - season_base["TOV"] + 3 * season_base["TD3"]
)
season_base["FANTASY_PPG"] = season_base["FANTASY_PTS"] / season_base["GP"].replace(0, np.nan)

params_by_stat = {}
for stat in KALMAN_STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        params_by_stat[stat] = json.load(f)["params"]


def project_for_holdout(holdout_season_year):
    cutoff_date = pd.Timestamp(f"{holdout_season_year}-10-01")
    fit_df = df[df["GAME_DATE"] < cutoff_date].copy()
    if len(fit_df) < 1000:
        return None

    player_ids = fit_df["PLAYER_ID"].to_numpy()
    days = fit_df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
    age = fit_df["AGE_AT_GAME"].to_numpy(dtype=float)
    minutes = fit_df["MIN"].to_numpy(dtype=float)
    last_game_idx = fit_df.groupby("PLAYER_ID").tail(1).index
    last_game_date = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["GAME_DATE"]
    last_game_age = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
    days_to_holdout = (cutoff_date - last_game_date).dt.days

    projections = pd.DataFrame(index=last_game_date.index)
    for stat in KALMAN_STATS:
        Q, R, peak_age, slope_up, slope_down = params_by_stat[stat]
        is_min = stat == "MIN"
        obs = minutes if is_min else fit_df[stat].to_numpy(dtype=float)
        gain = np.ones_like(minutes) if is_min else minutes
        x0 = float(minutes.sum()) / len(minutes) if is_min else float(obs.sum()) / max(float(gain.sum()), 1.0)
        posterior = run_filter_all_players(player_ids, days, age, gain, obs, Q, R, peak_age, slope_up, slope_down, x0)
        fit_df["_post"] = posterior
        end_state = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["_post"]
        diff = last_game_age - peak_age
        daily_slope = np.where(diff <= 0, slope_up, slope_down) / 365.0
        projections[stat] = (end_state + daily_slope * days_to_holdout).clip(lower=0)

    stl_rate = fit_df.groupby("PLAYER_ID").apply(lambda d: d["STL"].sum() / max(d["MIN"].sum(), 1.0), include_groups=False)
    projections["STL"] = stl_rate.reindex(projections.index).fillna(0.0)

    prior_year = season_base[season_base["SEASON_YEAR"] == holdout_season_year - 1]
    td3_pg = (prior_year.set_index("PLAYER_ID")["TD3"] / prior_year.set_index("PLAYER_ID")["GP"].replace(0, np.nan))
    td3_pg = td3_pg.reindex(projections.index).fillna(0.0)

    pred_ppg = (
        projections["PTS"] * projections["MIN"] + 1.5 * projections["REB"] * projections["MIN"]
        + 2 * projections["AST"] * projections["MIN"] + 3 * projections["STL"] * projections["MIN"]
        + 3 * projections["BLK"] * projections["MIN"] + projections["FG3M"] * projections["MIN"]
        + 2 * projections["FTM"] * projections["MIN"] - projections["FTA"] * projections["MIN"]
        - projections["TOV"] * projections["MIN"] + 3 * td3_pg
    ).clip(lower=0)

    actual = season_base[(season_base["SEASON_YEAR"] == holdout_season_year) & (season_base["GP"] >= 20)]
    actual = actual.set_index("PLAYER_ID")["FANTASY_PPG"]

    result = pd.DataFrame({"predicted": pred_ppg}).join(actual.rename("actual"), how="inner").dropna()
    result["holdout_year"] = holdout_season_year
    result["pred_source_year"] = holdout_season_year - 1
    result = result.reset_index().rename(columns={"index": "PLAYER_ID"})
    return result


if __name__ == "__main__":
    all_results = []
    for holdout_year in [2019, 2020, 2021, 2022, 2023, 2024]:
        print(f"Projecting for holdout {holdout_year}...")
        r = project_for_holdout(holdout_year)
        if r is not None:
            all_results.append(r)
            print(f"  {len(r)} players")

    pooled = pd.concat(all_results, ignore_index=True)
    pooled["residual"] = pooled["actual"] - pooled["predicted"]
    pooled.to_csv(ROOT / "data" / "pooled_residuals.csv", index=False)
    print(f"\nTotal pooled player-season predictions: {len(pooled)} across {pooled['holdout_year'].nunique()} seasons")

    from scipy.stats import spearmanr
    rho, p = spearmanr(pooled["predicted"], pooled["actual"])
    print(f"Sanity check -- pooled Spearman(predicted, actual): {rho:.4f} (p={p:.2e})")
