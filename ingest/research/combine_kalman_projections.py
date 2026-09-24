"""Combine all 9 per-stat Kalman filters + MIN into one composite fantasy
projection, and test it fairly against (a) the existing frozen Layer A
model and (b) naive persistence, all on the SAME single-season 2023-24
holdout target -- an apples-to-apples comparison, not a different metric
dressed up to look comparable.

TD3 (triple-doubles) isn't Kalman-filtered yet -- flagged, not hidden: this
uses each player's last known season's TD3 rate as a placeholder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kalman_engine import run_filter_all_players
from fit_talent_model import unpack_and_score

ROOT = Path(__file__).resolve().parent
# STL and TD3 excluded from the Kalman machinery: both tested WORSE than
# simple persistence with the final widened-bounds fits (STL: 0.635 vs 0.651
# naive; TD3: 0.347 vs 0.492 naive). STL uses a flat career per-minute rate
# below; TD3 keeps its season-carried-forward placeholder.
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]

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

projections = pd.DataFrame(index=last_game_date.index)

for stat in KALMAN_STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        fit = json.load(f)
    Q, R, peak_age, slope_up, slope_down = fit["params"]
    is_min = stat == "MIN"
    obs = minutes if is_min else fit_df[stat].to_numpy(dtype=float)
    gain = np.ones_like(minutes) if is_min else minutes
    x0 = float(minutes.sum()) / len(minutes) if is_min else float(obs.sum()) / max(float(gain.sum()), 1.0)

    posterior = run_filter_all_players(player_ids, days, age, gain, obs, Q, R, peak_age, slope_up, slope_down, x0)
    fit_df["_post"] = posterior
    end_state = fit_df.loc[last_game_idx].set_index("PLAYER_ID")["_post"]

    diff = last_game_age - peak_age
    daily_slope = np.where(diff <= 0, slope_up, slope_down) / 365.0
    projected = end_state + daily_slope * days_to_holdout
    projections[stat] = projected
    print(f"  {stat} projected.")

# STL: flat career per-minute rate, no age evolution (tested, no real signal found).
stl_rate = fit_df.groupby("PLAYER_ID").apply(
    lambda d: d["STL"].sum() / max(d["MIN"].sum(), 1.0), include_groups=False
)
projections["STL"] = stl_rate.reindex(projections.index).fillna(0.0)

# TD3 placeholder: last known season's TD3-per-game rate (simple carry-forward, not Kalman-filtered).
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
last_season = season_base[season_base["SEASON_YEAR"] == 2022].set_index("PLAYER_ID")
projections["TD3_PG"] = (last_season["TD3"] / last_season["GP"].replace(0, np.nan)).reindex(projections.index).fillna(0.0)

# Composite projected fantasy PPG: rate * minutes for rate stats, MIN itself for minutes.
proj_ppg = (
    projections["PTS"] * projections["MIN"]
    + 1.5 * projections["REB"] * projections["MIN"]
    + 2 * projections["AST"] * projections["MIN"]
    + 3 * projections["STL"] * projections["MIN"]
    + 3 * projections["BLK"] * projections["MIN"]
    + projections["FG3M"] * projections["MIN"]
    + 2 * projections["FTM"] * projections["MIN"]
    - projections["FTA"] * projections["MIN"]
    - projections["TOV"] * projections["MIN"]
    + 3 * projections["TD3_PG"]
)

# Real 2023-24 outcome.
holdout_df["FANTASY_PTS"] = (
    holdout_df["PTS"] + 1.5 * holdout_df["REB"] + 2 * holdout_df["AST"] + 3 * holdout_df["STL"]
    + 3 * holdout_df["BLK"] + holdout_df["FG3M"] + 2 * holdout_df["FTM"] - holdout_df["FTA"]
    - holdout_df["TOV"] + 3 * holdout_df["TD3"]
)
actual = holdout_df.groupby("PLAYER_ID").agg(fantasy_total=("FANTASY_PTS", "sum"), gp=("FANTASY_PTS", "size"))
actual["actual_ppg"] = actual["fantasy_total"] / actual["gp"]
actual = actual[actual["gp"] >= 20]

compare = pd.DataFrame({"kalman_composite": proj_ppg}).join(actual["actual_ppg"], how="inner").dropna()

# Baseline 1: naive persistence (2022-23 real fantasy PPG).
prior_season = season_base[season_base["SEASON_YEAR"] == 2022].copy()
prior_season["FANTASY_PTS"] = (
    prior_season["PTS"] + 1.5 * prior_season["REB"] + 2 * prior_season["AST"] + 3 * prior_season["STL"]
    + 3 * prior_season["BLK"] + prior_season["FG3M"] + 2 * prior_season["FTM"] - prior_season["FTA"]
    - prior_season["TOV"] + 3 * prior_season["TD3"]
)
prior_season["FANTASY_PPG"] = prior_season["FANTASY_PTS"] / prior_season["GP"].replace(0, np.nan)
naive = prior_season.set_index("PLAYER_ID")["FANTASY_PPG"]
compare["naive"] = naive.reindex(compare.index)

# Baseline 2: the existing frozen Layer A model, applied to the same players' pre-2023-24 features.
v2 = pd.read_csv(ROOT / "data" / "valuation_dataset_v2.csv")
layer_a_rows = v2[v2["SEASON_YEAR"] == 2022].copy()
for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
    layer_a_rows[col] = layer_a_rows[col].fillna(layer_a_rows[col.replace("_PREV", "")])
with open(ROOT / "data" / "layer_a_model.frozen.json") as f:
    la_params = np.array(json.load(f)["params_raw"])
layer_a_rows["layer_a_score"] = [
    unpack_and_score(la_params, layer_a_rows.iloc[[i]])[0] for i in range(len(layer_a_rows))
]
layer_a_scores = layer_a_rows.set_index("PLAYER_ID")["layer_a_score"]
compare["layer_a"] = layer_a_scores.reindex(compare.index)

compare = compare.dropna()
print(f"\nCompared on {len(compare)} real players, 2023-24 holdout season.\n")


def report(col, label):
    rho, _ = spearmanr(compare[col], compare["actual_ppg"])
    r, _ = pearsonr(compare[col], compare["actual_ppg"])
    mae = float(np.mean(np.abs(compare[col] - compare["actual_ppg"])))
    print(f"{label:34s} Spearman={rho:.4f}  Pearson r={r:.4f}  MAE={mae:.2f}")


report("naive", "Naive persistence (last season)")
report("layer_a", "Existing frozen Layer A model")
report("kalman_composite", "NEW: Kalman composite (9 stats)")

compare.to_csv(ROOT / "data" / "kalman_composite_vs_layer_a.csv")
