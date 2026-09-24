"""DARKO-style validation: does the model's belief, captured RIGHT BEFORE
each game (never peeking at that game's own result), predict that game's
real fantasy score? This tests real-time tracking, not just season-ahead
projection -- a genuinely different, finer-grained check than everything
validated so far.

Runs the filter continuously through the ENTIRE 2023-24 season (not just up
to its start) so the state keeps updating game-by-game the way it would in
real use, and captures the PRE-update prediction at every single game.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from kalman_engine import run_filter_all_players

ROOT = Path(__file__).resolve().parent
STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

player_ids = df["PLAYER_ID"].to_numpy()
days = df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = df["MIN"].to_numpy(dtype=float)

pre_predictions = pd.DataFrame(index=df.index)
for stat in STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        fit = json.load(f)
    Q, R, peak_age, slope_up, slope_down = fit["params"]
    is_min = stat == "MIN"
    obs = minutes if is_min else df[stat].to_numpy(dtype=float)
    gain = np.ones_like(minutes) if is_min else minutes
    x0 = float(minutes.sum()) / len(minutes) if is_min else float(obs.sum()) / max(float(gain.sum()), 1.0)

    _, x_pre = run_filter_all_players(
        player_ids, days, age, gain, obs, Q, R, peak_age, slope_up, slope_down, x0, return_pre=True
    )
    pre_predictions[stat] = x_pre
    print(f"  {stat} pre-game predictions computed.")

# Predicted per-game fantasy score, using PROJECTED minutes (pre_predictions["MIN"]),
# never the game's actual minutes -- this is a genuine blind forecast.
predicted_min = pre_predictions["MIN"]
predicted_fantasy = (
    pre_predictions["PTS"] * predicted_min
    + 1.5 * pre_predictions["REB"] * predicted_min
    + 2 * pre_predictions["AST"] * predicted_min
    + 3 * pre_predictions["STL"] * predicted_min
    + 3 * pre_predictions["BLK"] * predicted_min
    + pre_predictions["FG3M"] * predicted_min
    + 2 * pre_predictions["FTM"] * predicted_min
    - pre_predictions["FTA"] * predicted_min
    - pre_predictions["TOV"] * predicted_min
    # TD3 omitted here -- a per-game expected contribution isn't built yet (same known gap as before)
)

df["predicted_fantasy_pregame"] = predicted_fantasy
df["FANTASY_PTS_GAME"] = (
    df["PTS"] + 1.5 * df["REB"] + 2 * df["AST"] + 3 * df["STL"] + 3 * df["BLK"]
    + df["FG3M"] + 2 * df["FTM"] - df["FTA"] - df["TOV"]
    # TD3 omitted from the actual too, for a fair apples-to-apples comparison on this test
)

test = df[df["SEASON"] == "2023-24"].copy()
test = test[test["MIN"] > 0]  # exclude DNPs from this specific game-level test -- predicting "will they play" is a separate question

print(f"\nGame-level test set: {len(test)} real games (2023-24 season, players who played)")

# Baselines: (1) player's rolling last-10-game average fantasy score, (2) season-to-date average.
test = test.sort_values(["PLAYER_ID", "GAME_DATE"])
test["rolling10"] = test.groupby("PLAYER_ID")["FANTASY_PTS_GAME"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=3).mean()
)
test["season_to_date"] = test.groupby("PLAYER_ID")["FANTASY_PTS_GAME"].transform(
    lambda s: s.shift(1).expanding(min_periods=3).mean()
)
test = test.dropna(subset=["rolling10", "season_to_date", "predicted_fantasy_pregame"])

print(f"After requiring enough history for all methods: {len(test)} games\n")


def report(col, label):
    rho, _ = spearmanr(test[col], test["FANTASY_PTS_GAME"])
    r, _ = pearsonr(test[col], test["FANTASY_PTS_GAME"])
    mae = float(np.mean(np.abs(test[col] - test["FANTASY_PTS_GAME"])))
    print(f"{label:40s} Spearman={rho:.4f}  Pearson r={r:.4f}  MAE={mae:.2f}")


report("rolling10", "Baseline: rolling last-10-game average")
report("season_to_date", "Baseline: season-to-date average")
report("predicted_fantasy_pregame", "Kalman composite (blind pre-game)")

test[["PLAYER_ID", "GAME_DATE", "FANTASY_PTS_GAME", "predicted_fantasy_pregame", "rolling10", "season_to_date"]].to_csv(
    ROOT / "data" / "dfs_style_validation_results.csv", index=False
)
