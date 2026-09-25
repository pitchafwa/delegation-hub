"""Unify the two real game-level sources into one schema:
- GitHub bulk file (NocturneBear/NBA-Data-2010-2024): 2010-11 through 2023-24
- nba_api LeagueGameLog: 2024-25, 2025-26 (the 2 seasons the bulk file misses)

Then compute per-game triple-doubles (>=10 in 3 of PTS/REB/AST/STL/BLK) and
VALIDATE the summed count against the official season-total TD3 already
pulled from LeagueDashPlayerStats -- a real check, not an assumed-correct
derivation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GL_DIR = ROOT / "data" / "game_logs"

bulk_frames = []
for part in ["part_1", "part_2", "part_3"]:
    df = pd.read_csv(GL_DIR / f"regular_season_box_scores_2010_2024_{part}.csv")
    bulk_frames.append(df)
bulk = pd.concat(bulk_frames, ignore_index=True)

def parse_minutes(v):
    """The bulk source stores minutes as 'M:SS' strings (e.g. '34:21'), not
    decimal minutes -- a plain pd.to_numeric silently NaNs almost every
    played game (caught only because the Kalman filter came back all-NaN)."""
    if pd.isna(v):
        return 0.0
    s = str(v)
    if ":" in s:
        m, sec = s.split(":")
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


bulk_std = pd.DataFrame({
    "PLAYER_ID": bulk["personId"],
    "PLAYER_NAME": bulk["personName"],
    "SEASON": bulk["season_year"],
    "GAME_ID": bulk["gameId"],
    "GAME_DATE": bulk["game_date"],
    "TEAM": bulk["teamTricode"],
    "MIN": bulk["minutes"].apply(parse_minutes),
    "PTS": bulk["points"], "REB": bulk["reboundsTotal"], "AST": bulk["assists"],
    "STL": bulk["steals"], "BLK": bulk["blocks"], "TOV": bulk["turnovers"],
    "FG3M": bulk["threePointersMade"], "FTM": bulk["freeThrowsMade"], "FTA": bulk["freeThrowsAttempted"],
})

recent_frames = []
for _f in sorted(GL_DIR.glob("nba_api_*.csv")):          # every nba_api season file (the current season is added by pull_recent_gamelogs.py)
    season = _f.stem.replace("nba_api_", "")
    df = pd.read_csv(_f)
    recent_frames.append(pd.DataFrame({
        "PLAYER_ID": df["PLAYER_ID"], "PLAYER_NAME": df["PLAYER_NAME"], "SEASON": season,
        "GAME_ID": df["GAME_ID"], "GAME_DATE": df["GAME_DATE"], "TEAM": df["TEAM_ABBREVIATION"],
        "MIN": df["MIN"], "PTS": df["PTS"], "REB": df["REB"], "AST": df["AST"],
        "STL": df["STL"], "BLK": df["BLK"], "TOV": df["TOV"],
        "FG3M": df["FG3M"], "FTM": df["FTM"], "FTA": df["FTA"],
    }))
recent = pd.concat(recent_frames, ignore_index=True)

games = pd.concat([bulk_std, recent], ignore_index=True)
for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA"]:
    games[c] = pd.to_numeric(games[c], errors="coerce").fillna(0)

# Triple-double: double digits in >=3 of the 5 major categories (NBA's own rule).
cats_double_digit = (
    (games["PTS"] >= 10).astype(int) + (games["REB"] >= 10).astype(int)
    + (games["AST"] >= 10).astype(int) + (games["STL"] >= 10).astype(int)
    + (games["BLK"] >= 10).astype(int)
)
games["TD3"] = (cats_double_digit >= 3).astype(int)

games.to_csv(ROOT / "data" / "game_logs_unified.csv", index=False)
print(f"Unified game log: {len(games)} rows, seasons {sorted(games['SEASON'].unique())}")

# --- Validate computed TD3 against the official season-total TD3 already pulled ---
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
computed_td3 = games.groupby(["PLAYER_ID", "SEASON"])["TD3"].sum().reset_index()
computed_td3.columns = ["PLAYER_ID", "SEASON", "TD3_COMPUTED"]
check = season_base[["PLAYER_ID", "SEASON", "PLAYER_NAME", "TD3"]].merge(
    computed_td3, on=["PLAYER_ID", "SEASON"], how="inner"
)
check["diff"] = check["TD3_COMPUTED"] - check["TD3"]
mismatches = check[check["diff"] != 0]
print(f"\nTD3 validation: {len(check)} player-seasons compared, {len(mismatches)} mismatches "
      f"({len(mismatches)/max(len(check),1):.2%})")
if len(mismatches):
    print(mismatches.sort_values("diff", key=abs, ascending=False).head(15).to_string(index=False))
