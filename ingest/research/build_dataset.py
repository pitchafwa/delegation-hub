"""Build the supervised backtesting dataset:
  features from a player's season t  ->  target = fantasy PPG in season t+1

Fantasy formula is Tommy's league's REAL scoring settings (validated in
validate_scoring2.py against 94 real player-days, 0 mismatches):
  PTS*1 + REB*1.5 + AST*2 + STL*3 + BLK*3 + 3PM*1 + FTM*1 - FTMI*1 - TOV*1 + TD3*3
  (FTM - FTMI = FTM - (FTA-FTM) = 2*FTM - FTA)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
adv = pd.read_csv(ROOT / "data" / "player_season_advanced.csv")
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")


def season_start_year(s: str) -> int:
    return int(s.split("-")[0])


base["SEASON_YEAR"] = base["SEASON"].apply(season_start_year)

base["FANTASY_PTS"] = (
    base["PTS"]
    + 1.5 * base["REB"]
    + 2 * base["AST"]
    + 3 * base["STL"]
    + 3 * base["BLK"]
    + base["FG3M"]
    + 2 * base["FTM"]
    - base["FTA"]
    - base["TOV"]
    + 3 * base["TD3"]
)
base["FANTASY_PPG"] = base["FANTASY_PTS"] / base["GP"].replace(0, np.nan)

adv_cols = [
    "PLAYER_ID", "SEASON", "USG_PCT", "TS_PCT", "AST_PCT", "REB_PCT",
    "PACE", "PIE", "OFF_RATING", "DEF_RATING", "NET_RATING",
]
merged = base.merge(adv[adv_cols], on=["PLAYER_ID", "SEASON"], how="left")

bio_cols = [
    "PERSON_ID", "POSITION", "HEIGHT", "WEIGHT", "DRAFT_YEAR",
    "DRAFT_ROUND", "DRAFT_NUMBER", "FROM_YEAR",
]
bio_small = bio[bio_cols].rename(columns={"PERSON_ID": "PLAYER_ID"})
# a player can appear once in bio; guard against any dup ids
bio_small = bio_small.drop_duplicates("PLAYER_ID")
merged = merged.merge(bio_small, on="PLAYER_ID", how="left")


def parse_draft_number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan  # undrafted


merged["DRAFT_NUMBER_NUM"] = merged["DRAFT_NUMBER"].apply(parse_draft_number)
merged["UNDRAFTED"] = merged["DRAFT_NUMBER_NUM"].isna().astype(int)
merged["DRAFT_NUMBER_FILLED"] = merged["DRAFT_NUMBER_NUM"].fillna(61)  # worse than any real pick

merged["EXPERIENCE"] = merged["SEASON_YEAR"] - pd.to_numeric(merged["FROM_YEAR"], errors="coerce")

merged = merged.sort_values(["PLAYER_ID", "SEASON_YEAR"]).reset_index(drop=True)

# Prior-season (t-1) lag features, matched by player.
g = merged.groupby("PLAYER_ID")
for col in ["FANTASY_PPG", "USG_PCT", "TS_PCT", "MIN", "PACE", "GP", "TEAM_ABBREVIATION"]:
    merged[f"{col}_PREV"] = g[col].shift(1)
merged["PREV_SEASON_YEAR"] = g["SEASON_YEAR"].shift(1)
merged["HAS_PREV"] = (merged["PREV_SEASON_YEAR"] == merged["SEASON_YEAR"] - 1).astype(int)
merged.loc[merged["HAS_PREV"] == 0, [c for c in merged.columns if c.endswith("_PREV")]] = np.nan

merged["TEAM_CHANGED_OFFSEASON"] = (
    (merged["HAS_PREV"] == 1)
    & (merged["TEAM_ABBREVIATION"] != merged["TEAM_ABBREVIATION_PREV"])
).astype(int)
merged["TRADED_MIDSEASON"] = (merged["TEAM_COUNT"] > 1).astype(int)

merged["MIN_PG"] = merged["MIN"] / merged["GP"].replace(0, np.nan)

# --- Build the t -> t+1 supervised table ---
feat_cols = [
    "PLAYER_ID", "PLAYER_NAME", "SEASON", "SEASON_YEAR", "AGE", "EXPERIENCE",
    "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV", "USG_PCT", "USG_PCT_PREV",
    "TS_PCT", "TS_PCT_PREV", "AST_PCT", "REB_PCT", "PACE", "PACE_PREV", "PIE",
    "OFF_RATING", "DEF_RATING", "NET_RATING", "POSITION", "HEIGHT", "WEIGHT",
    "DRAFT_NUMBER_FILLED", "UNDRAFTED", "TEAM_CHANGED_OFFSEASON",
    "TRADED_MIDSEASON", "TEAM_ABBREVIATION",
]
current = merged[feat_cols].copy()

target = merged[["PLAYER_ID", "SEASON_YEAR", "FANTASY_PPG", "GP"]].rename(
    columns={"SEASON_YEAR": "TARGET_YEAR", "FANTASY_PPG": "TARGET_FANTASY_PPG", "GP": "TARGET_GP"}
)
current["TARGET_YEAR"] = current["SEASON_YEAR"] + 1

dataset = current.merge(target, on=["PLAYER_ID", "TARGET_YEAR"], how="left")

MAX_SEASON_YEAR = merged["SEASON_YEAR"].max()  # 2025 (2025-26) -- no 2026-27 data exists yet

# Rows whose TARGET_YEAR doesn't exist yet at all (this season's players) have
# no real answer to check against -- they are prediction INPUTS, not trainable
# rows. Split them out before doing the "missing = left league" fill, otherwise
# every current player gets wrongly scored as having left the league.
current_for_prediction = dataset[dataset["SEASON_YEAR"] == MAX_SEASON_YEAR].copy()
trainable = dataset[dataset["SEASON_YEAR"] < MAX_SEASON_YEAR].copy()

# Player not found next season at all (retired/out of league/missed the pull
# entirely) -> treat as washed out to 0, not a missing value, so the model
# isn't blind to real value collapsing to zero. Flag it so we can inspect.
trainable["LEFT_LEAGUE_NEXT_SEASON"] = trainable["TARGET_FANTASY_PPG"].isna().astype(int)
trainable["TARGET_FANTASY_PPG"] = trainable["TARGET_FANTASY_PPG"].fillna(0.0)

# Only keep rows with a real current-season sample size to learn from.
dataset = trainable[(trainable["GP"] >= 15) & (trainable["MIN_PG"] >= 8)].reset_index(drop=True)

out_path = ROOT / "data" / "valuation_dataset.csv"
dataset.to_csv(out_path, index=False)

pred_path = ROOT / "data" / "current_season_for_prediction.csv"
current_for_prediction.drop(columns=["TARGET_FANTASY_PPG", "TARGET_GP"]).to_csv(pred_path, index=False)
print(f"Current-season (2025-26) players held out as prediction inputs: {len(current_for_prediction)}")

print(f"Dataset rows: {len(dataset)}")
print(f"Season-transition years covered: {sorted(dataset['SEASON_YEAR'].unique())}")
print(f"Rows where player left the league entirely next season: {dataset['LEFT_LEAGUE_NEXT_SEASON'].sum()} "
      f"({dataset['LEFT_LEAGUE_NEXT_SEASON'].mean():.1%})")
print(f"Rows with a real t-1 lag available (HAS_PREV): {dataset['FANTASY_PPG_PREV'].notna().sum()}")
print(dataset.head(5)[["PLAYER_NAME", "SEASON", "AGE", "FANTASY_PPG", "TARGET_FANTASY_PPG"]])
