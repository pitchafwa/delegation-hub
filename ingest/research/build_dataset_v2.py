"""v2: multi-year dynasty target, mirroring WRPI/RUPI's 'best-N-of-next-M'
design (§2 of the methodology doc) instead of v1's naive next-season-only
target. Since existing players are evaluated from TODAY forward (not from
draft-entry forward, like a rookie model), the window is a rolling 3-season
lookahead from the evaluation season, not a career-anchored one.

Target = best 2 of the next 3 seasons' fantasy PPG (a season with 0 games,
i.e. out of the league, counts as 0 -- explicit, not a missing value).
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
    base["PTS"] + 1.5 * base["REB"] + 2 * base["AST"] + 3 * base["STL"] + 3 * base["BLK"]
    + base["FG3M"] + 2 * base["FTM"] - base["FTA"] - base["TOV"] + 3 * base["TD3"]
)
base["FANTASY_PPG"] = base["FANTASY_PTS"] / base["GP"].replace(0, np.nan)

adv_cols = ["PLAYER_ID", "SEASON", "USG_PCT", "TS_PCT", "AST_PCT", "REB_PCT",
            "PACE", "PIE", "OFF_RATING", "DEF_RATING", "NET_RATING"]
merged = base.merge(adv[adv_cols], on=["PLAYER_ID", "SEASON"], how="left")

bio_cols = ["PERSON_ID", "POSITION", "HEIGHT", "WEIGHT", "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER", "FROM_YEAR"]
bio_small = bio[bio_cols].rename(columns={"PERSON_ID": "PLAYER_ID"}).drop_duplicates("PLAYER_ID")
merged = merged.merge(bio_small, on="PLAYER_ID", how="left")

merged["DRAFT_NUMBER_NUM"] = pd.to_numeric(merged["DRAFT_NUMBER"], errors="coerce")
merged["UNDRAFTED"] = merged["DRAFT_NUMBER_NUM"].isna().astype(int)
merged["DRAFT_NUMBER_FILLED"] = merged["DRAFT_NUMBER_NUM"].fillna(61)
merged["EXPERIENCE"] = merged["SEASON_YEAR"] - pd.to_numeric(merged["FROM_YEAR"], errors="coerce")
merged["MIN_PG"] = merged["MIN"] / merged["GP"].replace(0, np.nan)

merged = merged.sort_values(["PLAYER_ID", "SEASON_YEAR"]).reset_index(drop=True)
g = merged.groupby("PLAYER_ID")
for col in ["FANTASY_PPG", "USG_PCT", "TS_PCT", "MIN", "PACE", "GP", "TEAM_ABBREVIATION"]:
    merged[f"{col}_PREV"] = g[col].shift(1)
merged["PREV_SEASON_YEAR"] = g["SEASON_YEAR"].shift(1)
merged["HAS_PREV"] = (merged["PREV_SEASON_YEAR"] == merged["SEASON_YEAR"] - 1).astype(int)
merged.loc[merged["HAS_PREV"] == 0, [c for c in merged.columns if c.endswith("_PREV")]] = np.nan
merged["TEAM_CHANGED_OFFSEASON"] = (
    (merged["HAS_PREV"] == 1) & (merged["TEAM_ABBREVIATION"] != merged["TEAM_ABBREVIATION_PREV"])
).astype(int)
merged["TRADED_MIDSEASON"] = (merged["TEAM_COUNT"] > 1).astype(int)

MAX_SEASON_YEAR = merged["SEASON_YEAR"].max()  # 2025

# Lookup table: player_id, season_year -> (fantasy_ppg, existed_that_year)
lookup = merged.set_index(["PLAYER_ID", "SEASON_YEAR"])["FANTASY_PPG"].to_dict()


def future_ppg(player_id, year):
    """0 if the player has no row that season (retired/out of league/not yet
    in the league) -- an explicit, intentional value, not a missing one."""
    return lookup.get((player_id, year), 0.0)


feat_cols = [
    "PLAYER_ID", "PLAYER_NAME", "SEASON", "SEASON_YEAR", "AGE", "EXPERIENCE",
    "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV", "USG_PCT", "USG_PCT_PREV",
    "TS_PCT", "TS_PCT_PREV", "AST_PCT", "REB_PCT", "PACE", "PACE_PREV", "PIE",
    "OFF_RATING", "DEF_RATING", "NET_RATING", "POSITION", "HEIGHT", "WEIGHT",
    "DRAFT_NUMBER_FILLED", "UNDRAFTED", "TEAM_CHANGED_OFFSEASON", "TRADED_MIDSEASON",
]
current = merged[feat_cols].copy()
current["GP_full"] = merged["GP"]
current["MIN_PG_full"] = merged["MIN_PG"]

rows = []
for _, row in current.iterrows():
    pid, yr = row["PLAYER_ID"], row["SEASON_YEAR"]
    future_years = [yr + 1, yr + 2, yr + 3]
    if max(future_years) > MAX_SEASON_YEAR:
        continue  # can't resolve a real 3-year-forward outcome yet
    vals = [future_ppg(pid, fy) for fy in future_years]
    best2 = sum(sorted(vals, reverse=True)[:2]) / 2.0
    row_dict = row.to_dict()
    row_dict["TARGET_BEST2OF3"] = best2
    row_dict["TARGET_Y1"], row_dict["TARGET_Y2"], row_dict["TARGET_Y3"] = vals
    row_dict["LEFT_LEAGUE_ANY_OF_3"] = int(any(v == 0.0 for v in vals))
    rows.append(row_dict)

dataset = pd.DataFrame(rows)
dataset = dataset[(dataset["GP"] >= 15) & (dataset["MIN_PG"] >= 8)].reset_index(drop=True)

out_path = ROOT / "data" / "valuation_dataset_v2.csv"
dataset.to_csv(out_path, index=False)

# Current-season players (2025-26) are prediction inputs -- can't resolve a
# 3-year target yet (2026-27, 2027-28, 2028-29 haven't happened).
current_for_prediction = current[current["SEASON_YEAR"] == MAX_SEASON_YEAR].copy()
current_for_prediction.to_csv(ROOT / "data" / "current_season_for_prediction_v2.csv", index=False)

print(f"v2 trainable dataset rows: {len(dataset)}")
print(f"Season-transition years covered: {sorted(dataset['SEASON_YEAR'].unique())}")
print(f"Rows where player left the league in at least one of the 3 future seasons: "
      f"{dataset['LEFT_LEAGUE_ANY_OF_3'].sum()} ({dataset['LEFT_LEAGUE_ANY_OF_3'].mean():.1%})")
print(f"Current-season (2025-26) prediction rows: {len(current_for_prediction)}")
print(dataset.head(5)[["PLAYER_NAME", "SEASON", "AGE", "FANTASY_PPG", "TARGET_Y1", "TARGET_Y2", "TARGET_Y3", "TARGET_BEST2OF3"]])
