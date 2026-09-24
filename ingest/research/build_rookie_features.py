"""Assemble the real, joined rookie-model dataset: pre-NBA features (college
final season, combine measurements, draft capital) + both real targets.

Output A target: best-3-of-ages-22-29 fantasy PPG (the long-term prospect
grade, already validated empirically against real draft classes).
Output B target: real rookie-season per-stat rates for each of the 9
Kalman-filtered stats + MIN -- what actually feeds the existing player-value
engine's prior, in the same units it already uses.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA"]  # rates are per-minute; MIN handled separately

# --- Pre-NBA features: college (final season per player) ---
college = pd.read_csv(ROOT / "data" / "college_player_season_2008_2025.csv", low_memory=False)
college_final = college.sort_values("year").groupby("id").tail(1).copy()
college_final = college_final.rename(columns={"id": "torvik_id"})

# career arc: did production improve from an earlier season to the final one?
# (a cheap version of "breakout age" -- more seasons = real trend available)
college_multi = college.groupby("id").agg(n_college_seasons=("year", "nunique"))
college_final = college_final.merge(college_multi, left_on="torvik_id", right_index=True, how="left")

# --- Identity links: torvik college player -> real NBA PLAYER_ID ---
links = pd.read_csv(ROOT / "data" / "college_to_nba_links.csv")
links = links.rename(columns={"college_player": "player_name_link"})

# re-derive torvik_id via name+year match (links.csv didn't keep torvik's own id column)
college_final["norm_name"] = college_final["player"].str.lower().str.strip()
links["norm_name"] = links["player_name_link"].str.lower().str.strip()
merged = college_final.merge(
    links[["norm_name", "college_last_year", "PLAYER_ID", "real_draft_year", "real_draft_number"]],
    left_on=["norm_name", "year"], right_on=["norm_name", "college_last_year"], how="inner",
)
print(f"College final-season rows matched to real NBA players: {len(merged)}")

# --- Combine measurements ---
combine = pd.read_csv(ROOT / "data" / "combine_all.csv")
combine_cols = ["PLAYER_ID", "HEIGHT_WO_SHOES", "WINGSPAN", "STANDING_REACH", "WEIGHT",
                 "STANDING_VERTICAL_LEAP", "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"]
merged = merged.merge(combine[combine_cols], on="PLAYER_ID", how="left")
print(f"Rows with real combine data: {merged['WINGSPAN'].notna().sum()} of {len(merged)}")

# --- Target A: best-3-of-ages-22-29 (already computed and validated) ---
target_a = pd.read_csv(ROOT / "data" / "target_window_test.csv")
merged = merged.merge(target_a[["PLAYER_ID", "age_22_29_best3"]], on="PLAYER_ID", how="left")

# --- Target B: real rookie-season per-stat rates ---
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
adv = pd.read_csv(ROOT / "data" / "player_season_advanced.csv")
adv["SEASON_YEAR"] = adv["SEASON"].apply(lambda s: int(s.split("-")[0]))

rookie_rows = season_base.merge(
    merged[["PLAYER_ID", "real_draft_year"]].drop_duplicates(), on="PLAYER_ID", how="inner"
)
rookie_rows = rookie_rows[rookie_rows["SEASON_YEAR"] == rookie_rows["real_draft_year"]]
for stat in KALMAN_STATS:
    rookie_rows[f"rookie_{stat}_per_min"] = rookie_rows[stat] / rookie_rows["MIN"].replace(0, np.nan)
rookie_rows["rookie_MIN_per_game"] = rookie_rows["MIN"] / rookie_rows["GP"].replace(0, np.nan)
rookie_rows["rookie_GP"] = rookie_rows["GP"]

target_b_cols = ["PLAYER_ID"] + [f"rookie_{s}_per_min" for s in KALMAN_STATS] + ["rookie_MIN_per_game", "rookie_GP"]
merged = merged.merge(rookie_rows[target_b_cols], on="PLAYER_ID", how="left")

print(f"\nRows with a resolvable Output A target: {merged['age_22_29_best3'].notna().sum()}")
print(f"Rows with a resolvable Output B target (real rookie season played): {merged['rookie_MIN_per_game'].notna().sum()}")

merged.to_csv(ROOT / "data" / "rookie_model_dataset.csv", index=False)
print(f"\nSaved {len(merged)} total rows to rookie_model_dataset.csv")
print(f"Columns: {merged.columns.tolist()}")
