"""Breakout age: the age at which a prospect FIRST posted a real, meaningful
(min. 20 mpg to exclude garbage-time flukes) BPM >= 6.0 college season --
using the full multi-year college trajectory (via torvik's stable per-player
'id'), not just their final season. Real prospect literature (Vashro-style
breakout-age work) finds developmental speed matters beyond current-season
production alone.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
BPM_THRESHOLD = 6.0
MIN_MPG = 20.0

college = pd.read_csv(ROOT / "data" / "college_player_season_2008_2025.csv", low_memory=False)
links = pd.read_csv(ROOT / "data" / "college_to_nba_links.csv")
rookie_df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")

# recover each linked player's stable torvik 'id' via their final college season row
final_rows = college[["player", "year", "id"]].rename(columns={"player": "college_player", "year": "college_last_year"})
links_with_id = links.merge(final_rows, on=["college_player", "college_last_year"], how="left")
links_with_id = links_with_id.dropna(subset=["id"]).drop_duplicates(subset=["PLAYER_ID"])
print(f"Linked {len(links_with_id)} / {len(links)} players to a stable torvik id")

# pull full multi-year trajectory for each linked player
traj = college.merge(links_with_id[["id", "PLAYER_ID"]], on="id", how="inner")

# real birthdates already pulled via nba_api
bd = pd.read_csv(ROOT / "data" / "birthdates_all.csv").rename(columns={"PERSON_ID": "PLAYER_ID"})
bd["BIRTHDATE"] = pd.to_datetime(bd["BIRTHDATE"])
traj = traj.merge(bd, on="PLAYER_ID", how="left")

# approximate age at each college season as age on Jan 1 of the season's listed year
season_date = pd.to_datetime(traj["year"].astype("Int64").astype(str) + "-01-01", errors="coerce")
traj["age_at_season"] = (season_date - traj["BIRTHDATE"]).dt.days / 365.25

qualifying = traj[(traj["bpm"] >= BPM_THRESHOLD) & (traj["mpg"] >= MIN_MPG) & traj["age_at_season"].between(15, 26)]
breakout = qualifying.groupby("PLAYER_ID")["age_at_season"].min().rename("breakout_age")
print(f"{len(breakout)} / {rookie_df['PLAYER_ID'].nunique()} rookie-dataset players ever broke out (BPM>={BPM_THRESHOLD}, {MIN_MPG}+ mpg)")

rookie_df = rookie_df.drop(columns=[c for c in ["breakout_age", "never_broke_out"] if c in rookie_df.columns])
rookie_df = rookie_df.merge(breakout, on="PLAYER_ID", how="left")
rookie_df["never_broke_out"] = rookie_df["breakout_age"].isna().astype(int)
# never-broke-out is a real, informative signal (worse prospect), not missing
# data -- fill with a fixed late value worse than any real observed breakout
worst_case = rookie_df["breakout_age"].max() + 1
rookie_df["breakout_age_filled"] = rookie_df["breakout_age"].fillna(worst_case)

print(rookie_df["breakout_age"].describe())
rookie_df.to_csv(ROOT / "data" / "rookie_model_dataset.csv", index=False)
print("Saved breakout_age / breakout_age_filled / never_broke_out columns.")
