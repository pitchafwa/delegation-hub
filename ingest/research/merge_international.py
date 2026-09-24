"""Merge real international/non-college prospects into the main rookie
dataset. The hard problem: BPM (college) and Game Score (international,
the best available proxy given no BPM-equivalent exists for non-NCAA
leagues) are on completely different numeric scales -- can't drop Game
Score into the "bpm" ramp-bounds slot and expect it to mean anything.

Fix: percentile-rank each within its OWN population (college BPM vs its
college peers; international Game Score vs its international peers), then
use that shared 0-1 percentile as the unified "talent" input -- the exact
same pattern already used elsewhere in this pipeline for cross-position
combine percentiles (comparability via rank, not raw units).

Other real, disclosed approximations for international rows:
- rec_filled = 0 (no US HS recruiting rank applies -- same treatment as a
  real unranked domestic recruit, not a new judgment call)
- exp_numeric = population median (no Fr/So/Jr/Sr concept for pro leagues)
- breakout_age = NaN / never_broke_out = 1 (would need full multi-year
  international trajectories to compute for real, which weren't pulled in
  this first pass -- a disclosed limitation, not faked)
- combine agility = real, when present (linked via real NBA PLAYER_ID,
  works identically regardless of pathway)
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

main = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")
intl = pd.read_csv(ROOT / "data" / "international_features.csv")
print(f"College rows: {len(main)}, international rows: {len(intl)}")

# real cross-check before merging: no PERSON_ID should appear in both (would
# mean a player got double-counted via both a college AND international
# match -- a real identity bug if it happens, not silently ignored)
overlap = set(main["PLAYER_ID"].dropna()) & set(intl["PERSON_ID"].dropna())
if overlap:
    print(f"WARNING: {len(overlap)} players matched via BOTH college and international -- {overlap}")
else:
    print("Identity check OK: no player matched via both pathways")

# --- real outcome target ---
target = pd.read_csv(ROOT / "data" / "target_window_test.csv")
intl = intl.merge(target[["PLAYER_ID", "age_22_29_best3"]], left_on="PERSON_ID", right_on="PLAYER_ID", how="left")
intl = intl.drop(columns=["PLAYER_ID"])  # redundant copy of PERSON_ID from the merge key -- PERSON_ID gets renamed to PLAYER_ID later
print(f"Real outcome (age_22_29_best3) available for {intl['age_22_29_best3'].notna().sum()} / {len(intl)} international players")

# --- real draft-day age ---
bd = pd.read_csv(ROOT / "data" / "birthdates_all.csv")
bd["BIRTHDATE"] = pd.to_datetime(bd["BIRTHDATE"])
intl = intl.merge(bd.rename(columns={"PERSON_ID": "PERSON_ID_bd"}), left_on="PERSON_ID", right_on="PERSON_ID_bd", how="left")
draft_day = pd.to_datetime(intl["real_draft_year"].astype("Int64").astype(str) + "-06-26", errors="coerce")
intl["draft_age"] = (draft_day - intl["BIRTHDATE"]).dt.days / 365.25
implausible = ~intl["draft_age"].between(16, 28)
if implausible.sum():
    print(f"Nulling {implausible.sum()} implausible international draft_age rows")
    intl.loc[implausible, "draft_age"] = np.nan
print(f"Real draft_age computed for {intl['draft_age'].notna().sum()} / {len(intl)} international players")

# --- real combine data (already linked by real NBA PLAYER_ID regardless of pathway) ---
combine = pd.read_csv(ROOT / "data" / "combine_all.csv")
combine_cols = ["PLAYER_ID", "HEIGHT_WO_SHOES", "WINGSPAN", "STANDING_REACH", "WEIGHT",
                 "STANDING_VERTICAL_LEAP", "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"]
intl = intl.merge(combine[combine_cols], left_on="PERSON_ID", right_on="PLAYER_ID", how="left")
intl = intl.drop(columns=["PLAYER_ID"])  # redundant copy of PERSON_ID again
print(f"Real combine data for {intl['WINGSPAN'].notna().sum()} / {len(intl)} international players")

# --- rename to match the college schema ---
intl = intl.rename(columns={
    "PERSON_ID": "PLAYER_ID", "real_name": "player", "Team": "team",
    "FG%": "fg_pct", "3P%": "three_pct", "FT%": "ft_pct",
    "TRB": "rpg", "AST": "apg", "PTS": "ppg", "STL": "spg", "BLK": "bpg", "TOV": "tov",
    "MP": "mpg", "G": "g",
})
intl["rec"] = 0.0  # no US HS recruiting rank applies -- treated the same as a real unranked domestic recruit
intl["exp"] = np.nan  # no Fr/So/Jr/Sr concept for pro leagues -- median-filled downstream like any missing exp
intl["pos"] = np.nan  # not scraped in this pass -- position group unavailable, a disclosed gap
intl["breakout_age"] = np.nan  # would need full multi-year international trajectories -- not pulled in this pass
intl["never_broke_out"] = 1
intl["n_college_seasons"] = np.nan
intl["data_source"] = "international"

main["data_source"] = "college"

combined = pd.concat([main, intl], ignore_index=True, sort=False)

# --- unified talent percentile: bpm within college, game_score within international ---
combined["talent_pctile"] = np.nan
c_mask = combined["data_source"] == "college"
i_mask = combined["data_source"] == "international"
combined.loc[c_mask, "talent_pctile"] = combined.loc[c_mask, "bpm"].rank(pct=True)
combined.loc[i_mask, "talent_pctile"] = combined.loc[i_mask, "game_score"].rank(pct=True)
print(f"\ntalent_pctile coverage: college {combined.loc[c_mask,'talent_pctile'].notna().sum()}/{c_mask.sum()}, "
      f"international {combined.loc[i_mask,'talent_pctile'].notna().sum()}/{i_mask.sum()}")

combined.to_csv(ROOT / "data" / "rookie_model_dataset_unified.csv", index=False, encoding="utf-8")
print(f"\nSaved {len(combined)} total rows to rookie_model_dataset_unified.csv")

# real, quick face-validity spot check on the two explicit acceptance-test players
for name in ["Victor Wembanyama", "Luka Dončić"]:
    row = combined[combined["player"] == name]
    if row.empty:
        print(f"\n{name}: NOT FOUND in unified dataset")
    else:
        r = row.iloc[0]
        print(f"\n{name}: talent_pctile={r['talent_pctile']:.3f}, draft_age={r['draft_age']}, "
              f"real_draft_number={r['real_draft_number']}, game_score={r.get('game_score')}")
