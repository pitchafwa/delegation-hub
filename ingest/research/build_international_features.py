"""Turn raw international-player season rows (pulled via
pull_international_data.py) into real, usable pre-draft features.

The core problem: Basketball-Reference's international tables have only
basic box-score counting stats -- no BPM/PORPAG equivalent exists for
non-NCAA leagues. Rather than assume box-score rates alone are enough
(the college model already found they badly underperform BPM), compute a
real, established, PACE-INDEPENDENT-ISH composite from what's actually
available: John Hollinger's Game Score, a standard, documented, simple box
-score formula computable without team pace/opponent data (unlike the full
PER, which needs team totals we don't have for every league). This is a
disclosed, honest APPROXIMATION -- its real predictive value gets tested
the same way every other feature in this project has been, not assumed.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

raw = pd.read_csv(ROOT / "data" / "international_player_seasons.csv")
print(f"Raw rows: {len(raw)}, real players: {raw['PERSON_ID'].nunique()}")

NUM_COLS = ["G", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%",
            "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
for c in NUM_COLS:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

# real Hollinger Game Score, per-game (all inputs are already per-game rates
# in this table, so this yields a per-game GmSc directly)
raw["game_score"] = (
    raw["PTS"] + 0.4 * raw["FG"] - 0.7 * raw["FGA"] - 0.4 * (raw["FTA"] - raw["FT"])
    + 0.7 * raw["ORB"] + 0.3 * raw["DRB"] + raw["STL"] + 0.7 * raw["AST"]
    + 0.7 * raw["BLK"] - 0.4 * raw["PF"] - raw["TOV"]
)

# final pre-draft season = the row with the most real games played, among
# whichever season_end_year is each player's own most recent real row --
# avoids picking a tiny-sample tournament row over their real primary
# domestic-league season in the same year.
raw["_is_final_year"] = raw.groupby("PERSON_ID")["season_end_year"].transform("max") == raw["season_end_year"]
final = raw[raw["_is_final_year"]].copy()
final = final.sort_values("G", ascending=False).drop_duplicates(subset=["PERSON_ID"], keep="first")

print(f"Final-season rows (one per real player): {len(final)}")
print(final[["real_name", "Season", "League", "G", "MP", "PTS", "game_score"]].head(15).to_string())

out_cols = ["PERSON_ID", "real_name", "real_draft_year", "real_draft_number", "bbref_slug",
            "Season", "Team", "League", "season_end_year", "G", "MP",
            "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PTS", "game_score"]
final[out_cols].to_csv(ROOT / "data" / "international_features.csv", index=False, encoding="utf-8")
print(f"\nSaved {len(final)} rows to international_features.csv")

print("\ngame_score distribution:")
print(final["game_score"].describe())
