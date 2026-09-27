"""Pull team-level game logs (win/loss, date, opponent) for every season 2010-11..2025-26. Used for standings-by-date / playoff-elimination status in the tanking study.
Writes data/tank/team_games.csv."""
import sys
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
import time
sys.stdout.reconfigure(encoding="utf-8")
seasons = [f"{y}-{str(y+1)[2:]}" for y in range(2010, 2026)]
rows = []
for s in seasons:
    for tries in range(3):
        try:
            df = leaguegamelog.LeagueGameLog(season=s, season_type_all_star="Regular Season", player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
            break
        except Exception as e:
            print(s, "retry", e); time.sleep(3)
    else:
        continue
    df["season"] = s
    rows.append(df[["season", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]])
    print(s, len(df), flush=True)
    time.sleep(0.6)
out = pd.concat(rows, ignore_index=True)
out.to_csv("research/data/tank/team_games.csv", index=False)
print("wrote", len(out), "rows")
