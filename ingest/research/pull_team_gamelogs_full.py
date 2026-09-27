"""Full team-level box-score game logs (2010-11..2025-26), for the game-level ("which specific game will he go off in") research: opponent identity,
pace, and points/production allowed by category and by position. Writes data/gamelevel/team_box.csv.
"""
import sys
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

sys.stdout.reconfigure(encoding="utf-8")
seasons = [f"{y}-{str(y+1)[2:]}" for y in range(2010, 2026)]
rows = []
for s in seasons:
    for tries in range(3):
        try:
            df = leaguegamelog.LeagueGameLog(season=s, season_type_all_star="Regular Season", player_or_team_abbreviation="T", timeout=60).get_data_frames()[0]
            break
        except Exception as e:
            print(s, "retry", e)
            time.sleep(3)
    else:
        continue
    df["season"] = s
    rows.append(df)
    print(s, len(df), flush=True)
    time.sleep(0.6)
out = pd.concat(rows, ignore_index=True)
out.to_csv("research/data/gamelevel/team_box.csv", index=False)
print("wrote", len(out), "rows,", out.shape[1], "columns:", out.columns.tolist())
