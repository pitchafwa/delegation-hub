from nba_api.stats.endpoints import leaguedashplayerstats
import time

t0 = time.time()
resp = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2023-24",
    season_type_all_star="Regular Season",
    per_mode_detailed="Totals",
    measure_type_detailed_defense="Base",
)
df = resp.get_data_frames()[0]
print("elapsed:", time.time() - t0)
print(df.shape)
print(df.columns.tolist())
print(df.head(3)[["PLAYER_NAME", "AGE", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM"]])
