from nba_api.stats.endpoints import leaguedashplayerstats
import time

t0 = time.time()
resp = leaguedashplayerstats.LeagueDashPlayerStats(
    season="2023-24",
    season_type_all_star="Regular Season",
    per_mode_detailed="PerGame",
    measure_type_detailed_defense="Advanced",
)
df = resp.get_data_frames()[0]
print("elapsed:", time.time() - t0)
print(df.shape)
print([c for c in df.columns if "RANK" not in c])
