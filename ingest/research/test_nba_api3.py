from nba_api.stats.endpoints import playerindex
import time

t0 = time.time()
resp = playerindex.PlayerIndex(season="2023-24")
df = resp.get_data_frames()[0]
print("elapsed:", time.time() - t0)
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
