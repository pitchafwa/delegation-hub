"""Pull the full real NBA combine history (first-party nba_api endpoint,
already in the same PLAYER_ID system -- no linking needed).
"""
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import draftcombinestats

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "combine_cache"
CACHE.mkdir(exist_ok=True)

frames = []
for y in range(2008, 2027):
    season = f"{y}-{str(y+1)[2:]}"
    path = CACHE / f"{season}.csv"
    if not path.exists():
        resp = draftcombinestats.DraftCombineStats(season_all_time=season)
        df = resp.get_data_frames()[0]
        df.to_csv(path, index=False)
    frames.append(pd.read_csv(path))

combine = pd.concat(frames, ignore_index=True).drop_duplicates("PLAYER_ID", keep="last")
combine.to_csv(ROOT / "data" / "combine_all.csv", index=False)
print(f"Combine data: {len(combine)} unique players, seasons {sorted(combine['SEASON'].unique())}")
