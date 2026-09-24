from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "birthdate_cache"
CACHE.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")
player_ids = df["PLAYER_ID"].dropna().unique()
print(f"{len(player_ids)} players to fetch")

rows = []
for i, pid in enumerate(player_ids):
    path = CACHE / f"{int(pid)}.csv"
    if path.exists():
        rows.append(pd.read_csv(path))
        continue
    for attempt in range(3):
        try:
            resp = commonplayerinfo.CommonPlayerInfo(player_id=int(pid), timeout=15)
            d = resp.get_data_frames()[0][["PERSON_ID", "BIRTHDATE"]]
            d.to_csv(path, index=False)
            rows.append(d)
            time.sleep(0.5)
            break
        except Exception as e:
            time.sleep(1.5)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(player_ids)} done")

result = pd.concat(rows, ignore_index=True)
result.to_csv(ROOT / "data" / "birthdates_all.csv", index=False)
print(f"Saved {len(result)} birthdates")
