from pathlib import Path
import time

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "birthdate_cache"
CACHE.mkdir(exist_ok=True)

intl = pd.read_csv(ROOT / "data" / "international_features.csv")
player_ids = intl["PERSON_ID"].dropna().unique()
print(f"{len(player_ids)} international players to fetch")

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
        except Exception:
            time.sleep(1.5)

result = pd.concat(rows, ignore_index=True)
# merge into the existing real birthdates file (already has all college
# players) rather than overwrite it
existing = pd.read_csv(ROOT / "data" / "birthdates_all.csv")
combined = pd.concat([existing, result], ignore_index=True).drop_duplicates(subset=["PERSON_ID"], keep="first")
combined.to_csv(ROOT / "data" / "birthdates_all.csv", index=False)
print(f"Saved {len(combined)} total birthdates (added {len(result)} international)")
