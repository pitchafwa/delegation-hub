"""Head coach per team per season (nba_api CommonTeamRoster.coaches).
Writes data/team_coaches.csv: team_id, season (e.g. 2019-20), head_coach(es).
"""
import sys, time
from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import commonteamroster
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
TEAM_IDS = list(range(1610612737, 1610612767))
import os
YEARS = [int(x) for x in os.environ.get("COACH_YEARS", "").split(",") if x] or list(range(2009, 2026))
APPEND = bool(os.environ.get("COACH_YEARS"))
rows = []
for yr in YEARS:
    season = f"{yr}-{str(yr+1)[-2:]}"
    for tid in TEAM_IDS:
        for attempt in range(3):
            try:
                c = commonteamroster.CommonTeamRoster(team_id=tid, season=season, timeout=30).coaches.get_data_frame()
                break
            except Exception as ex:
                time.sleep(2); c = None
        if c is None: print("fail", season, tid, flush=True); continue
        hc = c[c["COACH_TYPE"].str.contains("Head", na=False)]
        for _, r in hc.iterrows():
            rows.append(dict(team_id=tid, yr=yr, coach=r["COACH_NAME"], coach_type=r["COACH_TYPE"], is_assistant=r["IS_ASSISTANT"]))
        time.sleep(0.6)
    print(season, "done", len(rows), flush=True)
new = pd.DataFrame(rows)
path = ROOT / "data" / "team_coaches.csv"
if APPEND and path.exists():
    old = pd.read_csv(path)
    new = pd.concat([old[~old["yr"].isin(YEARS)], new])
new.to_csv(path, index=False)
