"""Pull the full 2026-27 NBA schedule from ESPN's public scoreboard (one request per date) and write dashboard/nba_schedule.json:
{"generated":..., "games": {"2026-10-20": [["BOS","DET","19:00Z"], ...]}}  (away, home, tip-off UTC).  Used for schedule-aware roster planning
(games per team per fantasy week, back-to-backs, playoff-week game counts). Team codes are normalised to nba_api style (NYK, SAS, GSW, ...)."""
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
FIX = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}
START, END = date(2026, 10, 20), date(2027, 4, 12)
games, d = {}, START
while d <= END:
    for i in range(3):
        try:
            j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", params={"dates": d.strftime("%Y%m%d")}, timeout=30).json()
            break
        except Exception:
            j, _ = {}, time.sleep(2)
    day = []
    for ev in j.get("events", []):
        comp = ev["competitions"][0]["competitors"]
        h = next((c for c in comp if c["homeAway"] == "home"), None)
        a = next((c for c in comp if c["homeAway"] == "away"), None)
        if h and a and h["team"].get("abbreviation") and a["team"].get("abbreviation"):
            day.append([FIX.get(a["team"]["abbreviation"], a["team"]["abbreviation"]), FIX.get(h["team"]["abbreviation"], h["team"]["abbreviation"]), ev["date"][11:16] + "Z"])
    if day:
        games[d.isoformat()] = day
    d += timedelta(days=1)
    time.sleep(0.15)
out = {"generated": datetime.now(timezone.utc).isoformat(), "games": games}
(HUB / "nba_schedule.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
n = sum(len(v) for v in games.values())
print(f"{len(games)} game days, {n} games, {min(games)} .. {max(games)}")
