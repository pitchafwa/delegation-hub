"""Game-day logger for grading the DailyFantasyFuel projections (and measuring how fast they react to injury news). Run every ~10 minutes on game days (the scheduled task does this):

  uv run python research/dfs_monitor.py            (from ingest/)

Each run, politely (one request to DFF, one to ESPN's public injuries feed):
  * data/dfs/log.csv      every DFF row whose projection, injury tag or opponent CHANGED since the last run (plus first sightings), stamped with the fetch time -> the full history of each player's number
  * data/dfs/inj_log.csv  every change in ESPN's injury designation (status, ESPN's own update time, comment), stamped with the fetch time -> when the news broke
Both are local only (gitignored). grade_dfs.py reads them after games are played.
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from pull_dfs_projections import COLS, OUT, URL, parse

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
ESPN_INJ = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
INJ_COLS = ["fetched_utc", "name", "team", "status", "espn_time", "comment"]
STATE = OUT / "monitor_state.json"


def append(path, cols, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"dff": {}, "inj": {}}
    # --- DFF
    new_rows = []
    try:
        r = requests.get(URL, headers=UA, timeout=30)
        r.raise_for_status()
        rows = parse(r.text)
        for x in rows:
            key = f"{x['game_date']}|{x['name']}"
            sig = f"{x['dk_proj']}|{x['inj']}|{x['opp']}|{x['depth_rank']}"
            if st["dff"].get(key) != sig:
                st["dff"][key] = sig
                new_rows.append(dict(x, fetched_utc=now))
        append(OUT / "log.csv", COLS, new_rows)
    except Exception as ex:
        print("DFF fetch failed:", repr(ex)[:100])
        rows = []
    # --- ESPN injuries
    inj_rows = []
    try:
        j = requests.get(ESPN_INJ, headers=UA, timeout=30).json()
        for t in j.get("injuries", []):
            for it in t.get("injuries", []):
                a = it.get("athlete") or {}
                name = a.get("displayName") or ""
                status = it.get("status") or ""
                sig = f"{status}|{it.get('date')}|{(it.get('shortComment') or '')[:60]}"
                if name and st["inj"].get(name) != sig:
                    st["inj"][name] = sig
                    inj_rows.append({"fetched_utc": now, "name": name, "team": t.get("displayName"), "status": status, "espn_time": it.get("date"), "comment": (it.get("shortComment") or "")[:200]})
        append(OUT / "inj_log.csv", INJ_COLS, inj_rows)
    except Exception as ex:
        print("ESPN injuries fetch failed:", repr(ex)[:100])
    STATE.write_text(json.dumps(st), encoding="utf-8")
    print(f"{now}: DFF {len(rows)} rows ({len(new_rows)} new/changed), ESPN injury changes {len(inj_rows)}")


if __name__ == "__main__":
    main()
