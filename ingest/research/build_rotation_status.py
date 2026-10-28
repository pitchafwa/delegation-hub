"""Who is out of his team's rotation while healthy (2026-10-28, Tommy: Oscar Tshiebwe, a two-way player who sat every Rockets game, still planned at 28).
For each player this season: how many of his last 5 HEALTHY team games he played (an injury/illness, not-with-team or suspension absence is skipped;
a coach's decision, G League / two-way assignment or DNP counts as not playing). Box scores from pull_box_v3.py (data/game_logs/box_v3_<season>.csv).
The weekly plan scales his chance of playing and his points if he plays by ROT_TABLE in build_week_plan.py (measured on 2024-25 and 2025-26; see
that comment). Writes dashboard/rotation_status.json: {generated, players: {normalized name: {played, n, team, last}}}.
Run from ingest/:  .venv/Scripts/python research/build_rotation_status.py
"""
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from season import current_season_id

R = Path(__file__).resolve().parent
OUT = R.parent.parent / "dashboard" / "rotation_status.json"
sid = current_season_id()
season = f"{sid - 1}-{str(sid)[2:]}"


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


b = pd.read_csv(R / "data" / "game_logs" / f"box_v3_{season}.csv", dtype={"gameId": str})
c = b.comment.fillna("")
b["played"] = c.eq("") & b.minutes.notna()
b = b[~c.str.contains("Injury|Illness|Not With Team|Suspen|Personal", case=False)]
b = b.sort_values(["game_date", "gameId"])
players = {}
for pid, sub in b.groupby("personId"):
    last = sub.tail(5)
    name = norm(f"{last.firstName.iloc[-1]} {last.familyName.iloc[-1]}")
    players[name] = {"played": int(last.played.sum()), "n": len(last), "team": last.teamTricode.iloc[-1], "last": str(last.game_date.iloc[-1])[:10]}
OUT.write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": season, "players": players}, separators=(",", ":")), encoding="utf-8")
out_of = sorted((k for k, v in players.items() if v["n"] >= 3 and v["played"] <= 1))
print(f"rotation status: {len(players)} players, {len(out_of)} healthy but sitting (played 0-1 of their last 3-5 healthy games), e.g. {out_of[:8]}")
