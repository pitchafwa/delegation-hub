"""Pull the league's real 2025-26 (seasonId 2026) daily lineups so start/sit strategy can be grounded in how
this league is actually played: who started, who sat, how many starting slots stayed empty, what scored.

Per scoring period (day) and team we keep every roster entry's lineup slot, whether the player played (GP stat 42),
his applied fantasy points, and the matchup period. Writes data/league_days_2026.json (gitignored, regenerable).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "data" / "league_days_2026.json"
lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
days = json.load(open(OUT)) if OUT.exists() else {}
for d in range(1, 161):
    if str(d) in days:
        continue
    try:
        raw = lg.espn_request.league_get(params={"view": "mBoxscore", "scoringPeriodId": d})
    except Exception as e:
        print("period", d, "failed:", repr(e)[:120], flush=True)
        continue
    rec = []
    for m in raw.get("schedule", []):
        for side in ("home", "away"):
            t = m.get(side, {})
            ros = t.get("rosterForCurrentScoringPeriod") or {}
            if m.get("matchupPeriodId") is None or not ros.get("entries"):
                continue
            ents = []
            for e in ros["entries"]:
                pl = e.get("playerPoolEntry", {}).get("player", {})
                st = [s for s in pl.get("stats", []) if s.get("scoringPeriodId") == d and s.get("statSourceId") == 0]
                gp = bool(st and st[0].get("stats", {}).get("42", 0))
                ents.append([pl.get("id"), pl.get("fullName"), e.get("lineupSlotId"), int(gp),
                             (st[0].get("appliedTotal") if st else None), pl.get("defaultPositionId"), pl.get("injuryStatus")])
            rec.append({"mp": m["matchupPeriodId"], "team": t.get("teamId"), "e": ents})
    days[str(d)] = rec
    if d % 10 == 0:
        OUT.write_text(json.dumps(days), encoding="utf-8")
        print("done through", d, flush=True)
    time.sleep(0.4)
OUT.write_text(json.dumps(days), encoding="utf-8")
print("finished", len(days))
