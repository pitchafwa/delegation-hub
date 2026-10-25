"""Reclaim study (Tommy, 2026-10-25): when you drop a player, how likely is another team to take him before you can re-add him?

The stream manager (build_week_plan.py) charges a drop for the chance the player is gone when the rental ends: lose_p = max(RECLAIM_LOSE 0.15, p_taken(level, days)),
p_taken = 25% (33+ ppg) / 16% (27-33) / 12% (below) per 2 days, compounding. An early hint from this season was 2 of 29 drops taken within 2 days.

Data: every executed DROP in this league, 2023-24 .. 2026-27 so far (data/league_transactions_<year>.json from waiver_history_check.py), and every later ADD of the same player.
  quality   his fantasy points per game over his last (up to) 10 games BEFORE the drop, from every team's daily lineups (data/league_days_<year>.json): what the planner would know
  healthy   he played at least 2 games in the 10 days before the drop (an injured player is a different case)
  taken     another team added him (free-agent add or waiver claim) within t days; if the team that dropped him re-adds him first, the episode ends (counts as not taken)
Drops in the last 8 days of a season are left out (no time to be taken). Results by quality tier for t = 1, 2, 3, 4, 7 days, healthy players only, plus this season alone.
Run from ingest/:  .venv/Scripts/python research/reclaim_study.py   (writes data/reclaim_study.json)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
YEARS = [2024, 2025, 2026, 2027]
DAY_MS = 86400000
TIERS = [("33+", 33, 99), ("27-33", 27, 33), ("20-27", 20, 27), ("<20", -99, 20)]
HOR = [1, 2, 3, 4, 7]

eps = []
for y in YEARS:
    tf, lf = D / f"league_transactions_{y}.json", D / f"league_days_{y}.json"
    if not (tf.exists() and lf.exists()):
        continue
    tx = json.load(open(tf))
    days = json.load(open(lf))
    games = collections.defaultdict(list)                      # pid -> [(scoring period, points)] for games played
    for sp, recs in days.items():
        for r in recs:
            for e in r["e"]:
                if e[3] and e[4] is not None:
                    games[e[0]].append((int(sp), float(e[4])))
    for v in games.values():
        v.sort()
    last_sp = max(int(k) for k in days) if days else 0
    moves = []
    for t in tx:
        if t.get("status") != "EXECUTED" or t.get("type") not in ("FREEAGENT", "WAIVER", "ROSTER"):
            continue
        ts = t.get("processDate") or t.get("proposedDate")
        for it in t.get("items", []):
            if it.get("type") in ("ADD", "DROP"):
                moves.append({"ts": ts, "sp": t.get("scoringPeriodId") or 0, "pid": it.get("playerId"), "kind": it["type"],
                              "team": it.get("toTeamId") if it["type"] == "ADD" else it.get("fromTeamId")})
    moves.sort(key=lambda m: m["ts"])
    by_pid = collections.defaultdict(list)
    for m in moves:
        by_pid[m["pid"]].append(m)
    played_sp = max((sp for v in games.values() for sp, _ in v), default=0)
    for m in moves:
        if m["kind"] != "DROP" or not m["sp"]:
            continue
        if m["sp"] > played_sp - 8:                             # too close to the end of what we have (season end, or today for this season)
            continue
        prev = [p for sp, p in games.get(m["pid"], []) if sp < m["sp"]][-10:]
        recent = [sp for sp, _ in games.get(m["pid"], []) if m["sp"] - 10 <= sp < m["sp"]]
        if len(prev) < 3:
            continue
        nxt = next((x for x in by_pid[m["pid"]] if x["ts"] > m["ts"] and x["kind"] == "ADD"), None)
        taken_days = None
        if nxt and nxt["team"] != m["team"]:
            taken_days = (nxt["ts"] - m["ts"]) / DAY_MS
        eps.append({"y": y, "ppg": float(np.mean(prev)), "healthy": len(recent) >= 2, "taken_days": taken_days, "readd_self": bool(nxt and nxt["team"] == m["team"]),
                    "self_days": (nxt["ts"] - m["ts"]) / DAY_MS if nxt and nxt["team"] == m["team"] else None})

print(f"{len(eps)} drops of players with 3+ recent games ({sum(e['healthy'] for e in eps)} healthy), seasons {sorted({e['y'] for e in eps})}\n")


def table(rows, label):
    print(label)
    print("  tier     drops   " + "  ".join(f"<= {h}d" for h in HOR) + "   plan assumes (2d / 4d)")
    out = {}
    for name, lo, hi in TIERS:
        s = [e for e in rows if lo <= e["ppg"] < hi]
        if not s:
            continue
        fr = [np.mean([e["taken_days"] is not None and e["taken_days"] <= h for e in s]) for h in HOR]
        p48 = 0.25 if lo >= 33 else (0.16 if lo >= 27 else 0.12)
        plan2, plan4 = max(0.15, p48), max(0.15, 1 - (1 - p48) ** 2)
        out[name] = {"n": len(s), "taken": {h: round(float(f), 3) for h, f in zip(HOR, fr)}}
        print(f"  {name:6s} {len(s):6d}   " + "  ".join(f"{f:5.0%}" for f in fr) + f"   {plan2:.0%} / {plan4:.0%}")
    return out


res = {"all": table([e for e in eps if e["healthy"]], "Healthy players: share taken by another team within ...")}
res["this_season"] = table([e for e in eps if e["healthy"] and e["y"] == 2027], "\nThis season only (2026-27):")
res["injured"] = table([e for e in eps if not e["healthy"]], "\nPlayers who had NOT played in the 10 days before the drop:")
sd = [e["self_days"] for e in eps if e["readd_self"] and e["self_days"] is not None]
print(f"\nThe dropping team re-added the same player itself in {len(sd)} cases (median {np.median(sd) if sd else 0:.1f} days later).")
json.dump(res, open(D / "reclaim_study.json", "w"), indent=1)
