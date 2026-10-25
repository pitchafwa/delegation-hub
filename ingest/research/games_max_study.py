"""Games-maximization study (Tommy, 2026-10-25): are we leaving starts on the table, and how much is a start worth?

The cap: a day counts in full if the team STARTS it under the cap, so the most starts that can count in a 7-day matchup is 39 + 10 = 49 (cap 40).
Data: every team's daily lineups and points, 2023-24 .. 2025-26 (data/league_days_<year>.json), 7-day regular-season matchups, human-run teams.
Per team-week: counted starts (starts on days the team began under the cap), overflow (counted starts beyond the cap), points, opponent points, win.
  1. How many counted starts do teams use; how often do they reach the cap / use overflow; how far from 49.
  2. What one more counted start is worth: weekly points regressed on counted starts and the team's points per start (roster quality), and win chance.
  3. Last-day overflow: teams that began the final day under the cap, by how many starters they had that day.
Run from ingest/:  .venv/Scripts/python research/games_max_study.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
CAP = 40.0
rows = []
for y in (2024, 2025, 2026):
    days = json.load(open(D / f"league_days_{y}.json"))
    wk = defaultdict(dict)
    for sp, recs in days.items():
        for r in recs:
            wk[(r["mp"], r["team"])][int(sp)] = r["e"]
    by_mp = defaultdict(dict)
    for (mp, team), dd in wk.items():
        sps = sorted(dd)
        if len(sps) != 7:
            continue
        cum, counted, pts, last_start, overflow, empty, benched_playing = 0, 0, 0.0, 0, 0, 0, 0
        for i, sp in enumerate(sps):
            e = dd[sp]
            st = [x for x in e if x[2] is not None and x[2] <= 11 and x[3]]
            if cum < CAP:
                counted += len(st)
                pts += sum((x[4] or 0) for x in st)
                if cum + len(st) > CAP:
                    overflow += cum + len(st) - CAP
            if i == len(sps) - 1:
                last_start = (cum, len(st))
            players_with_game = len([x for x in e if x[2] is not None and x[2] != 13 and x[3]])
            empty += max(0, min(10, players_with_game) - len(st)) if cum < CAP else 0
            cum += len(st)
        by_mp[mp][team] = {"y": y, "mp": mp, "team": team, "counted": counted, "pts": pts, "overflow": overflow, "last": last_start, "pps": pts / max(counted, 1), "empty": empty}
    for mp, teams in by_mp.items():
        # opponents: pair by ESPN schedule is not in this file; use the league median as the bar (win = above median) plus all-play share
        allp = sorted(t["pts"] for t in teams.values())
        for t in teams.values():
            t["allplay"] = sum(1 for p in allp if p < t["pts"]) / max(1, len(allp) - 1)
            rows.append(t)
print(f"{len(rows)} team-weeks (7-day matchups, 2023-26)\n")
c = np.array([r["counted"] for r in rows], float)
pts = np.array([r["pts"] for r in rows], float)
pps = np.array([r["pps"] for r in rows], float)
ap = np.array([r["allplay"] for r in rows], float)
print("1. Counted starts per matchup (most possible 49):")
for lo, hi in ((0, 36), (36, 40), (40, 41), (41, 44), (44, 47), (47, 50)):
    s = (c >= lo) & (c < hi)
    print(f"   {lo:2d}-{hi - 1:2d}: {s.mean():5.1%} of team-weeks, avg points {pts[s].mean():6.0f}, all-play win share {ap[s].mean():.2f}")
print(f"   average {c.mean():.1f}; reached the cap {np.mean(c >= 40):.0%}; used overflow {np.mean([r['overflow'] > 0 for r in rows]):.0%} (avg overflow when used {np.mean([r['overflow'] for r in rows if r['overflow'] > 0]):.1f})")
X = np.column_stack([np.ones(len(c)), c, pps])
b = np.linalg.lstsq(X, pts, rcond=None)[0]
print(f"\n2. weekly points = {b[0]:.0f} + {b[1]:.1f} x counted starts + {b[2]:.1f} x points per start   -> one more start ~ {b[1]:.0f} points")
Xw = np.column_stack([np.ones(len(c)), c, pps])
bw = np.linalg.lstsq(Xw, ap, rcond=None)[0]
print(f"   all-play win share: +{bw[1] * 100:.1f} points of win share per extra start (holding points per start fixed); +5 starts = +{bw[1] * 500:.0f}")
print("\n3. Final day, teams that began it under the cap:")
for lo, hi in ((0, 30), (30, 34), (34, 37), (37, 40)):
    s = [r for r in rows if lo <= r["last"][0] < hi]
    if s:
        print(f"   began at {lo}-{hi - 1} starts: {len(s):4d} team-weeks, final-day starters avg {np.mean([r['last'][1] for r in s]):.1f}, counted total {np.mean([r['counted'] for r in s]):.1f}")
print(f"\nEmpty slots on counted days (a player with a game was available but the slot sat empty, or fewer than 10 played): avg {np.mean([r['empty'] for r in rows]):.1f} per team-week")
