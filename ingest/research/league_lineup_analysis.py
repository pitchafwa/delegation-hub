"""How was this league actually played in 2025-26? (start/sit strategy grounding)

From data/league_days_2026.json (every team's daily lineup) measure, per team-week and league-wide:
  - games started per week (10 starting slots, so the ceiling is 10 x days with games)
  - idle-slot days: player-games lost because a slot sat empty / held a non-playing player while a rostered player
    who DID play sat on the bench  (pure lineup-management leakage, approximate: uses default positions for eligibility)
  - hindsight-optimal lineup gap (upper bound: assumes you knew who'd score)
  - how much weekly points depend on games started, and how random a week's result is
Position eligibility is approximated from ESPN's default position (PG->PG/G/UTIL, SG->SG/G/UTIL, SF->SF/F/UTIL, PF->PF/F/UTIL,
C->C/UTIL), so multi-position players make the leakage figures a slight OVER-estimate; treat as directional.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.stdout.reconfigure(encoding="utf-8")
D = json.load(open(Path(__file__).resolve().parent / "data" / "league_days_2026.json"))
SLOTS = [0, 1, 2, 3, 4, 5, 6, 11, 11, 11]                      # PG SG SF PF C G F UTIL x3
ELIG = {1: {0, 5, 11}, 2: {1, 5, 11}, 3: {2, 6, 11}, 4: {3, 6, 11}, 5: {4, 11}}
STARTING = set(SLOTS)

rows = []
for d, recs in D.items():
    for r in recs:
        ents = r["e"]
        played = [e for e in ents if e[3] and e[2] != 13 and e[4] is not None]
        started = [e for e in played if e[2] in STARTING]
        benched = [e for e in played if e[2] == 12]
        # hindsight-optimal assignment among players who played
        if played:
            M = np.full((len(played), len(SLOTS)), -1e6)
            for i, e in enumerate(played):
                for j, s in enumerate(SLOTS):
                    if s in ELIG.get(e[5], {11}):
                        M[i, j] = e[4]
            ri, ci = linear_sum_assignment(-M)
            opt = float(sum(M[i, j] for i, j in zip(ri, ci) if M[i, j] > -1e5))
            opt_n = int(sum(1 for i, j in zip(ri, ci) if M[i, j] > -1e5))
        else:
            opt, opt_n = 0.0, 0
        act = float(sum(e[4] for e in started))
        rows.append(dict(day=int(d), mp=r["mp"], team=r["team"], n_played=len(played), n_started=len(started), n_bench_played=len(benched),
                         bench_pts=float(sum(e[4] for e in benched)), act=act, opt=opt, opt_n=opt_n,
                         idle=max(0, min(10, len(played)) - len(started)), roster_active=len([e for e in ents if e[2] != 13])))
df = pd.DataFrame(rows)
df = df[df.mp <= 19]  # regular season only
print(f"{len(df)} team-days, {df.team.nunique()} teams, matchup periods 1-19")

wk = df.groupby(["mp", "team"]).agg(started=("n_started", "sum"), played=("n_played", "sum"), idle=("idle", "sum"), act=("act", "sum"),
                                     opt=("opt", "sum"), bench_played=("n_bench_played", "sum"), bench_pts=("bench_pts", "sum"), days=("day", "nunique")).reset_index()
wk["pts_per_start"] = wk.act / wk.started
print("\n== per team-week (regular season) ==")
print(wk[["started", "played", "idle", "act", "opt", "bench_played", "pts_per_start"]].describe().round(1).loc[["mean", "std", "min", "50%", "max"]])
print(f"\nshare of team-weeks in which a starting slot lost a game to lineup management (idle>0): {(wk.idle > 0).mean():.0%}; mean idle player-games/wk {wk.idle.mean():.1f}")
print(f"hindsight-optimal lineup would add {(wk.opt - wk.act).mean():.0f} pts/week on average (upper bound; {((wk.opt - wk.act) / wk.act).mean():.1%} of actual)")

# what explains weekly points: games started vs quality
wk["z_starts"] = wk.started
c = np.corrcoef(wk.started, wk.act)[0, 1]
slope = np.polyfit(wk.started, wk.act, 1)
print(f"\nweekly points vs games started: corr {c:.2f}, {slope[0]:.1f} pts per extra started game (average points per started game is {wk.pts_per_start.mean():.1f})")
by_team = wk.groupby("team").agg(starts=("started", "mean"), pts=("act", "mean"), idle=("idle", "mean"), pps=("pts_per_start", "mean")).round(1).sort_values("pts", ascending=False)
print("\nteam averages per week:\n", by_team.to_string())

# matchups: how random is a week?
pairs = []
for mp, g in wk.groupby("mp"):
    pass
print("\n(matchup pairings are joined below from ESPN's schedule to size week-to-week randomness)")
json.dump({"wk": wk.to_dict("records")}, open(Path(__file__).resolve().parent / "data" / "league_weeks_2026.json", "w"))
