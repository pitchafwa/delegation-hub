"""Weekly team projections: which way of turning the lineup solver into a weekly forecast is most accurate? (2026-11-02, Tommy: the live "blend"
dropped DRNK from #1 to #7 after one full week, by scaling every team by how far it beat or missed its own model last week.)

Replays this league's 2025-26 season (backtest_phase2's data: real rosters, real box scores). At the start of each regular-season week T, with only
what was known then (each team's roster that morning, player levels from earlier games), forecast each team's points for week T, and score it against
what the team really scored. "Manager factor" for a team in a past week w = its real points / the solver's forecast for week w from its roster that
morning: how much it beat its roster on paper (streaming, lineup care, luck, injuries).
  solver      the solver alone
  league      solver x the league-wide factor so far (every team's real / forecast, pooled over past weeks): fixes the solver's overall bias only
  live1       solver x this team's factor in the most recent full week (today's live blend while only one week has been played)
  live3       solver x the mean of this team's factors in its last 3 full weeks (the live blend's design)
  shrunkK     solver x league factor x (team's season-to-date factor / league factor) ^ (n / (n + K)), n = weeks played: the team-specific part
              is trusted more as weeks pile up; K = 2, 5, 10, 20
Reported for every week T, and split by how many weeks had been played (1, 2-3, 4-7, 8+), since early season is what matters now.
Run from ingest/:  python research/weekly_proj_study.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import backtest_phase2 as P

sys.stdout.reconfigure(encoding="utf-8")
REG = P.REG                                   # full 7-day regular-season weeks
fc = {}                                       # (team, week) -> solver forecast from that week's opening roster
for w in REG:
    for t in P.TEAMS:
        m1 = P.project(t, w, w)[0]
        if m1:
            fc[(t, w)] = m1
act = {(t, w): P.actual_pts(w, t) for (t, w) in fc}

rows = []
for i, T in enumerate(REG):
    past = REG[:i]
    if not past:
        continue
    pool = [(act[k], fc[k]) for k in ((t, w) for t in P.TEAMS for w in past) if k in fc]
    lf = sum(a for a, _ in pool) / sum(f for _, f in pool)
    for t in P.TEAMS:
        if (t, T) not in fc:
            continue
        mine = [w for w in past if (t, w) in fc]
        if not mine:
            continue
        f = {w: act[(t, w)] / fc[(t, w)] for w in mine}
        tf = sum(act[(t, w)] for w in mine) / sum(fc[(t, w)] for w in mine)
        s = fc[(t, T)]
        r = dict(T=T, team=t, n=len(mine), act=act[(t, T)], solver=s, league=s * lf, live1=s * f[mine[-1]], live3=s * np.mean([f[w] for w in mine[-3:]]))
        for K in (2, 5, 10, 20):
            r[f"shrunk{K}"] = s * lf * (tf / lf) ** (len(mine) / (len(mine) + K))
        rows.append(r)
df = pd.DataFrame(rows)
M = ["solver", "league", "live1", "live3", "shrunk2", "shrunk5", "shrunk10", "shrunk20"]
df["stage"] = pd.cut(df.n, [0, 1, 3, 7, 99], labels=["1 week played", "2-3", "4-7", "8+"])
print(f"{len(df)} team-weeks, weeks {REG[1]}-{REG[-1]}\n")
out = []
for name, g in [("all", df)] + list(df.groupby("stage", observed=True)):
    o = {"stage": name, "n": len(g)}
    for m in M:
        o[m] = round((g[m] - g.act).abs().mean())
    out.append(o)
print("mean miss (points per team-week)")
print(pd.DataFrame(out).set_index("stage").to_string())
print("\naverage bias (forecast - real)")
print(pd.Series({m: round((df[m] - df.act).mean()) for m in M}).to_string())
# does a team's factor carry over at all? week-to-week correlation of the same team's factors
df["f_prev"] = df.live1 / df.solver
df["f_now"] = df.act / df.solver
print("\ncorrelation of a team's factor with its factor the week before:", round(df[["f_prev", "f_now"]].corr().iloc[0, 1], 3))
print("correlation with its last-3-weeks factor:", round(np.corrcoef(df.f_now, df.live3 / df.solver)[0, 1], 3))
df.to_csv(P.B.D / "weekly_proj_study.csv", index=False)
