"""Why do forecasts for later weeks run high? (2026-11-07, follow-up to title_odds_study.py: forecasts made at a week's start ran ~65 points per 7 days above
what teams really scored in the weeks after, for every team.)  2025-26 replay (backtest_phase2 data).

A. PLAYER LEVEL.  Every rostered player at each week's start (level 20+, healthy that morning: played one of his team's last 2 games): what the solver assumes for
   each of his team's later games (chance to play x his level) against what he really produced in those games (0 when he sat), by how many weeks ahead.
   Split into how often he played (vs the assumed 85%) and points per game when he played (vs his level).
B. ROSTER.  Of those players, how many were still on the same fantasy team that week (dropped, traded), and the points the team really got from players
   added after the forecast.
Run from ingest/:  python research/forecast_horizon_study.py
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import backtest_phase2 as P
import backtest_weekly_planner as B

sys.stdout.reconfigure(encoding="utf-8")
rows = []
roster_rows = []
for c in [w for w in P.REG if w <= 16]:
    d0 = P.dates_of(c)[0]
    for t in P.TEAMS:
        if (c, t) not in P.W:
            continue
        ents = P.W[(c, t)][P.mp_sps[c][0]]
        names0 = {B.norm(e[1]) for e in ents}
        for e in ents:
            if e[2] == 13:
                continue
            n = B.norm(e[1])
            L = B.level(n, d0)
            tm = B.team_on(n, d0)
            if not L or L < 20 or not tm:
                continue
            last2 = sorted(d for d in B.team_days.get(tm, ()) if d < d0)[-2:]
            if not any((n, d) in B.played_fp for d in last2):
                continue
            for w in P.ALL_MP:
                if w < c or w > 19 or P.LEN.get(w) != 7:
                    continue
                for dd in P.dates_of(w):
                    tm2 = B.team_on(n, dd)
                    if not tm2 or dd not in B.team_days.get(tm2, ()):
                        continue
                    fp = B.played_fp.get((n, dd))
                    rows.append(dict(c=c, h=w - c, n=n, L=L, pf=B.p_future(L), played=fp is not None, fp=fp or 0.0, team_changed=tm2 != tm))
        # roster churn: anchor-roster players still on this fantasy team in later weeks; points from players who arrived later
        for w in P.ALL_MP:
            if w <= c or w > 19 or P.LEN.get(w) != 7 or (w, t) not in P.W:
                continue
            days = P.W[(w, t)]
            on = {B.norm(e[1]) for sp in days for e in days[sp]}
            got = sum(e[4] or 0 for sp in days for e in days[sp] if e[2] in P.STARTERS and e[3])
            new = sum(e[4] or 0 for sp in days for e in days[sp] if e[2] in P.STARTERS and e[3] and B.norm(e[1]) not in names0)
            roster_rows.append(dict(c=c, h=w - c, team=t, kept=len(names0 & on) / max(1, len(names0)), pts=got, new_share=new / got if got else np.nan))
df = pd.DataFrame(rows)
rr = pd.DataFrame(roster_rows)
df["exp"] = df.pf * df.L
df["hb"] = pd.cut(df.h, [-1, 0, 1, 3, 6, 9, 20], labels=["same week", "1 ahead", "2-3", "4-6", "7-9", "10+"])
print("A. PLAYER LEVEL (healthy, level 20+ at the forecast; per scheduled team game)")
print(f"{'weeks ahead':12s} {'games':>7s} {'assumed':>8s} {'real':>7s} {'ratio':>6s} | {'played':>7s} {'vs 85%':>7s} | {'pts when played':>15s} {'vs level':>8s}")
for hb, g in df.groupby("hb", observed=True):
    pl = g[g.played]
    print(f"{hb:12s} {len(g):7d} {g.exp.mean():8.1f} {g.fp.mean():7.1f} {g.fp.mean() / g.exp.mean():6.2f} | {g.played.mean():7.2f} {g.played.mean() / g.pf.mean():7.2f} | "
          f"{pl.fp.mean():15.1f} {pl.fp.mean() / pl.L.mean():8.2f}")
print("\n   by level at the forecast (all horizons 1+):")
g1 = df[df.h >= 1]
for lo, hi in ((20, 30), (30, 40), (40, 50), (50, 99)):
    g = g1[(g1.L >= lo) & (g1.L < hi)]
    pl = g[g.played]
    print(f"   level {lo}-{hi}: played {g.played.mean():.2f}  pts when played / level {pl.fp.mean() / pl.L.mean():.2f}  real / assumed {g.fp.mean() / g.exp.mean():.2f}  (n {len(g)})")
print("\nB. ROSTER CHURN")
rr["hb"] = pd.cut(rr.h, [0, 1, 3, 6, 9, 20], labels=["1 ahead", "2-3", "4-6", "7-9", "10+"])
for hb, g in rr.groupby("hb", observed=True):
    print(f"   {hb:8s} anchor roster still on the team {g.kept.mean():.0%}   share of real points from players added since {g.new_share.mean():.0%}")
