"""Does the free-agent gap fade through a season? (2026-10-27, Tommy: maybe the free-agent markdown should shrink as the season goes on.)
fa_bias_study.py PART 3 on two seasons of the league's real rosters (league_days_2025.json = 2024-25, league_days_2026.json = 2025-26), with DELCO's
pregame number (delco_pregame_all.csv.gz): for players DELCO put at 20-35 at the first game of a matchup week, actual weekly average minus DELCO,
rostered vs free agent, matchup week by matchup week; also split by level (does the gap grow with level, as the current shrink assumes?).
Run from ingest/:  .venv/Scripts/python research/fa_gap_by_week.py
"""
import json
import sys
from datetime import timedelta

import pandas as pd

sys.argv = sys.argv[:1]
import backtest_weekly_planner as B

k = pd.read_csv(B.D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "SEASON", "pre"])
k["d"] = pd.to_datetime(k.GAME_DATE).dt.date
rows = []
for season, gl, lfile in (("2024-25", B.g24, "league_days_2025.json"), ("2025-26", B.g, "league_days_2026.json")):
    league = json.load(open(B.D / lfile))
    date0 = gl.GAME_DATE.min()
    sp_date = lambda sp: (date0 + timedelta(days=int(sp) - 1)).date()
    ks = k[k.SEASON == season].copy()
    ks["n"] = ks.PLAYER_ID.map(dict(zip(gl.PLAYER_ID, gl.n)))
    fp = {(n, d.date()): v for n, d, v in zip(gl.n, gl.GAME_DATE, gl.fp)}
    mps = sorted({recs[0]["mp"] for recs in league.values() if recs})
    for mp in mps:
        sps = sorted(int(sp) for sp, recs in league.items() if recs and recs[0]["mp"] == mp)
        d0, d1 = sp_date(sps[0]), sp_date(sps[-1])
        rostered = {B.norm(e[1]) for r in league[str(sps[0])] for e in r["e"]}
        wk = ks[(ks.d >= d0) & (ks.d <= d1)].sort_values("d")
        for n, sub in wk.groupby("n"):
            acts = [fp[(n, d)] for d in sub.d if (n, d) in fp]
            if acts and not pd.isna(sub.pre.iloc[0]):
                rows.append(dict(season=season, mp=mp, n=n, pre=float(sub.pre.iloc[0]), act=sum(acts) / len(acts), fa=n not in rostered))
df = pd.DataFrame(rows)
df["err"] = df.act - df.pre
mid = df[(df.pre >= 20) & (df.pre < 35)]


def gap(s):
    r, f = s[~s.fa].err, s[s.fa].err
    if len(f) < 15 or len(r) < 15:
        return None
    return r.mean() - f.mean(), 1.96 * (r.var() / len(r) + f.var() / len(f)) ** 0.5, r.mean(), f.mean(), len(r), len(f)


print("players DELCO put at 20-35: gap = how much more free agents were overrated than rostered players (pts per game)\n")
print(f"{'matchup week':<14}" + "".join(f"{s:>34}" for s in ("2024-25", "2025-26", "both")))
for lab, lo, hi in [(f"week {w}", w, w) for w in range(1, 7)] + [("weeks 7-9", 7, 9), ("weeks 10-13", 10, 13), ("weeks 14-19", 14, 19)]:
    cells = []
    for s in (mid[mid.season == "2024-25"], mid[mid.season == "2025-26"], mid):
        x = gap(s[(s.mp >= lo) & (s.mp <= hi)])
        cells.append(f"{x[0]:+5.1f} +/-{x[1]:3.1f} (FA n={x[5]:>3})" if x else "too few")
    print(f"{lab:<14}" + "".join(f"{c:>34}" for c in cells))
print("\nboth seasons, by level (weeks 1-2 | weeks 3+):")
for lo, hi in ((20, 25), (25, 30), (30, 35)):
    s = df[(df.pre >= lo) & (df.pre < hi)]
    a, b = gap(s[s.mp <= 2]), gap(s[s.mp >= 3])
    fmt = lambda x: f"gap {x[0]:+5.1f} +/-{x[1]:3.1f} (rostered {x[2]:+4.1f}, FA {x[3]:+4.1f}, FA n={x[5]})" if x else "too few"
    print(f"  {lo}-{hi}:  {fmt(a)}   |   {fmt(b)}")
print("\ncurrent markdown: 25 -> -1.2, 30 -> -3.2, 35 -> -5.2")
