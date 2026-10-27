"""Free-agent bias study (2026-10-27, Tommy): is the free-agent shrink (FA_ANCHOR/FA_SHRINK in build_week_plan.py) right, or do we overrate
rostered players of the same level just as much? Two explanations to separate:
  1. level mix: free agents are mostly 20-35 pts players and rostered players include stars, whose numbers are steadier; compare within the same level band.
  2. selection: league managers leave unrostered exactly the players whose number is too high (role lost, crowded rotation), so at the same level a
     free agent is overrated more than a rostered player.
And the plan's own winner's curse: the free agents the plan would actually pick (top of the pool) vs the rest.

PART 1, 2025-26 (the league's real rosters, league_days_2026.json): at the start of each matchup week, every player with 8+ games and a game in the
last 10 days gets the backtest's projection (recency-weighted average, backtest_weekly_planner.level) and is compared with his actual average over
the games he played that week.
PART 2, this season: projection_log.csv numbers vs actual points the same day (games played), by level band and free agent / rostered.
Run from ingest/:  .venv/Scripts/python research/fa_bias_study.py
"""
import csv
import statistics as S
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.argv = sys.argv[:1]
import backtest_weekly_planner as B

BANDS = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 45), (45, 999)]
band = lambda v: next(f"{a}-{b}" if b < 999 else f"{a}+" for a, b in BANDS if a <= v < b)


def line(label, xs):
    if len(xs) < 15:
        return f"  {label:<34} n={len(xs):<5} (too few)"
    se = S.pstdev(xs) / len(xs) ** 0.5
    return f"  {label:<34} n={len(xs):<5} actual minus ours {S.mean(xs):+5.1f}  (+/- {1.96 * se:.1f})"


# ---------------- PART 1
rows = []
for (mp, team), days in sorted(B.weeks().items()):
    if mp == 1 or mp > 19:
        continue
    sp0 = min(days)
    if team != min(t for (m, t) in B.weeks() if m == mp):           # one pass per matchup week (rosters for all teams come from the same scoring period)
        continue
    d0 = B.sp_date(sp0)
    rostered = {B.norm(e[1]) for r in B.league[str(sp0)] for e in r["e"]}
    top_pool = []
    for n, sub in B.series.items():
        past = sub[sub.GAME_DATE.dt.date < d0]
        if len(past) < 8 or past.GAME_DATE.dt.date.max() < d0 - timedelta(days=10):
            continue
        wk = sub[(sub.GAME_DATE.dt.date >= d0) & (sub.GAME_DATE.dt.date < d0 + timedelta(days=7))]
        if wk.empty:
            continue
        L = B.level(n, d0)
        last3 = float(past.fp.tail(3).mean())
        r = dict(mp=mp, n=n, L=L, act=float(wk.fp.mean()), g=len(wk), fa=n not in rostered, trend=last3 - L)
        rows.append(r)
        if r["fa"]:
            top_pool.append(r)
    top_pool.sort(key=lambda r: -r["L"])
    for r in top_pool[:40]:                                          # the pool the backtest planner picked from
        r["pickable"] = True
df = pd.DataFrame(rows)
df["err"] = df.act - df.L
df["band"] = df.L.map(band)
print(f"PART 1: 2025-26, {df.mp.nunique()} matchup weeks, {len(df)} player-weeks ({df.fa.sum()} free agent, {(~df.fa).sum()} rostered)")
print("actual weekly average minus the backtest projection (negative = we overrated him), weighted by player-week:\n")
for b in [band(a) for a, _ in BANDS]:
    sub = df[df.band == b]
    if sub.empty:
        continue
    print(f"level {b}:")
    print(line("rostered", list(sub[~sub.fa].err)))
    print(line("free agents", list(sub[sub.fa].err)))
    print(line("  free agents the plan would pick", list(sub[sub.fa & (sub.get("pickable") == True)].err)))
print("\n20-35 combined:")
mid = df[(df.L >= 20) & (df.L < 35)]
print(line("rostered", list(mid[~mid.fa].err)))
print(line("free agents", list(mid[mid.fa].err)))
print(line("  free agents the plan would pick", list(mid[mid.fa & (mid.get("pickable") == True)].err)))
print("\n20-35, split by the last 3 games vs his projection (a crude 'role changing' signal the projection ignores):")
for lab, m in (("last 3 well BELOW (< -5)", mid.trend < -5), ("about the same", mid.trend.abs() <= 5), ("last 3 well ABOVE (> +5)", mid.trend > 5)):
    s = mid[m]
    print(f" {lab}:  share of free agents {s.fa.mean():.0%}")
    print(line("   rostered", list(s[~s.fa].err)))
    print(line("   free agents", list(s[s.fa].err)))
print("\nwhat the current shrink would do at each level (22 + 0.6 x (level - 22)):", ", ".join(f"{v}->{22 + 0.6 * (v - 22):.1f} ({0.6 * (v - 22) - (v - 22):+.1f})" for v in (25, 30, 35, 40)))

# ---------------- PART 2
ROOT = Path(__file__).resolve().parent.parent.parent
g = pd.read_csv(B.D / "game_logs" / "nba_api_2026-27.csv")
g["fp"] = g.PTS + 1.5 * g.REB + 2 * g.AST + 3 * g.STL + 3 * g.BLK + g.FG3M + 2 * g.FTM - g.FTA - g.TOV
g["d"] = pd.to_datetime(g.GAME_DATE).dt.date.map(lambda x: x.isoformat())
act = {(r.d, B.norm(r.PLAYER_NAME)): r.fp for r in g.itertuples()}
by = defaultdict(list)
seen = set()
for r in csv.DictReader(open(ROOT / "dashboard/projection_log.csv", encoding="utf-8")):
    k = (r["date"], B.norm(r["name"]))
    if k not in act or k in seen:
        continue
    seen.add(k)
    for col in ("model", "espn", "st"):
        if r.get(col):
            v = float(r[col])
            if 20 <= v < 35:
                by[(col, r.get("is_fa") == "1")].append(act[k] - v)
            by[(col, r.get("is_fa") == "1", "all")].append(act[k] - v)
print(f"\nPART 2: this season, projection log vs the same day's actual points (games played), {len(seen)} player-games")
for col, lab in (("model", "season DELCO"), ("espn", "ESPN"), ("st", "next-few-days DELCO (from 10-26)")):
    print(f"{lab}, players our number puts at 20-35:")
    print(line("rostered", by[(col, False)]))
    print(line("free agents", by[(col, True)]))
    print(f"{lab}, all levels:")
    print(line("rostered", by[(col, False, "all")]))
    print(line("free agents", by[(col, True, "all")]))

# ---------------- PART 3 (2026-10-27): last season again, but with DELCO's own pregame number (data/delco_pregame_all.csv.gz: built from games before
# that one only, leaning on earlier seasons early on -- the same kind of number the plan uses now), week by week, to see how the free-agent gap
# behaves through a season instead of waiting for this one
k = pd.read_csv(B.D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "SEASON", "pre"])
k = k[k.SEASON == "2025-26"]
k["d"] = pd.to_datetime(k.GAME_DATE).dt.date
pid_name = dict(zip(B.g.PLAYER_ID, B.g.n))
k["n"] = k.PLAYER_ID.map(pid_name)
fp = {(n, d.date()): v for n, d, v in zip(B.g.n, B.g.GAME_DATE, B.g.fp)}
rows3 = []
for mp in range(1, 20):
    sps = sorted(int(sp) for sp, recs in B.league.items() if recs and recs[0]["mp"] == mp)
    if not sps:
        continue
    d0, d1 = B.sp_date(sps[0]), B.sp_date(sps[-1])
    rostered = {B.norm(e[1]) for r in B.league[str(sps[0])] for e in r["e"]}
    wk = k[(k.d >= d0) & (k.d <= d1)].sort_values("d")
    for n, sub in wk.groupby("n"):
        acts = [fp[(n, d)] for d in sub.d if (n, d) in fp]
        if not acts or pd.isna(sub.pre.iloc[0]):
            continue
        rows3.append(dict(mp=mp, n=n, pre=float(sub.pre.iloc[0]), act=sum(acts) / len(acts), fa=n not in rostered))
d3 = pd.DataFrame(rows3)
d3["err"] = d3.act - d3.pre
print(f"\nPART 3: 2025-26 with DELCO's pregame number (the first game of each week), {len(d3)} player-weeks")
print("players DELCO put at 20-35, by matchup week:  actual minus DELCO for rostered | free agents | the gap (how much more free agents were overrated)")
for lab, lo, hi in (("weeks 1-2", 1, 2), ("weeks 3-4", 3, 4), ("weeks 5-6", 5, 6), ("weeks 7-9", 7, 9), ("weeks 10-13", 10, 13), ("weeks 14-19", 14, 19)):
    s = d3[(d3.mp >= lo) & (d3.mp <= hi) & (d3.pre >= 20) & (d3.pre < 35)]
    r, f = s[~s.fa].err, s[s.fa].err
    if len(f) < 15:
        print(f"  {lab:<12} too few free agents ({len(f)})")
        continue
    se = (r.var() / len(r) + f.var() / len(f)) ** 0.5
    print(f"  {lab:<12} rostered {r.mean():+5.1f} (n={len(r)})   free agents {f.mean():+5.1f} (n={len(f)})   gap {r.mean() - f.mean():+4.1f} (+/- {1.96 * se:.1f})")
