"""How teams in this league use the games cap (40 started games per 7-day matchup, checked at the START of each day) over the last three seasons, and who is best at it.

Data: data/league_days_{2024,2025,2026}.json (python research/pull_league_days.py YEAR; ESPN seasonId = the year the season ends, so 2026 = 2025-26).
A "start" = a player in an active lineup slot (ids 0-11) who actually played that day. Only 7-day matchups are analysed (cap 40), regular season and playoffs.

Per team-week:
  total      started games that week
  over       total > 40  (the cap lets you start anyone on a day that BEGINS at 39 or fewer, so a team can finish as high as 49)
  early      crossed 40 BEFORE the final day: every later day was locked (lost starts), the mistake the cap punishes
  final      arrived at the final day with room (<= 39) -- the way to use the cap fully
  push       arrived with room and then started 8+ players on the last day (a deliberate full final day)
  lockedout  the final day was lost because the team was already at 40+
  surplus    total minus 40 when over (how far over)
Usage: uv run python research/cap_usage_analysis.py   (from ingest/)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
YEARS = [2024, 2025, 2026]
STARTING = set(range(0, 12))
CAP7 = 40


def season_rows(year):
    f = D / f"league_days_{year}.json"
    if not f.exists():
        return []
    days = json.load(open(f, encoding="utf-8"))
    rows = []
    for d, recs in days.items():
        for r in recs:
            st = sum(1 for e in r["e"] if e[2] in STARTING and e[3])
            bench_played = sum(1 for e in r["e"] if e[2] == 12 and e[3])
            rows.append((year, int(d), r["mp"], r["team"], st, bench_played))
    return rows


rows = [x for y in YEARS for x in season_rows(y)]
df = pd.DataFrame(rows, columns=["year", "day", "mp", "team", "starts", "bench_played"]).sort_values(["year", "mp", "team", "day"])
df["cum"] = df.groupby(["year", "mp", "team"]).starts.cumsum()
df["idx"] = df.groupby(["year", "mp", "team"]).cumcount()
df["n"] = df.groupby(["year", "mp", "team"]).day.transform("count")
df["last"] = df.idx == df.n - 1
w = df.groupby(["year", "mp", "team"]).agg(total=("starts", "sum"), n=("n", "first")).reset_index()
pre = df[~df["last"]].groupby(["year", "mp", "team"]).starts.sum().rename("pre_last")
lastd = df[df["last"]].set_index(["year", "mp", "team"]).starts.rename("last_day")
w = w.set_index(["year", "mp", "team"]).join(pre).join(lastd).reset_index()
w = w[w.n == 7].copy()
w["over"] = w.total > CAP7
w["early"] = w.pre_last >= CAP7
w["final"] = w.pre_last <= CAP7 - 1
w["push"] = w.final & (w.last_day >= 8)
w["lockedout"] = w.early
w["surplus"] = (w.total - CAP7).clip(lower=0)

print(f"7-day team-weeks analysed: {len(w)}  ({', '.join(f'{y}: {int((w.year == y).sum())}' for y in YEARS)})")
print("\n=== BY SEASON ===")
by = w.groupby("year").agg(weeks=("total", "size"), avg_starts=("total", "mean"), over=("over", "mean"), early=("early", "mean"), final=("final", "mean"), push=("push", "mean"),
                           avg_over_by=("surplus", lambda s: s[s > 0].mean()))
print((by * [1, 1, 100, 100, 100, 100, 1]).round(1).to_string())

# fix: friendly overall block
tot = w.agg({"total": "mean", "over": "mean", "early": "mean", "push": "mean"})
print(f"\nALL THREE SEASONS: avg {tot.total:.1f} starts/week; finished over the cap {tot.over * 100:.0f}% of weeks; crossed it EARLY (lost later days) {tot.early * 100:.0f}%; "
      f"arrived at the last day with room and started 8+ {tot.push * 100:.0f}%")
over_w = w[w.over]
print(f"Of the {len(over_w)} weeks that finished over 40: {((over_w.pre_last <= 39)).mean() * 100:.0f}% did it on the final day (the cap's built-in room), "
      f"{(over_w.pre_last >= 40).mean() * 100:.0f}% crossed earlier than the final day (days after that were locked)")

# what a lockout costs: bench players who played on days after the team was already locked
locked_days = df[(df.cum - df.starts >= CAP7) & (df.n == 7)]
print(f"\nDays played while already at the cap (locked): {len(locked_days)}; bench-played games on those days (starts you could not use): {int(locked_days.bench_played.sum())}")

# per team
names = {}
tp = HUB / "team_profiles.json"
if tp.exists():
    for t in json.load(open(tp, encoding="utf-8"))["teams"]:
        names[t["id"]] = t["abbrev"]
lw = HUB / "league_rosters.json"
if lw.exists():
    for t in json.load(open(lw, encoding="utf-8"))["teams"]:
        names.setdefault(t["id"], t["abbrev"])
g = w.groupby("team").agg(weeks=("total", "size"), avg_starts=("total", "mean"), over=("over", "mean"), early=("early", "mean"), push=("push", "mean"), avg_final_day=("last_day", "mean"))
g["team"] = [names.get(i, str(i)) for i in g.index]
g["over"] *= 100
g["early"] *= 100
g["push"] *= 100
print("\n=== BY TEAM, all three seasons (sorted by how often they finish over the cap) ===")
print(g.sort_values("over", ascending=False)[["team", "weeks", "avg_starts", "over", "early", "push", "avg_final_day"]].round(1).to_string(index=False))
print("\nover = % of weeks finishing above 40 | early = % of weeks the cap locked out the final day | push = % of weeks arriving with room and starting 8+ on the last day")

# per team per season, over-cap rate
pv = w.groupby(["team", "year"]).over.mean().unstack() * 100
pv.index = [names.get(i, str(i)) for i in pv.index]
print("\n=== over-the-cap rate by team and season (%) ===")
print(pv.round(0).to_string())
w.to_csv(D / "cap_usage_team_weeks.csv", index=False)
