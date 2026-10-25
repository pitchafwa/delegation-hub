"""Net value of an owner's in-week moves (2025-26): points with his real day-by-day rosters minus points if he had kept his week-start roster,
both with the site's lineup logic (backtest_weekly_planner). Gross 'stream points' (opponent_adds_study.py) overstate the gain because an added player
takes the place of someone. Writes data/opponent_adds_net.json (with every team-week) and the committed summary opponent_streaming.json (read by build_team_usage.py).
Run from ingest/:  .venv/Scripts/python research/opponent_adds_net.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.argv = [sys.argv[0]]
import backtest_weekly_planner as B
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
own26 = {t["id"]: t.get("primaryOwner") or "" for t in League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID).espn_request.league_get(params={"view": ["mTeam"]})["teams"]}
W = B.weeks()
rows = []
for (mp, team), days in sorted(W.items()):
    if mp > 19 or mp == 17 or mp == 1 or not own26.get(team):
        continue
    sps = sorted(days)
    dates = [B.sp_date(sp) for sp in sps]
    cap = B.cap_for(len(sps))
    actual_rosters, _ = B.replay(days, "planner")                      # his real rosters day by day
    d0 = dates[0]
    r0 = [e for e in days[sps[0]] if e[2] != 13]
    fixed = [B.mk_player(B.norm(e[1]), d0, dates, B.pos_by_name.get(B.norm(e[1])) or B.DEFPOS.get(e[5])) for e in r0]
    fixed_pts = B.realized_value([dict(p) for p in fixed], dates, cap)
    rows.append({"mp": mp, "team": team, "owner": own26[team], "net": actual_rosters - fixed_pts, "days": len(sps)})
by = defaultdict(list)
for r in rows:
    by[r["owner"]].append(r["net"] * 7.0 / r["days"])
lg = float(np.mean([r["net"] * 7.0 / r["days"] for r in rows]))
print(f"{len(rows)} team-weeks; league-average net gain from in-week moves: {lg:+.0f} points per 7 days")
for o, v in sorted(by.items(), key=lambda kv: -np.mean(kv[1])):
    t = next(r["team"] for r in rows if r["owner"] == o)
    print(f"  team {t:2d} ({o[:7]}): {np.mean(v):+5.0f} per week  (n={len(v)})")
# how much to trust an owner's own average: week-to-week noise vs real differences between owners -> shrink weight k (in weeks)
noise = float(np.mean([np.var(v, ddof=1) for v in by.values()]))
means = np.array([np.mean(v) for v in by.values()])
true_var = max(1.0, float(np.var(means, ddof=1)) - noise / np.mean([len(v) for v in by.values()]))
k = noise / true_var
half = np.array([(np.mean(v[0::2]), np.mean(v[1::2])) for v in by.values()])
print(f"weekly noise SD {noise ** .5:.0f}, real owner spread SD {true_var ** .5:.0f}  ->  shrink weight {k:.1f} weeks;  split-half r = {np.corrcoef(half[:, 0], half[:, 1])[0, 1]:.2f}")
json.dump({"league_per7": lg, "owner_per7": {o: float(np.mean(v)) for o, v in by.items()}, "n": {o: len(v) for o, v in by.items()}, "k": k,
           "rows": rows}, open(R / "data" / "opponent_adds_net.json", "w"), indent=1)
json.dump({"note": "opponent_adds_net.py, 2025-26: net points per 7 days from each owner's in-week adds (vs keeping his week-start roster)", "league_per7": round(lg, 1), "k": round(k, 1),
           "owner_per7": {o: round(float(np.mean(v)), 1) for o, v in by.items()}, "n": {o: len(v) for o, v in by.items()}}, open(R / "opponent_streaming.json", "w"), indent=1)
