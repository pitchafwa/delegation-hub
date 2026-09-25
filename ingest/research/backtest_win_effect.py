"""How many matchups would flip if a team used the planner's lineups while everyone else played as they actually did?
Reads data/backtest_lineups_2026.csv (from backtest_weekly_planner.py) and the league's real 2025-26 schedule from ESPN."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
raw = lg.espn_request.league_get(params={"view": "mMatchupScore"})
pairs = []
for m in raw["schedule"]:
    if "away" in m and m["matchupPeriodId"] <= 19 and m["matchupPeriodId"] != 17:
        pairs.append((m["matchupPeriodId"], m["home"]["teamId"], m["home"]["totalPoints"], m["away"]["teamId"], m["away"]["totalPoints"]))
opp = {}
for mp, h, hp, a, ap in pairs:
    opp[(mp, h)] = (a, ap, hp)
    opp[(mp, a)] = (h, hp, ap)
bt = pd.read_csv(R / "data" / "backtest_lineups_2026.csv")
rows = []
for r in bt.itertuples():
    o = opp.get((r.mp, r.team))
    if not o:
        continue
    oid, opp_pts, my_pts = o
    rows.append(dict(team=r.team, mp=r.mp, actual_win=my_pts > opp_pts, planner_win=r.planner > opp_pts, greedy_win=r.greedy > opp_pts, margin_actual=my_pts - opp_pts, margin_planner=r.planner - opp_pts))
d = pd.DataFrame(rows)
print(f"{len(d)} team-matchups")
g = d.groupby("team").agg(actual_wins=("actual_win", "sum"), planner_wins=("planner_win", "sum"), greedy_wins=("greedy_win", "sum"), n=("mp", "size"))
g["extra_wins"] = g.planner_wins - g.actual_wins
g["actual_pct"] = (g.actual_wins / g.n * 100).round(0)
g["planner_pct"] = (g.planner_wins / g.n * 100).round(0)
print(g.sort_values("extra_wins", ascending=False).to_string())
print(f"\nleague-wide: actual win% (each matchup counted from both sides) {d.actual_win.mean():.1%}; with planner lineups {d.planner_win.mean():.1%}; greedy {d.greedy_win.mean():.1%}")
print(f"average extra wins per team over {int(g.n.mean())} regular-season matchups: {g.extra_wins.mean():+.1f}")
me = d[d.team == 11]
print("\nDRNK (team 11) week by week: margin actual -> planner")
print(me[["mp", "margin_actual", "margin_planner", "actual_win", "planner_win"]].round(0).to_string(index=False))
