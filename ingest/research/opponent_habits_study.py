"""Strategy study #3 (2026-10-24): should the plan project opponents from how each owner actually manages, instead of assuming perfect lineups?

Inputs
  data/backtest_lineups_2026.csv  every 2025-26 team-week: ACTUAL points vs PLANNER points (the site's cap-aware lineup on that team's own roster, decided each morning with
                                  that morning's information, scored with real results). actual / planner = how much an owner's lineup decisions leave on the table.
  dashboard/team_usage.json       starts used per matchup, 2023-24 .. 2025-26
  ESPN team owners by season      so habits follow the OWNER, not the team slot (teams change hands)
Excluded: DRNK (Tommy did not set lineups in past seasons; he does now) and the computer-run team (CPU1, 2025-26).

Tests
  1. Is lineup efficiency a stable owner trait? split-half reliability within 2025-26 (odd vs even weeks)
  2. Is cap use stable from one season to the next for the SAME owner (and not for a new owner)?
  3. Forecast: from week 4 on, predict each team's actual points as planner x owner efficiency (estimated only from EARLIER weeks, shrunk to the league mean), vs
     perfect play (x 1.0, what the plan assumes now) and vs one league-wide factor. Scored on points error and on matchup win-probability accuracy (Brier).
Run from ingest/:  .venv/Scripts/python research/opponent_habits_study.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
DASH = R.parent.parent / "dashboard"
WEEK_SD = 253.0
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

owners = {}
for y in (2024, 2025, 2026, 2027):
    raw = League(league_id=config.LEAGUE_ID, year=y, espn_s2=config.ESPN_S2, swid=config.SWID).espn_request.league_get(params={"view": ["mTeam"]})
    for t in raw["teams"]:
        owners[(y, t["id"])] = (t.get("primaryOwner") or "", t.get("abbrev"))
TOMMY = owners[(2027, 12)][0]
human = lambda y, tid: bool(owners.get((y, tid), ("",))[0]) and owners[(y, tid)][0] != TOMMY

# ---- 1. lineup efficiency, 2025-26
bt = pd.read_csv(R / "data" / "backtest_lineups_2026.csv")
bt = bt[[human(2026, t) for t in bt.team]].copy()
bt["E"] = bt.actual / bt.planner
odd = bt[bt.mp % 2 == 1].groupby("team").E.mean()
even = bt[bt.mp % 2 == 0].groupby("team").E.mean()
print(f"lineup efficiency (actual / planner), 2025-26, {bt.team.nunique()} human teams other than DRNK:")
print("  by team:", ", ".join(f"{owners[(2026, t)][1]} {v:.3f}" for t, v in bt.groupby('team').E.mean().sort_values().items()))
print(f"  split-half reliability (odd vs even weeks): r = {np.corrcoef(odd, even.reindex(odd.index))[0, 1]:.2f}")

# ---- 2. cap use across seasons, same owner vs new owner
tu = pd.DataFrame(json.load(open(DASH / "team_usage.json"))["rows"])
tu["U"] = tu.total / tu.cap * 40
by = tu.groupby(["y", "team"]).U.mean()
same, new = [], []
for (y, t), u in by.items():
    if (y + 1, t) in by.index and human(y, t) and human(y + 1, t):
        (same if owners[(y, t)][0] == owners[(y + 1, t)][0] else new).append((u, by[(y + 1, t)]))
if len(same) > 2:
    a = np.array(same)
    print(f"\nstarts per matchup, season to next season, SAME owner (n={len(a)}): r = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.2f}  (means {a[:, 0].mean():.1f} -> {a[:, 1].mean():.1f})")
if len(new) > 1:
    a = np.array(new)
    print(f"  NEW owner (n={len(a)}): {', '.join(f'{x:.1f}->{y:.1f}' for x, y in a)}")
u26 = by.xs(2026, level="y")
corrEU = np.corrcoef(bt.groupby("team").E.mean(), u26.reindex(bt.groupby("team").E.mean().index))[0, 1]
print(f"  2025-26: lineup efficiency vs starts per matchup across teams r = {corrEU:.2f}")

# ---- 3. forecast test
opp = {}
raw = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID).espn_request.league_get(params={"view": "mMatchupScore"})
for m in raw["schedule"]:
    if "away" in m and m["matchupPeriodId"] <= 19:
        opp[(m["matchupPeriodId"], m["home"]["teamId"])] = m["away"]["teamId"]
        opp[(m["matchupPeriodId"], m["away"]["teamId"])] = m["home"]["teamId"]
allbt = pd.read_csv(R / "data" / "backtest_lineups_2026.csv").set_index(["mp", "team"])
allbt["E"] = allbt.actual / allbt.planner


def est(team, mp, k):
    prior = bt[(bt.mp < mp)]
    lg = prior.E.mean() if len(prior) else 1.0
    if k is None:
        return lg
    mine = prior[prior.team == team].E
    n = len(mine)
    return (mine.sum() + k * lg) / (n + k) if n else lg


results = {}
for name, k in (("perfect play (current)", "one"), ("league-wide factor", None), ("owner, shrink k=2", 2), ("owner, shrink k=4", 4), ("owner, shrink k=8", 8)):
    err, br = [], []
    for (mp, t), r in allbt.iterrows():
        if mp < 4 or not human(2026, t):
            continue
        f = 1.0 if k == "one" else est(t, mp, k)
        err.append(abs(r.planner * f - r.actual))
        o = opp.get((mp, t))
        if o is None or (mp, o) not in allbt.index:
            continue
        ro = allbt.loc[(mp, o)]
        fo = 1.0 if k == "one" else (est(o, mp, k) if human(2026, o) else 1.0)     # an opponent we cannot model (DRNK, CPU) is taken at face value in every variant
        p = Phi((r.planner * f - ro.planner * fo) / (WEEK_SD * math.sqrt(2)))
        br.append((p - (1.0 if r.actual > ro.actual else 0.0)) ** 2)
    results[name] = (np.mean(err), np.mean(br), len(err))
    print(f"{name:24s} points error {np.mean(err):6.1f}   win-prob Brier {np.mean(br):.4f}   (n={len(err)})")
