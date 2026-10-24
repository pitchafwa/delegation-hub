"""Does valuing late-week gains by their effect on WIN CHANCE (build_week_plan.py's this_week_value) win more than the flat points rule?

Uses the league's real 2025-26 matchups: for every team-matchup, the margin going into the final day and what the final day actually did.
A hypothetical last-day rental gains g points this week and costs c points later (drop risk). Two rules decide whether to make it:
  flat      g - c >= 6                               (the old rule)
  leverage  points-equivalent(g | margin, 1 day) - c >= 6
Each decision is scored in WINS with real data: the change in P(win) from g, using the empirical spread of final-day swings,
minus c points valued at the rate an average close week turns points into wins (measured from the same season's weekly margins).

Run from ingest/:  .venv/Scripts/python research/leverage_check.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
WPM = json.load(open(R / "win_prob_model.json"))
BETA, SIG = WPM["beta"], {int(k): v for k, v in WPM["sigma"].items()}
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
raw = lg.espn_request.league_get(params={"view": "mMatchupScore"})
rows = []
for m in raw["schedule"]:
    if "away" not in m or m["matchupPeriodId"] > 19:
        continue
    a, h = m["away"], m["home"]
    pa, ph = a.get("pointsByScoringPeriod") or {}, h.get("pointsByScoringPeriod") or {}
    ks = sorted(set(pa) | set(ph), key=int)
    if len(ks) < 3:
        continue
    last = ks[-1]
    pre_a = sum(v for k, v in pa.items() if k != last)
    pre_h = sum(v for k, v in ph.items() if k != last)
    for me_pre, op_pre, me_l, op_l in ((pre_a, pre_h, pa.get(last, 0), ph.get(last, 0)), (pre_h, pre_a, ph.get(last, 0), pa.get(last, 0))):
        rows.append((me_pre - op_pre, me_l - op_l, (me_pre + me_l) - (op_pre + op_l)))
d = np.array(rows)
pre, swing, final = d[:, 0], d[:, 1], d[:, 2]
print(f"{len(d)} team-matchups; final-day swing SD {swing.std():.0f} (model's 1-day sigma {SIG[1]:.0f}); final margin SD {final.std():.0f}")
# how an average close week turns points into wins: density of final margins near zero
f0 = np.mean(np.abs(final) < 60) / 120.0
f0_model = BETA * 0.3989 / SIG[7]
print(f"wins per point, average week: empirical {f0:.5f}, model {f0_model:.5f}")

rng = np.random.default_rng(7)
G = rng.uniform(5, 35, 400)
C = rng.uniform(0, 30, 400)
sw_sorted = np.sort(swing)


def p_win(margin_pre, extra):
    # empirical: share of real final-day swings that would leave this team ahead
    return 1.0 - np.searchsorted(sw_sorted, -(margin_pre + extra), side="right") / len(sw_sorted)


def lev_value(margin_pre, g):
    s = SIG[1]
    x0 = margin_pre / s
    return (Phi(x0 + BETA * g / s) - Phi(x0)) / f0_model


tot = {"flat": 0.0, "leverage": 0.0}
taken = {"flat": 0, "leverage": 0}
for mp in pre:
    base = p_win(mp, 0.0)
    for g, c in zip(G, C):
        win_gain = p_win(mp, g) - base - c * f0
        if g - c >= 6:
            tot["flat"] += win_gain
            taken["flat"] += 1
        if lev_value(mp, g) - c >= 6:
            tot["leverage"] += win_gain
            taken["leverage"] += 1
n_team_seasons = len(pre) / 19.0
for k in tot:
    print(f"{k:9s}: makes {taken[k] / len(pre) / len(G):.0%} of the candidate moves; expected wins from them per team-season {tot[k] / n_team_seasons / len(G):+.3f} per candidate-move-per-week")
print(f"difference (leverage - flat): {(tot['leverage'] - tot['flat']) / n_team_seasons / len(G):+.3f} wins per team-season, per candidate move offered each final day")
