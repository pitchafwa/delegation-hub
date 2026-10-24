"""Strategy study #6 (2026-10-24): how much is a point in a fantasy-playoff week (20-22) worth compared with a regular-season point, and how much does the NBA
playoff-week schedule (games per team) move a player's value?

1. Title value of points, from the season simulation (playoff_sim) on the current standings and projections: add 20 points to MY team in one week only and
   measure the change in title odds, for each remaining regular-season week and each playoff week. Ratio = how much a playoff-week point is worth.
2. Schedule spread: games per NBA team in weeks 20-22 (dashboard/nba_schedule.json), and what an extra playoff game is worth for a typical starter after
   the 40-start cap (the cap binds more in 4-game weeks; measured with the trade analyzer's weekly lineup engine on real rosters).
Run from ingest/:  .venv/Scripts/python research/playoff_schedule_study.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_playoff_odds as B
from playoff_sim import simulate

sys.stdout.reconfigure(encoding="utf-8")
DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"
N = 40000

teams, games = B.pull(2027)
sp = json.load(open(DASH / "schedule_plan.json", encoding="utf-8"))
ids = sorted(teams)
wins, pf = B.standings(games)
for t in ids:
    wins.setdefault(t, 0)
    pf.setdefault(t, 0.0)
lens = {w["id"]: w["days"] for w in sp["calendar"]}
mu = {(t["id"], int(w)): float(v) for t in sp["teams"] for w, v in t["proj"].items()}
pairs = [(g["week"], g["a"], g["b"]) for g in games if not g["done"]]
my = next(t["id"] for t in sp["teams"] if t["abbrev"] == sp["my_abbrev"])
basis = sp["teams"][0].get("proj_basis")
PR = {"tau": B.PARAMS["tau"], "sigma_e": B.PARAMS["sigma_e"], "bias": B.PARAMS["bias_blend"] if basis == "blend" else B.PARAMS["bias_preseason"]}
mi = ids.index(my)
base = simulate(ids, wins, pf, pairs, mu, lens, B.REG_LAST, n=N, seed=5, **PR)
t0 = float(base["title"][mi])
print(f"my title odds now {t0:.1%}, playoffs {float(base['made'][mi]):.1%}\n")
rows = []
weeks = sorted({w for w, _, _ in pairs}) + [20, 21, 22]
for w in weeks:
    mu2 = dict(mu)
    mu2[(my, w)] = mu2.get((my, w), 0.0) + 20.0
    r = simulate(ids, wins, pf, pairs, mu2, lens, B.REG_LAST, n=N, seed=5, **PR)
    rows.append((w, float(r["title"][mi]) - t0))
reg = [d for w, d in rows if w <= B.REG_LAST]
po = {w: d for w, d in rows if w > B.REG_LAST}
print("title odds gained from +20 points in ONE week:")
print(f"  a regular-season week (average of weeks {min(w for w, _ in rows)}-{B.REG_LAST}): {np.mean(reg) * 100:+.2f} pts")
for w, d in po.items():
    print(f"  playoff week {w}: {d * 100:+.2f} pts   ({d / np.mean(reg):.1f}x a regular-season week)")
json.dump({"reg_avg": np.mean(reg), "playoff": po, "by_week": dict(rows)}, open(Path(__file__).resolve().parent / "data" / "playoff_schedule_study.json", "w"), indent=1)
