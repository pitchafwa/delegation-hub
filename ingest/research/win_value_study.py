"""Strategy study #2 (2026-10-24): is every win worth the same? If not, should the plan push harder in weeks whose result moves title odds more?

On the league's real 2025-26 season, before each regular-season week w (standings through w-1), for every team:
  V(t,w) = title odds if t WINS week w  -  title odds if t LOSES week w      (forced result, everything else simulated: causal, not conditional)
  F(t,w) = average V(t,w') over t's later regular-season weeks (what a point spread over the future is worth, per win)
Then a policy test like leverage_check.py: each team-week gets candidate moves that add g points this week and cost c points later. Rules:
  flat      take if g - c >= 6                                    (every win treated alike)
  weighted  take if g x V(t,w)/F(t,w) - c >= 6                     (this week's points scaled by how much this week's win matters vs the future's)
Both scored in TITLE ODDS: g's effect on this week's win chance x V(t,w), minus c's effect on future wins x F(t,w).
Run from ingest/:  .venv/Scripts/python research/win_value_study.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_playoff_odds as B
from playoff_sim import simulate

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
WPM = json.load(open(R / "win_prob_model.json"))
BETA, SIG7 = WPM["beta"], WPM["sigma"]["7"]
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
N = 6000

teams, games = B.pull(2026)
ids = sorted(teams)
weeks = sorted({g["week"] for g in games})
lens = {w: next((g["n"] for g in games if g["week"] == w and g["n"]), 7) for w in weeks}
hist = {t: {} for t in ids}
for g in games:
    hist[g["a"]][g["week"]] = g["sa"]
    hist[g["b"]][g["week"]] = g["sb"]
PR = dict(tau=100.0, sigma_e=175.0, bias=0.0)

V, MU = {}, {}
for w in range(2, B.REG_LAST + 1):
    c = w - 1
    wins, pf = B.standings(games, upto=c)
    for t in ids:
        wins.setdefault(t, 0)
        pf.setdefault(t, 0.0)
    per7 = {t: np.mean([hist[t][k] / lens[k] * 7 for k in range(1, c + 1)]) for t in ids}
    lgm = np.mean(list(per7.values()))
    mu = {(t, k): (0.5 * per7[t] + 0.5 * lgm) * lens[k] / 7 for t in ids for k in weeks}
    rest = [(g["week"], g["a"], g["b"]) for g in games if g["week"] > c]
    this = [(k, a, b) for k, a, b in rest if k == w]
    later = [p for p in rest if p[0] != w]
    for k, a, b in this:
        for me, op in ((a, b), (b, a)):
            res = {}
            for won in (True, False):
                w2, pf2 = dict(wins), dict(pf)
                w2[me if won else op] += 1
                pf2[me] += mu[(me, w)]
                pf2[op] += mu[(op, w)]
                pairs = later + [(k2, a2, b2) for k2, a2, b2 in this if {a2, b2} != {a, b}]
                r = simulate(ids, w2, pf2, pairs, mu, lens, B.REG_LAST, n=N, seed=11, **PR)
                res[won] = float(r["title"][ids.index(me)])
            V[(me, w)] = res[True] - res[False]
            MU[(me, w)] = mu[(me, w)] - mu[(op, w)]
    print(f"week {w}: mean value of a win {np.mean([V[(t, w)] for t in ids]):.3f} title odds", flush=True)

F = {}
for t in ids:
    for w in range(2, B.REG_LAST):
        fut = [V[(t, k)] for k in range(w + 1, B.REG_LAST + 1) if (t, k) in V]
        F[(t, w)] = float(np.mean(fut)) if fut else None
vals = np.array([v for v in V.values()])
print(f"\nvalue of a win (title odds): mean {vals.mean():.3f}, median {np.median(vals):.3f}, 10th pct {np.percentile(vals, 10):.3f}, 90th pct {np.percentile(vals, 90):.3f}")
ratio = np.array([V[k] / F[k] for k in V if F.get(k)])
print(f"this week vs the average later week: under 0.5x in {np.mean(ratio < 0.5):.0%} of team-weeks, over 1.5x in {np.mean(ratio > 1.5):.0%}, over 2x in {np.mean(ratio > 2):.0%}")

# policy test
rng = np.random.default_rng(3)
G = rng.uniform(5, 40, 300)
C = rng.uniform(0, 30, 300)
dens0 = BETA * 0.3989 / SIG7
tot = {"flat": 0.0, "weighted": 0.0}
take = {"flat": 0, "weighted": 0}
n_tw = 0
for k, v in V.items():
    f = F.get(k)
    if not f or f <= 0:
        continue
    n_tw += 1
    m = MU[k]
    x0 = BETA * m / SIG7
    for g, c in zip(G, C):
        dwp = Phi(x0 + BETA * g / SIG7) - Phi(x0)
        gain = dwp * v - c * dens0 * f
        if g - c >= 6:
            tot["flat"] += gain
            take["flat"] += 1
        if g * (v / f) - c >= 6:
            tot["weighted"] += gain
            take["weighted"] += 1
seasons = n_tw / (len(ids) * (B.REG_LAST - 2))
for kk in tot:
    print(f"{kk:9s}: takes {take[kk] / n_tw / len(G):.0%} of candidate moves; title odds gained per team-season per candidate move each week {tot[kk] / len(G) / len(ids):+.4f}")
print(f"difference (weighted - flat): {(tot['weighted'] - tot['flat']) / len(G) / len(ids):+.4f} title odds per team-season, per candidate move offered each week")
json.dump({"V": {f"{a}:{b}": round(x, 4) for (a, b), x in V.items()}}, open(R / "data" / "win_value_study.json", "w"))
