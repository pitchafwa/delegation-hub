"""Counterfactual replay for the trade analyzer's SEASON part (RESEARCH_trade_value.md section 4.1).
At checkpoints in past seasons we take the real rosters, apply hypothetical trades (1-for-1, 2-for-1, 2-for-2), and compare each predictor's verdict with what the traded
players ACTUALLY did for the next 8 matchup weeks, scored the way this league scores (up to 10 starters a day, 40-game cap a week, perfect-hindsight lineup, same for all).
Predictors (all see only the days before the checkpoint):
  LINEUP : roster-aware. Expected points = trailing points per game x trailing play rate; value = best 10-a-day / 40-a-week lineup of those expectations, replacing a dropped player.
  SUM    : naive market-style, difference in the sum of expected points per game of the players moved (no lineup, no roster spot).
  LEVEL  : sum of trailing points per game only (ignores play rate).
Metric: Spearman correlation with the realized change in the hindsight lineup, and sign accuracy where |realized change| is material (>= 40 points over 8 weeks).
Run: uv run python research/trade_replay.py        (from ingest/)
Limits: tests the roster-structure logic and per-game level forecasting, not the title-odds conversion, the keeper side or acceptance. Positions are ignored.
"""
import json
import random
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
SEASONS = [2024, 2025, 2026]
CHECK_MP = [5, 9, 13]      # checkpoint = start of this matchup period
HORIZON = 8                # matchup periods
SLOTS, CAP, ROSTER_MAX = 10, 40, 15
random.seed(7)


def load(y):
    raw = json.load(open(D / f"league_days_{y}.json"))
    days = {int(k): v for k, v in raw.items()}
    # per day: team -> {pid: (slot, gp, pts)}, and day -> matchup period
    mp = {}
    t = {}
    for d, recs in days.items():
        for r in recs:
            mp.setdefault(d, r["mp"])
            t.setdefault(d, {})[r["team"]] = {e[0]: (e[2], e[3], e[4], e[1]) for e in r["e"]}
    return mp, t


def best_lineup(daily):
    """daily: list over days of week -> list of expected/actual points of players available that day. Hindsight lineup: top 10 a day, then top 40 of the week."""
    picks = []
    for pts in daily:
        picks += sorted(pts, reverse=True)[:SLOTS]
    return sum(sorted(picks, reverse=True)[:CAP])


def week_days(mp, m):
    return sorted(d for d, x in mp.items() if x == m)


def run():
    rows = []
    for y in SEASONS:
        mp, t = load(y)
        if not mp:
            continue
        maxmp = max(mp.values())
        for cm in CHECK_MP:
            if cm + HORIZON - 1 > maxmp:
                continue
            first = min(d for d, x in mp.items() if x == cm)
            past = [d for d in sorted(mp) if d < first and d >= first - 28]
            fut = [d for d in sorted(mp) if cm <= mp[d] < cm + HORIZON]
            if not past or not fut or first not in t:
                continue
            ros = {tm: set(es) for tm, es in t[first].items()}
            # trailing stats
            ppg, rate = {}, {}
            for tm in ros:
                for pid in ros[tm]:
                    pts, g, on = 0.0, 0, 0
                    for d in past:
                        e = t.get(d, {}).get(tm, {}).get(pid) or next((t.get(d, {}).get(o, {}).get(pid) for o in t.get(d, {}) if pid in t[d][o]), None)
                        if e is None:
                            continue
                        if e[0] != 13:          # not on IR
                            on += 1
                            if e[1]:
                                g += 1
                                pts += e[2] or 0
                    ppg[pid] = (pts / g) if g else 20.0
                    rate[pid] = (g / on) if on >= 6 else 0.6
            # realized daily points of each player over the horizon, wherever he is rostered
            real = {}
            for d in fut:
                for tm, es in t.get(d, {}).items():
                    for pid, e in es.items():
                        if e[1] and e[2] is not None:
                            real.setdefault(pid, {})[d] = e[2]
            wk = {m: [d for d in fut if mp[d] == m] for m in range(cm, cm + HORIZON)}

            def realized(pids):
                s = 0.0
                for m, ds in wk.items():
                    s += best_lineup([[real.get(p, {}).get(d) for p in pids if real.get(p, {}).get(d) is not None] for d in ds])
                return s

            def expected(pids):
                # expected lineup: each player contributes ppg*rate on every day of the horizon with probability 1 spread: use rate as daily play probability scaled to ~3.5 games/week
                s = 0.0
                for m, ds in wk.items():
                    exp = sorted([ppg[p] for p in pids for _ in range(1)], reverse=True)
                    # a week offers len(ds) days; a player appears on ~ (3.5/7) of them when he plays; treat weekly games = min(len(ds)*0.5, 3.5+) scaled by rate
                    games = {p: len(ds) * 0.5 * rate[p] for p in pids}
                    # fill 10 slots per day, cap 40: fractional-games greedy
                    cap = CAP
                    slots = SLOTS * len(ds)
                    order = sorted(pids, key=lambda p: -ppg[p])
                    for p in order:
                        if cap <= 0 or slots <= 0:
                            break
                        g = min(games[p], cap, slots)
                        s += g * ppg[p]
                        cap -= g
                        slots -= g
                return s

            teams = list(ros)
            for _ in range(160):
                a, b = random.sample(teams, 2)
                na, nb = random.choice([(1, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)])
                top = lambda tm: sorted(ros[tm], key=lambda p: -ppg[p])[:10]
                ga = random.sample(top(a), min(na, len(top(a))))
                gb = random.sample(top(b), min(nb, len(top(b))))
                new_a = (ros[a] - set(ga)) | set(gb)
                new_b = (ros[b] - set(gb)) | set(ga)

                def trim(s):
                    s = set(s)
                    while len(s) > ROSTER_MAX:
                        s.remove(min(s, key=lambda p: ppg[p] * rate[p]))
                    return s
                new_a, new_b = trim(new_a), trim(new_b)
                for who, old, new, give, get in ((a, ros[a], new_a, ga, gb), (b, ros[b], new_b, gb, ga)):
                    truth = realized(new) - realized(old)
                    if abs(truth) < 1e-9 and not get:
                        continue
                    lineup = expected(new) - expected(old)
                    sm = sum(ppg[p] * rate[p] for p in get) - sum(ppg[p] * rate[p] for p in give)
                    lv = sum(ppg[p] for p in get) - sum(ppg[p] for p in give)
                    rows.append((y, cm, len(give), len(get), truth, lineup, sm, lv))
    return rows


rows = run()
A = np.array([[r[4], r[5], r[6], r[7]] for r in rows])
print("trade sides evaluated:", len(rows), " by season:", {y: sum(1 for r in rows if r[0] == y) for y in SEASONS})


def report(name, mask):
    sub = A[mask]
    if len(sub) < 20:
        return
    out = [f"{name:<28} n={len(sub):<5}"]
    mat = np.abs(sub[:, 0]) >= 40
    for i, lab in ((1, "LINEUP"), (2, "SUM"), (3, "LEVEL")):
        rho = spearmanr(sub[:, 0], sub[:, i])[0]
        acc = float(np.mean(np.sign(sub[mat, 0]) == np.sign(sub[mat, i]))) if mat.sum() else float("nan")
        out.append(f"{lab}: rho {rho:.2f} sign {acc:.2f}")
    print("  ".join(out))


ng = np.array([r[2] for r in rows])
nt = np.array([r[3] for r in rows])
report("all", np.ones(len(rows), bool))
report("even swaps (n=m)", ng == nt)
report("I get MORE players (consolidate away from)", nt > ng)
report("I get FEWER players (consolidation)", nt < ng)
for y in SEASONS:
    report(f"season {y}", np.array([r[0] == y for r in rows]))
