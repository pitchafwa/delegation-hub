"""Strategy study #5 (2026-10-24): what is depth worth under the 40-start cap, and what is a streaming spot worth?

Replays every 2025-26 regular-season team-week (real week-start rosters from league_days_2026.json, real points, the site's cap-aware lineup logic with
that-morning availability; backtest_weekly_planner.py's realized_value):
  A. DEPTH   remove the team's k-th best player (by level at week start, k = 6..13) and replay: points lost = what that roster spot was worth that week.
             The same with no cap, to show how much the cap shrinks depth value.
  B. STREAM  turn the s weakest spots (s = 1..3) into streaming spots: each day a spot whose player has no game swaps to the best free agent who plays
             that day (free agents = not rostered anywhere at week start, 8+ recent games), within the league's 7 adds a week. Compared with keeping those players.
  C. CONSOLIDATION (what it implies for trades): two mid players -> one better player + a streaming spot, from A and B's numbers.
Run from ingest/:  .venv/Scripts/python research/depth_value_study.py   (slow: thousands of week replays)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0]]
import backtest_weekly_planner as B

sys.stdout.reconfigure(encoding="utf-8")
ADDS = 7


def realized(roster, dates, cap):
    """B.realized_value, but a player only counts on the days he is on the roster (p['held'] = set of dates, or None = all week)"""
    c, total = 0, 0.0
    for i, d in enumerate(dates):
        if c >= cap:
            break
        for p in roster:
            held = p.get("held")
            on = held is None or d in held
            p["played"] = on and (p["n"], d) in B.played_fp
            p["pts"] = B.played_fp.get((p["n"], d), 0.0) if on else 0.0
        opts, first = [], None
        for j, dd in enumerate(dates[i:]):
            for p in roster:                                   # later days: scheduled AND on the roster that day
                held = p.get("held")
                if held is not None:
                    p["sched"] = {x: (x in held and x in B.team_days.get(p["team"], set())) for x in dates}
            cands, acc, vals, cnt = B.day_opts(roster, dd, d, True)
            opts.append((vals, cnt))
            if j == 0:
                first = (cands, acc)
        m = B.plan_value(opts, c, cap)[1]
        cands, acc = first
        total += sum(cands[acc[k]][2]["pts"] for k in range(m))
        c += m
    return total


REALISTIC = True     # 2026-10-24 (Tommy's challenge): only players actually available that day: not on any roster that day, and not dropped in the last two days (waivers)
AVAIL = {}


def available(n, d):
    if not REALISTIC:
        return True
    return AVAIL.get((n, d), True)


def streamers(pool, dates, slots, adds=ADDS):
    """greedy daily streaming: slot players held over consecutive days; swap when the held player has no game today and adds remain"""
    out, used, left = [], set(), adds
    cur = [None] * slots
    for d in dates:
        for s in range(slots):
            h = cur[s]
            if h is not None and d in B.team_days.get(h["team"], set()):
                h["held"].add(d)
                continue
            if left <= 0:
                if h is not None:
                    h["held"].add(d)
                continue
            best = next((f for f in pool if f["n"] not in used and d in B.team_days.get(f["team"], set()) and available(f["n"], d)), None)
            if best is None:
                continue
            nf = dict(best, held={d})
            used.add(best["n"])
            out.append(nf)
            cur[s] = nf
            left -= 1
    return out, left


W = B.weeks()
rostered_by_sp = {int(sp): {B.norm(e[1]) for r in recs for e in r["e"]} for sp, recs in B.league.items()}
for sp in rostered_by_sp:
    d = B.sp_date(sp)
    recent = rostered_by_sp[sp] | rostered_by_sp.get(sp - 1, set()) | rostered_by_sp.get(sp - 2, set())     # rostered today, or dropped within two days (still on waivers)
    for n in recent:
        AVAIL[(n, d)] = False
rows = []
import time
t0 = time.time()
for (mp, team), days in sorted(W.items()):
    if mp > 19 or mp == 17 or mp == 1:
        continue
    sps = sorted(days)
    dates = [B.sp_date(sp) for sp in sps]
    d0, cap = dates[0], B.cap_for(len(sps))
    r0 = [e for e in days[sps[0]] if e[2] != 13]
    roster = [B.mk_player(B.norm(e[1]), d0, dates, B.pos_by_name.get(B.norm(e[1])) or B.DEFPOS.get(e[5])) for e in r0]
    if len(roster) < 12:
        continue
    roster.sort(key=lambda p: -p["level"])
    base = realized([dict(p) for p in roster], dates, cap)
    base_nc = realized([dict(p) for p in roster], dates, 999)
    row = {"mp": mp, "team": team, "base": base, "n": len(roster)}
    for k in range(6, min(14, len(roster) + 1)):
        rest = [dict(p) for i, p in enumerate(roster) if i != k - 1]
        row[f"d{k}"] = base - realized(rest, dates, cap)
        row[f"nc{k}"] = base_nc - realized([dict(p) for i, p in enumerate(roster) if i != k - 1], dates, 999)
        row[f"lvl{k}"] = roster[k - 1]["level"]
    taken = rostered_by_sp[sps[0]]
    pool = []
    for n, sub in B.series.items():
        if n in taken or n not in B.pos_by_name:
            continue
        past = sub[sub.GAME_DATE.dt.date < d0]
        if len(past) < 8 or past.GAME_DATE.dt.date.max() < d0 - pd.Timedelta(days=10).to_pytimedelta():
            continue
        pool.append(B.mk_player(n, d0, dates, B.pos_by_name[n]))
    pool.sort(key=lambda p: -p["level"])
    pool = pool[:60]
    for s in (1, 2, 3):
        keep = [dict(p) for p in roster[:len(roster) - s]]
        st, _left = streamers(pool, dates, s)
        row[f"s{s}"] = realized(keep + st, dates, cap) - base
        row[f"bottom{s}"] = sum(p["level"] for p in roster[-s:]) / s
        # the s-th streaming spot on its own: s-1 streamers vs s streamers, with the s weakest players gone either way (value of that spot vs leaving it empty)
        st_prev, _ = streamers(pool, dates, s - 1) if s > 1 else ([], None)
        row[f"slot{s}"] = realized([dict(p) for p in roster[:len(roster) - s]] + st, dates, cap) - realized([dict(p) for p in roster[:len(roster) - s]] + st_prev, dates, cap)
    rows.append(row)
    if len(rows) % 20 == 0:
        print(f"{len(rows)} team-weeks ({time.time() - t0:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(B.D / "depth_value_study.csv", index=False)
print(f"\n{len(df)} team-weeks replayed in {time.time() - t0:.0f}s\n")
print("A. what the k-th best player is worth per week (points lost if he is removed):")
for k in range(6, 14):
    if f"d{k}" in df:
        print(f"  #{k:<2} (avg level {df[f'lvl{k}'].mean():4.1f}): with the cap {df[f'd{k}'].mean():6.1f}   with no cap {df[f'nc{k}'].mean():6.1f}")
print("\nB. streaming spots instead of the weakest players (points per week vs keeping them):")
for s in (1, 2, 3):
    print(f"  {s} spot(s) (players replaced avg level {df[f'bottom{s}'].mean():.1f}): {df[f's{s}'].mean():+6.1f} per week  (better in {(df[f's{s}'] > 0).mean():.0%} of weeks)")
print("\nC. each streaming spot on its own, vs leaving that roster spot empty (7 adds a week shared by all streaming spots):")
for s in (1, 2, 3):
    print(f"  streaming spot #{s}: {df[f'slot{s}'].mean():+6.1f} per week")
lv = np.concatenate([df[f"lvl{k}"].to_numpy() for k in range(6, 14) if f"lvl{k}" in df])
dv = np.concatenate([df[f"d{k}"].to_numpy() for k in range(6, 14) if f"d{k}" in df])
_m = ~np.isnan(lv) & ~np.isnan(dv)
b_, a_ = np.polyfit(lv[_m], dv[_m], 1)
print(f"\nheld player's weekly value ~ {a_:.1f} + {b_:.2f} x his points per game (ranks 6-13, with the cap)")
for s in (1, 2, 3):
    print(f"  streaming spot #{s} is worth about as much as holding a {(df[f'slot{s}'].mean() - a_) / b_:.0f}-point-per-game player")
