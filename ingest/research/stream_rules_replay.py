"""Stream-rules replay (Tommy, 2026-10-25): are the plan's add rules costing us games?

Replays every 2025-26 regular-season team-week (7-day matchups) from the team's REAL roster at the start of the week. Each morning a planner in the
spirit of the site's stream manager may add free agents (players on no roster that day, 8+ games, active in the last 10 days), as a rental (held through
the week) or a keep, dropping a non-protected player or using an open spot. Lineups are set each morning with the site's cap-aware plan (the dynamic
program in backtest_weekly_planner.py), and the week is scored on what players ACTUALLY scored, with the games cap counted the ESPN way.

Decision value of a move = expected points it adds this week (rest of the week, cap-aware) - its expected future cost:
  rental  P(dropped player is gone when the week ends) x (his level - re-stream level) x 3.3 games x 6 weeks   (P from the rule being tested)
  keep    (dropped level - added level, if positive) x 3.3 x 6
A move must clear MIN_NET. Protected (never dropped): the team's top 6 by level, and anyone projecting PROTECT+.

Judged the same way for every rule (so dropping good players is never free): realized points this week + the FUTURE value of the swap, using what the
players actually did over the next 6 weeks and the MEASURED chance of losing a dropped player (reclaim_study.py), minus the same for no moves.
Variants:  none | old (MIN 6, protect 35, loss max(15%, p_taken)) | now (measured loss table) | now+min0 | now+protect45 | now+halfcost | now+lastdays (adds only in the last 3 days)
Run from ingest/:  .venv/Scripts/python research/stream_rules_replay.py   (writes data/stream_rules_replay.csv; slow: run in the background)
"""
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]
import backtest_weekly_planner as B     # noqa: E402  (loads 2025-26 data; its main does not run on import)

D = Path(__file__).resolve().parent / "data"
ADDS = 8
GPW, ROS = 3.3, 6
FA_ANCHOR, FA_SHRINK = 22.0, 0.85
N_CAND, N_DROPS = 14, 4
TABLE = [(33, [0.32, 0.38, 0.43, 0.51]), (27, [0.19, 0.27, 0.36, 0.44]), (20, [0.11, 0.18, 0.22, 0.34]), (-99, [0.07, 0.12, 0.16, 0.24])]


def lose_measured(ll, days):
    cur = next(c for lo, c in TABLE if ll >= lo)
    return float(np.interp(max(2, days), [2, 3, 4, 7], cur)) if days <= 7 else cur[3]


def lose_old(ll, days):
    p48 = 0.25 if ll >= 33 else (0.16 if ll >= 27 else 0.12)
    return max(0.15, 1 - (1 - p48) ** (max(2, days) / 2.0))


VARIANTS = {
    "none": None,
    "old": dict(min_net=6, protect=35, lose=lose_old, cost_mult=1.0, first_day=0),
    "now": dict(min_net=6, protect=35, lose=lose_measured, cost_mult=1.0, first_day=0),
    "now+min0": dict(min_net=0, protect=35, lose=lose_measured, cost_mult=1.0, first_day=0),
    "now+protect45": dict(min_net=6, protect=45, lose=lose_measured, cost_mult=1.0, first_day=0),
    "now+halfcost": dict(min_net=6, protect=35, lose=lose_measured, cost_mult=0.5, first_day=0),
    "now+lastdays": dict(min_net=6, protect=35, lose=lose_measured, cost_mult=1.0, first_day=4),
    "now+top6only": dict(min_net=6, protect=999, lose=lose_measured, cost_mult=1.0, first_day=0),
    "now+min12": dict(min_net=12, protect=35, lose=lose_measured, cost_mult=1.0, first_day=0),
}


def future_level(n, d_end):
    s = B.series.get(n)
    if s is None:
        return 0.0
    f = s[(s.GAME_DATE.dt.date > d_end) & (s.GAME_DATE.dt.date <= d_end + timedelta(days=42))]
    return float(f.fp.mean()) if len(f) else 0.0


def ev(roster, dates_rem, today, c, cap):
    opts = []
    for dd in dates_rem:
        _, _, vals, cnt = B.day_opts(roster, dd, today, True)
        opts.append((vals, cnt))
    return B.plan_value(opts, c, cap)[0]


def set_today(pl, d):
    for p in pl:
        p["played"] = (p["n"], d) in B.played_fp
        p["pts"] = B.played_fp.get((p["n"], d), 0.0)


def run_week(days, rostered_by_sp, var):
    sps = sorted(days)
    dates = [B.sp_date(sp) for sp in sps]
    cap = B.cap_for(len(sps))
    d0 = dates[0]
    r0 = [e for e in days[sps[0]] if e[2] != 13]
    roster = [B.mk_player(B.norm(e[1]), d0, dates, B.pos_by_name.get(B.norm(e[1])) or B.DEFPOS.get(e[5])) for e in r0]
    if len(roster) < 8:
        return None
    for p in roster:
        p["start_level"] = p["level"]
    lv_rank = sorted(roster, key=lambda p: -p["level"])
    c, total, adds, moves = 0, 0.0, 0, []
    for i, d in enumerate(dates):
        rem = dates[i:]
        for p in roster:
            p["level"] = B.level(p["n"], d)
        set_today(roster, d)
        if var and adds < ADDS and i >= var["first_day"] and c < cap:
            taken = rostered_by_sp.get(sps[i], set())
            pool = []
            for n, sub in B.series.items():
                if n in taken or n not in B.pos_by_name or any(p["n"] == n for p in roster):
                    continue
                past = sub[sub.GAME_DATE.dt.date < d]
                if len(past) < 8 or past.GAME_DATE.dt.date.max() < d - timedelta(days=10):
                    continue
                f = B.mk_player(n, d, rem, B.pos_by_name[n])
                f["level"] = FA_ANCHOR + FA_SHRINK * (f["level"] - FA_ANCHOR)
                ng = sum(f["sched"].values())
                if ng == 0:
                    continue
                pool.append(f)
            pool.sort(key=lambda p: -(p["level"] * sum(p["sched"].values())))
            restream = sorted((f["level"] for f in pool), reverse=True)[2] if len(pool) >= 3 else 22.0
            pool = pool[:N_CAND]
            set_today(pool, d)
            prot = {p["n"] for p in lv_rank[:6]} | {p["n"] for p in roster if p["level"] >= var["protect"]}
            for _step in range(2):                    # up to two moves a morning
                if adds >= ADDS:
                    break
                base = ev(roster, rem, d, c, cap)
                drops = sorted([p for p in roster if p["n"] not in prot], key=lambda p: p["level"])[:N_DROPS]
                if len(roster) < 15:
                    drops = [None] + drops
                best = None
                for f in pool:
                    if any(p["n"] == f["n"] for p in roster):
                        continue
                    for dr in drops:
                        wk = ev([p for p in roster if p is not dr] + [f], rem, d, c, cap) - base
                        if dr is None:
                            cost = 0.0
                        else:
                            rent = var["lose"](dr["level"], len(rem) + 2) * max(0.0, dr["level"] - restream) * GPW * ROS
                            keep = max(0.0, dr["level"] - f["level"]) * GPW * ROS
                            cost = min(rent, keep) * var["cost_mult"]
                        net = wk - cost
                        if best is None or net > best[0]:
                            best = (net, wk, f, dr, restream)
                if not best or best[0] < var["min_net"]:
                    break
                net, wk, f, dr, rs = best
                roster = [p for p in roster if p is not dr] + [f]
                adds += 1
                moves.append((f, dr, rs, len(rem)))
        # lineup today with the cap-aware plan, actual points
        opts, first = [], None
        for j, dd in enumerate(rem):
            cands, acc, vals, cnt = B.day_opts(roster, dd, d, True)
            opts.append((vals, cnt))
            if j == 0:
                first = (cands, acc)
        if c < cap:
            m = B.plan_value(opts, c, cap)[1]
            cands, acc = first
            total += sum(cands[acc[k]][2]["pts"] for k in range(m))
            c += m
    # future value of the week's swaps, judged on what really happened next and the MEASURED loss chance (same yardstick for every variant)
    fut = 0.0
    d_end = dates[-1]
    for f, dr, rs, nrem in moves:
        if dr is None:
            continue
        dl, fl = future_level(dr["n"], d_end), future_level(f["n"], d_end)
        rent = lose_measured(dr.get("start_level", dr["level"]), nrem + 2) * max(0.0, dl - rs) * GPW * ROS
        keep = max(0.0, dl - fl) * GPW * ROS
        fut -= min(rent, keep)
    return dict(points=total, starts=c, adds=adds, future=fut)


def main():
    W = B.weeks()
    rostered_by_sp = {}
    for sp, recs in B.league.items():
        rostered_by_sp[int(sp)] = {B.norm(e[1]) for r in recs for e in r["e"]}
    only = os.environ.get("ONLY_TEAM")
    out = []
    keys = [(mp, team) for (mp, team), days in sorted(W.items()) if 2 <= mp <= 19 and mp != 17 and len(days) == 7 and (not only or team == int(only))]
    step = int(os.environ.get("STEP", "1"))
    keys = keys[::step]
    print(f"{len(keys)} team-weeks x {len(VARIANTS)} variants", flush=True)
    for k, (mp, team) in enumerate(keys):
        for name, var in VARIANTS.items():
            r = run_week(W[(mp, team)], rostered_by_sp, var)
            if r:
                out.append(dict(mp=mp, team=team, variant=name, **r))
        if k % 10 == 9:
            pd.DataFrame(out).to_csv(D / "stream_rules_replay.csv", index=False)
            print(f"  {k + 1} done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(D / "stream_rules_replay.csv", index=False)
    report(df)


def report(df):
    base = df[df.variant == "none"].set_index(["mp", "team"])
    rows = []
    for name in VARIANTS:
        x = df[df.variant == name].set_index(["mp", "team"])
        j = x.join(base, rsuffix="_b", how="inner")
        gain = (j.points - j.points_b) + j.future
        rows.append(dict(variant=name, weeks=len(j), adds=j.adds.mean(), starts=j.starts.mean(), starts_gain=(j.starts - j.starts_b).mean(),
                         week_pts_gain=(j.points - j.points_b).mean(), future=j.future.mean(), net_gain=gain.mean(), se=gain.std() / max(1, len(j)) ** 0.5))
    r = pd.DataFrame(rows)
    print("\nPer team-week, vs making no moves (points; 'future' = expected season-long cost of the drops, judged on what really happened):")
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    if "--report" in ARGV:
        report(pd.read_csv(D / "stream_rules_replay.csv"))
    else:
        main()
