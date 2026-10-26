"""Cap-risk replay (Tommy, 2026-10-26): does a lineup planner that weighs every possible number of no-shows beat the current one?

CURRENT (build_week_plan.py plan_team / backtest_weekly_planner.plan_value): a dynamic program over starts used, where starting k players on a future day
moves the count by their EXPECTED starts (sum of chances to play, rounded). It hides the cliff: ten starters 'use 8.5', but if all ten play the team crosses
the cap a day early and the remaining days count for nothing.
STOCHASTIC: the same dynamic program, but each future day's starts are a distribution (each chosen starter plays with his own chance, independently), and the
value of a choice is averaged over every outcome: V(day, c) = max_k [ points of the top k + sum_s P(s of k play) x V(day+1, c+s) ]. Today's starts are known.

Replay: every 2025-26 regular-season team-week with the team's REAL rosters day by day; each morning the planner picks how many of the best available
players to start (availability today known, later days by schedule x measured chance to play); scored on ACTUAL points with ESPN's cap rule.
Run from ingest/:  .venv/Scripts/python research/cap_risk_replay.py   (writes data/cap_risk_replay.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]
import backtest_weekly_planner as B    # noqa: E402

D = Path(__file__).resolve().parent / "data"


def pb_dist(ps):
    d = np.array([1.0])
    for q in ps:
        d = np.convolve(d, [1 - q, q])
    return d


def plan_value_stoch(days_opts, c0, cap):
    """days_opts: per day (vals prefix, list of chances of the accepted candidates in order). Returns (best expected points, k for the first day)."""
    NS = int(cap) + 12
    memo = {}

    def V(di, c):
        if di == len(days_opts) or c >= cap:
            return 0.0, 0
        key = (di, c)
        if key in memo:
            return memo[key]
        vals, ps = days_opts[di]
        best = (-1e9, 0)
        for k in range(len(vals)):
            dist = pb_dist(ps[:k])
            v = vals[k] + sum(pr * V(di + 1, min(c + s, NS))[0] for s, pr in enumerate(dist) if pr > 1e-9)
            if v > best[0] + 1e-9:
                best = (v, k)
        memo[key] = best
        return best

    return V(0, int(round(c0)))


def replay(team_days_entries, mode):
    sps = sorted(team_days_entries)
    dates = [B.sp_date(sp) for sp in sps]
    cap = B.cap_for(len(sps))
    c, total = 0, 0.0
    info = {}
    for sp, d in zip(sps, dates):
        pls = []
        for e in team_days_entries[sp]:
            pid, name, slot, gp, pts, dpos, inj = e
            if slot == 13:
                continue
            n = B.norm(name)
            pls.append({"key": pid, "n": n, "elig": B.elig_from(B.pos_by_name.get(n), dpos), "level": None, "played": bool(gp) and pts is not None,
                        "pts": pts or 0.0, "team": B.team_on(n, d)})
        info[d] = pls
    for i, d in enumerate(dates):
        if c >= cap:
            break
        today_roster = info[d]
        for p in today_roster:
            p["level"] = B.level(p["n"], d)
        proto = []
        for p in today_roster:
            q = dict(p)
            q["sched"] = {dd: B.team_on_sched(p, dd) for dd in dates[i:]}
            proto.append(q)
        opts_det, opts_sto, first = [], [], None
        for j, dd in enumerate(dates[i:]):
            cands, acc, vals, cnt = B.day_opts(proto, dd, d, True)
            opts_det.append((vals, cnt))
            opts_sto.append((vals, [cands[a][3] for a in acc]))
            if j == 0:
                first = (cands, acc)
        cands, acc = first
        m = B.plan_value(opts_det, c, cap)[1] if mode == "current" else plan_value_stoch(opts_sto, c, cap)[1]
        total += sum(cands[acc[k]][2]["pts"] for k in range(m))
        c += m
    return total, c


def main():
    W = B.weeks()
    rows = []
    for (mp, team), days in sorted(W.items()):
        if mp > 19 or mp == 17:
            continue
        a = replay(days, "current")
        b = replay(days, "stoch")
        rows.append(dict(mp=mp, team=team, cur_pts=a[0], cur_starts=a[1], sto_pts=b[0], sto_starts=b[1], ndays=len(days)))
    df = pd.DataFrame(rows)
    df.to_csv(D / "cap_risk_replay.csv", index=False)
    report(df)


def report(df):
    d = df.sto_pts - df.cur_pts
    print(f"{len(df)} team-weeks (2025-26, real rosters and real missed games)")
    print(f"points per team-week: current {df.cur_pts.mean():.1f}, stochastic {df.sto_pts.mean():.1f}; difference {d.mean():+.1f} (se {d.std() / len(d) ** 0.5:.1f})")
    print(f"weeks that differ: {(d.abs() > 0.5).mean():.0%};  better {(d > 0.5).mean():.0%}, worse {(d < -0.5).mean():.0%};  among weeks that differ: {d[d.abs() > 0.5].mean():+.1f}")
    print(f"counted starts: current {df.cur_starts.mean():.1f}, stochastic {df.sto_starts.mean():.1f}")
    for lo, hi in ((0, 38), (38, 44), (44, 99)):
        s = (df.cur_starts >= lo) & (df.cur_starts < hi)
        if s.any():
            print(f"   weeks with {lo}-{hi - 1} counted starts (current): n={s.sum()}, difference {d[s].mean():+.1f}")
    print("biggest gains:", df.assign(d=d).nlargest(3, "d")[["mp", "team", "cur_pts", "sto_pts", "cur_starts", "sto_starts"]].round(0).to_dict("records"))
    print("biggest losses:", df.assign(d=d).nsmallest(3, "d")[["mp", "team", "cur_pts", "sto_pts", "cur_starts", "sto_starts"]].round(0).to_dict("records"))


if __name__ == "__main__":
    if "--report" in ARGV:
        report(pd.read_csv(D / "cap_risk_replay.csv"))
    else:
        main()
