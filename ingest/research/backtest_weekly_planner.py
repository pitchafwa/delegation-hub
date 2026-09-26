"""Backtest the weekly planner on the league's real 2025-26 season.

PART 1 - LINEUP MANAGEMENT.  For every team-week (regular season) replay the team's own roster day by day, exactly as the team had it, and let
the planner set the lineup each morning:
  * decisions use only information available that morning: player levels = a recency-weighted average of games BEFORE that day (half-life 8
    games; prior-season average when fewer than 5 games), who actually plays that day (proxy for the injury report), the known NBA schedule
    for the rest of the week (94% chance a scheduled player suits up), and the team's own games-so-far under the 40-game cap
  * scoring uses what the players ACTUALLY scored (ESPN's applied points), so hindsight only enters through who is available that day
  * three policies are compared with the team's ACTUAL result:  greedy (start everyone who can play; ignores the cap),
    planner (cap-aware daily plan, the dynamic program the site uses),  and the team's own real lineups.
  * then each team-week's planner points are played against the opponent's ACTUAL points to see how many results flip.
Position eligibility is approximated from ESPN's current eligibility list (default position when missing), so multi-position flexibility is
slightly understated for the planner.

PART 2 - ADDS (see backtest_adds section below, added after part 1 is checked).
Run from ingest/:  uv run python research/backtest_weekly_planner.py
"""
import json
import re
import sys
import unicodedata
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
D = R / "data"
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
P_PLAY = 0.94             # chance an ACTIVE player plays TODAY (status known that morning)
def p_future(level):      # measured (availability_by_level.py): chance a currently-healthy player suits up in a later scheduled game
    return 0.61 if level < 15 else (0.81 if level < 20 else 0.85)
CAP7 = 40.0
HL = 8.0


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


POSMAP = {"PG": {"PG", "G", "UT"}, "SG": {"SG", "G", "UT"}, "SF": {"SF", "F", "UT"}, "PF": {"PF", "F", "UT"}, "C": {"C", "UT"}, "G": {"PG", "SG", "G", "UT"}, "F": {"SF", "PF", "F", "UT"}}
DEFPOS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}


def elig_from(posstr, defpos):
    toks = [t for t in re.split(r"[/,\s]+", str(posstr or "")) if t in POSMAP]
    if not toks:
        toks = [DEFPOS.get(defpos, "SF")]
    out = set()
    for t in toks:
        out |= POSMAP[t]
    return frozenset(out)


# ---------------- data
league = json.load(open(D / "league_days_2026.json"))
g = pd.read_csv(D / "game_logs" / "nba_api_2025-26.csv")
g24 = pd.read_csv(D / "game_logs" / "nba_api_2024-25.csv")
for df in (g, g24):
    df["fp"] = df.PTS + 1.5 * df.REB + 2 * df.AST + 3 * df.STL + 3 * df.BLK + df.FG3M + 2 * df.FTM - df.FTA - df.TOV
    df["GAME_DATE"] = pd.to_datetime(df.GAME_DATE)
    df["n"] = df.PLAYER_NAME.map(norm)
DATE0 = g.GAME_DATE.min()
sp_date = lambda sp: (DATE0 + timedelta(days=int(sp) - 1)).date()
prior = g24.groupby("n").fp.mean().to_dict()
series = {n: sub.sort_values("GAME_DATE")[["GAME_DATE", "fp"]].reset_index(drop=True) for n, sub in g.groupby("n")}
team_days = {}     # NBA team -> set of dates it plays
for (t, d), _ in g.groupby(["TEAM_ABBREVIATION", "GAME_DATE"]):
    team_days.setdefault(t, set()).add(d.date())
last_team = {}
for n, sub in g.sort_values("GAME_DATE").groupby("n"):
    last_team[n] = list(zip(sub.GAME_DATE.dt.date, sub.TEAM_ABBREVIATION))
pos_csv = pd.read_csv(D / "espn_positions.csv")
pos_by_name = {norm(r.espn_name): r.espn_position for r in pos_csv.itertuples()}

_lvl_cache = {}


def level(n, d):
    """recency-weighted fantasy points per game using only games before date d"""
    key = (n, d)
    if key in _lvl_cache:
        return _lvl_cache[key]
    s = series.get(n)
    v = None
    if s is not None:
        past = s[s.GAME_DATE.dt.date < d]
        if len(past) >= 5:
            w = 0.5 ** (np.arange(len(past))[::-1] / HL)
            v = float((past.fp.to_numpy() * w).sum() / w.sum())
        elif len(past) > 0 and n in prior:
            v = 0.5 * float(past.fp.mean()) + 0.5 * prior[n]
    if v is None:
        v = prior.get(n, 18.0)
    _lvl_cache[key] = v
    return v


def team_on(n, d):
    tl = last_team.get(n)
    if not tl:
        return None
    t = None
    for dd, tm in tl:
        if dd <= d:
            t = tm
        else:
            break
    return t or tl[0][1]


# ---------------- planner core (same logic as build_week_plan.py)
def greedy_accept(cands):
    """cands: list of (ef, elig, key). Returns indices accepted in order (matroid greedy by expected points)."""
    match = {}

    def go(ci, seen):
        for si, s in enumerate(SLOTS):
            if s in cands[ci][1] and si not in seen:
                seen.add(si)
                if si not in match or go(match[si], seen):
                    match[si] = ci
                    return True
        return False

    acc = []
    for ci in range(len(cands)):
        if go(ci, set()):
            acc.append(ci)
    return acc


def plan_value(opts, c0, cap):
    """opts: per remaining day (vals prefix, counts prefix). Returns best expected total and the first day's chosen m."""
    memo = {}

    def best(di, c):
        if di == len(opts):
            return 0.0, 0
        if c >= cap:
            return 0.0, 0
        key = (di, c)
        if key in memo:
            return memo[key]
        vals, cnt = opts[di]
        top = (-1e9, 0)
        for m in range(len(vals)):
            v = vals[m] + best(di + 1, min(int(round(c + cnt[m])), int(cap) + 12))[0]
            if v > top[0] + 1e-9:
                top = (v, m)
        memo[key] = top
        return top

    return best(0, int(round(c0)))


def day_opts(players, d, today, oracle):
    """players: list of dict(key, elig, level, played, sched). For `today` availability is known (played); later days use schedule x P_PLAY."""
    cands = []
    for p in players:
        if d == today:
            if p["played"]:
                cands.append((p["level"], p["elig"], p, 1.0))
        elif p["sched"].get(d, False):
            pf = p_future(p["level"])
            cands.append((pf * p["level"], p["elig"], p, pf))
    cands.sort(key=lambda x: -x[0])
    acc = greedy_accept(cands)
    vals, cnt = [0.0], [0.0]
    for ci in acc:
        vals.append(vals[-1] + cands[ci][0])
        cnt.append(cnt[-1] + cands[ci][3])
    return cands, acc, vals, cnt


# ---------------- team-week replay
def weeks():
    by = {}
    for sp, recs in league.items():
        for r in recs:
            by.setdefault((r["mp"], r["team"]), {})[int(sp)] = r["e"]
    return by


def cap_for(n_days):
    return CAP7 * n_days / 7.0


def replay(team_days_entries, mode):
    """team_days_entries: {sp: entries}. mode 'planner' | 'greedy'. Returns (points, starts, per-day starts)"""
    sps = sorted(team_days_entries)
    dates = [sp_date(sp) for sp in sps]
    cap = cap_for(len(sps))
    c, total = 0, 0.0
    info = {}
    for sp, d in zip(sps, dates):
        pls = []
        for e in team_days_entries[sp]:
            pid, name, slot, gp, pts, dpos, inj = e
            if slot == 13:       # IR: cannot start
                continue
            n = norm(name)
            tm = team_on(n, d)
            pls.append({"key": pid, "n": n, "elig": elig_from(pos_by_name.get(n), dpos), "level": None, "played": bool(gp) and pts is not None, "pts": pts or 0.0,
                        "team": tm})
        info[d] = pls
    for i, d in enumerate(dates):
        if c >= cap:
            break
        today_roster = info[d]
        for p in today_roster:
            p["level"] = level(p["n"], d)
        # schedule knowledge for later days: the team's NBA schedule; roster held fixed at today's
        proto = []
        for p in today_roster:
            q = dict(p)
            q["sched"] = {dd: (team_on_sched(p, dd)) for dd in dates[i:]}
            proto.append(q)
        opts, first = [], None
        for j, dd in enumerate(dates[i:]):
            cands, acc, vals, cnt = day_opts(proto, dd, d, True)
            opts.append((vals, cnt))
            if j == 0:
                first = (cands, acc, vals, cnt)
        cands, acc, vals, cnt = first
        m = len(vals) - 1 if mode == "greedy" else plan_value(opts, c, cap)[1]
        chosen = [cands[acc[k]][2] for k in range(m)]
        total += sum(p["pts"] for p in chosen)
        c += m
    return total, c


def team_on_sched(p, d):
    tm = p["team"]
    if tm is None:
        return False
    return d in team_days.get(tm, set())


# ---------------- PART 2: adds
KEEPER_PROTECT, PROTECT_LEVEL, GAMES_PER_WEEK, ROS_WEEKS, MIN_NET = 6, 35, 3.3, 6, 10
import os
FA_ANCHOR, FA_SHRINK = 22.0, float(os.environ.get("FA_SHRINK", "1.0"))
played_fp = {(n, d.date()): v for n, sub in series.items() for d, v in zip(sub.GAME_DATE, sub.fp)}


def mk_player(n, d0, dates, pos):
    tm = team_on(n, d0)
    return {"key": n, "n": n, "elig": elig_from(pos, None), "level": level(n, d0), "team": tm, "sched": {dd: (dd in team_days.get(tm, set())) for dd in dates}}


def expected_value(roster, dates, cap):
    opts = [day_opts(roster, dd, None, False)[2:] for dd in dates]
    return plan_value(opts, 0, cap)[0]


def realized_value(roster, dates, cap):
    """sequential replay with oracle availability (played_fp) and real points; same policy as part 1's planner"""
    c, total = 0, 0.0
    for i, d in enumerate(dates):
        if c >= cap:
            break
        for p in roster:
            p["played"] = (p["n"], d) in played_fp
            p["pts"] = played_fp.get((p["n"], d), 0.0)
        opts, first = [], None
        for j, dd in enumerate(dates[i:]):
            cands, acc, vals, cnt = day_opts(roster, dd, d, True)
            opts.append((vals, cnt))
            if j == 0:
                first = (cands, acc)
        m = plan_value(opts, c, cap)[1]
        cands, acc = first
        total += sum(cands[acc[k]][2]["pts"] for k in range(m))
        c += m
    return total


def backtest_adds():
    W = weeks()
    rostered_by_sp = {}
    for sp, recs in league.items():
        rostered_by_sp[int(sp)] = {norm(e[1]) for r in recs for e in r["e"]}
    out = []
    for (mp, team), days in sorted(W.items()):
        if mp > 19 or mp == 17 or mp == 1:
            continue
        if os.environ.get("ONLY_TEAM") and team != int(os.environ["ONLY_TEAM"]):
            continue
        sps = sorted(days)
        dates = [sp_date(sp) for sp in sps]
        d0, cap = dates[0], cap_for(len(sps))
        r0 = [e for e in days[sps[0]] if e[2] != 13]
        roster = [mk_player(norm(e[1]), d0, dates, pos_by_name.get(norm(e[1])) or DEFPOS.get(e[5])) for e in r0]
        if len(roster) < 8:
            continue
        taken = rostered_by_sp[sps[0]]
        pool = []
        for n, sub in series.items():
            if n in taken or n not in pos_by_name:
                continue
            past = sub[sub.GAME_DATE.dt.date < d0]
            if len(past) < 8 or past.GAME_DATE.dt.date.max() < d0 - timedelta(days=10):
                continue
            pl = mk_player(n, d0, dates, pos_by_name[n])
            pl["level"] = FA_ANCHOR + FA_SHRINK * (pl["level"] - FA_ANCHOR)   # regression to the mean for the free agents we pick because they look good
            if sum(pl["sched"].values()) == 0:
                continue
            pool.append(pl)
        pool.sort(key=lambda p: -(p["level"] * sum(p["sched"].values())))
        pool = pool[:40]
        base_e = expected_value(roster, dates, cap)
        lvl_rank = sorted(roster, key=lambda p: -p["level"])
        prot = {p["n"] for p in lvl_rank[:KEEPER_PROTECT]} | {p["n"] for p in roster if p["level"] >= PROTECT_LEVEL}
        cur, r2, moves, used = base_e, list(roster), [], set()
        for step in range(4):
            best = None
            for f in pool:
                if f["n"] in used:
                    continue
                drops = ([None] if len(r2) < 15 else []) + [p for p in r2 if p["n"] not in prot]
                for dr in drops:
                    wk = expected_value([p for p in r2 if p is not dr] + [f], dates, cap) - cur
                    net = wk - (0 if dr is None else max(0.0, dr["level"] - f["level"]) * GAMES_PER_WEEK * ROS_WEEKS)
                    if best is None or net > best[0]:
                        best = (net, wk, f, dr)
            if not best or best[0] < MIN_NET:
                break
            net, wk, f, dr = best
            r2 = [p for p in r2 if p is not dr] + [f]
            cur += wk
            used.add(f["n"])
            moves.append((f["n"], None if dr is None else dr["n"], round(wk, 1)))
        pred = cur - base_e
        real_base = realized_value([dict(p) for p in roster], dates, cap)
        real_new = realized_value([dict(p) for p in r2], dates, cap)
        out.append(dict(mp=mp, team=team, moves=len(moves), predicted=round(pred, 1), realized=round(real_new - real_base, 1), base=round(real_base, 1)))
    df = pd.DataFrame(out)
    df.to_csv(D / ("backtest_adds_2026_team%s.csv" % os.environ["ONLY_TEAM"] if os.environ.get("ONLY_TEAM") else "backtest_adds_2026.csv"), index=False)
    print(f"{len(df)} team-weeks; weeks with a suggested move: {(df.moves > 0).mean():.0%}; mean moves when suggested {df[df.moves > 0].moves.mean():.1f}")
    s = df[df.moves > 0]
    print(f"when moves were suggested: predicted gain {s.predicted.mean():+.1f} pts/wk, REALIZED gain {s.realized.mean():+.1f} pts/wk (median {s.realized.median():+.1f}); positive in {(s.realized > 0).mean():.0%} of those weeks")
    print(f"averaged over all team-weeks: {df.realized.mean():+.1f} pts/wk")
    print("by number of moves:"); print(s.groupby("moves").agg(n=("realized", "size"), predicted=("predicted", "mean"), realized=("realized", "mean")).round(1).to_string())
    print("by team:"); print(df.groupby("team").agg(weeks_with_moves=("moves", lambda x: (x > 0).sum()), realized=("realized", "mean")).round(1).T.to_string())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "adds":
    backtest_adds()
elif __name__ == "__main__":
    W = weeks()
    rows = []
    for (mp, team), days in sorted(W.items()):
        if mp > 19 or mp == 17:          # regular season; period 17 is the 14-day matchup (cap 80 never binds), skipped for speed
            continue
        act_pts = sum(e[4] or 0 for sp in days for e in days[sp] if e[2] in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} and e[3])
        act_starts = sum(1 for sp in days for e in days[sp] if e[2] in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} and e[3])
        pp, ps = replay(days, "planner")
        gp_, gs = replay(days, "greedy")
        rows.append(dict(mp=mp, team=team, actual=act_pts, actual_starts=act_starts, planner=pp, planner_starts=ps, greedy=gp_, greedy_starts=gs, ndays=len(days)))
    df = pd.DataFrame(rows)
    df.to_csv(D / "backtest_lineups_2026.csv", index=False)
    print(f"{len(df)} team-weeks replayed")
    print("\nmean points per team-week:")
    print(df[["actual", "greedy", "planner"]].mean().round(1).to_string())
    print(f"planner vs actual: {(df.planner - df.actual).mean():+.1f} pts/wk  (median {(df.planner - df.actual).median():+.1f}); planner beats actual in {(df.planner > df.actual).mean():.0%} of team-weeks")
    print(f"planner vs greedy: {(df.planner - df.greedy).mean():+.1f} pts/wk; weeks the cap logic changed the result: {((df.planner - df.greedy).abs() > 0.5).mean():.0%}")
    print("\nby team (mean pts/wk): actual -> planner (games started)")
    bt = df.groupby("team").agg(actual=("actual", "mean"), planner=("planner", "mean"), a_st=("actual_starts", "mean"), p_st=("planner_starts", "mean")).round(1)
    bt["gain"] = (bt.planner - bt.actual).round(1)
    print(bt.sort_values("gain", ascending=False).to_string())
    json.dump(rows, open(D / "backtest_lineups_2026.json", "w"))
