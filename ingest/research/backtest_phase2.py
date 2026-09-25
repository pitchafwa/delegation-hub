"""Backtest of the Phase 2 start/sit features on this league's real 2025-26 season (216 team-weeks, real rosters, real box scores).

A  SCHEDULE-AWARE FORECASTS (what the Schedule tab's "projected points" and the playoff planner are).
   From the start of an anchor week, project each team's points for a later week h weeks ahead using ONLY information available then:
   the team's roster on that day (held fixed), each player's recency-weighted level from games before the anchor, and the NBA schedule for the
   target week, run through the cap-aware lineup solver with the future-game play rates.  Compared with
     blind     the same roster and levels, but no schedule: a flat 3.4-game week for everyone
     persist   the team's own actual points in the anchor week (what people eyeball)
   Scored on how close the forecast is to the team's real points, and on whether it explains week-to-week swings (within-team correlation).
B  PLAYOFF-WEEK RANKING.  The same forecasts for the last three weeks (matchups 20-22) made from week 16 and 19, scored on how well they rank
   the 12 teams against their real playoff-week points (12 teams, one season: noisy).
C  ADD TIMING.  For every team-week the best suggested add (same rules as Phase 1 adds) is valued if made on each day of the week,
   predicted vs realized (real box scores, day-by-day replan).  Policies: always add on day 1, follow the model's timing verdict, oracle best day.
D  Does the number of games in a week change how often a player plays (rest patterns on light vs heavy weeks)?

Run from ingest/:  uv run python research/backtest_phase2.py [A|B|C|D]     (no arg = all)
"""
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import backtest_weekly_planner as B

sys.stdout.reconfigure(encoding="utf-8")
W = B.weeks()                                           # (mp, team) -> {sp: entries}
mp_sps = {}
for (mp, team), days in W.items():
    mp_sps[mp] = sorted(days)
ALL_MP = sorted(mp_sps)
LEN = {mp: len(sps) for mp, sps in mp_sps.items()}
REG = [mp for mp in ALL_MP if mp not in (1, 17) and LEN[mp] == 7 and mp <= 19]
PO = [mp for mp in ALL_MP if mp >= 20 and LEN[mp] == 7]
TEAMS = sorted({t for (_, t) in W})
STARTERS = set(range(12))


def actual_pts(mp, team):
    days = W[(mp, team)]
    return sum(e[4] or 0 for sp in days for e in days[sp] if e[2] in STARTERS and e[3])


def roster_at(mp, team):
    first = mp_sps[mp][0]
    return [e for e in W[(mp, team)][first] if e[2] != 13]


def dates_of(mp):
    return [B.sp_date(sp) for sp in mp_sps[mp]]


def build_roster(entries, d0, dates):
    return [B.mk_player(B.norm(e[1]), d0, dates, B.pos_by_name.get(B.norm(e[1])) or B.DEFPOS.get(e[5])) for e in entries]


_proj_cache = {}


def project(team, anchor_mp, target_mp):
    key = (team, anchor_mp, target_mp)
    if key in _proj_cache:
        return _proj_cache[key]
    d0 = dates_of(anchor_mp)[0]
    dates = dates_of(target_mp)
    roster = build_roster(roster_at(anchor_mp, team), d0, dates)
    if len(roster) < 8:
        _proj_cache[key] = (None, None)
        return _proj_cache[key]
    cap = B.cap_for(len(dates))
    m1 = B.expected_value(roster, dates, cap)
    # blind: best slot-matched 10 by level, each playing a flat 3.4 of 7 days x future play rate
    cands = sorted([(p["level"], p["elig"], p) for p in roster], key=lambda x: -x[0])
    acc = B.greedy_accept(cands)[:10]
    m2 = sum(cands[i][0] * B.p_future(cands[i][0]) * 3.4 for i in acc)
    _proj_cache[key] = (m1, m2)
    return _proj_cache[key]


def hybrid(team, anchor_mp, target_mp):
    """the team's real points in the anchor week, scaled by how much lighter/heavier the target week's schedule is (solver ratio target/anchor)"""
    a = project(team, anchor_mp, anchor_mp)[0]
    t = project(team, anchor_mp, target_mp)[0]
    if not a or not t:
        return None
    return actual_pts(anchor_mp, team) * t / a


def part_a():
    rows = []
    for h in (1, 2, 4, 6):
        for anchor in REG:
            tgt = anchor + h
            if tgt not in REG or anchor < 3:
                continue
            for t in TEAMS:
                m1, m2 = project(t, anchor, tgt)
                if m1 is None:
                    continue
                rows.append(dict(h=h, anchor=anchor, target=tgt, team=t, m1=m1, blind=m2, persist=actual_pts(anchor, t), hybrid=hybrid(t, anchor, tgt), act=actual_pts(tgt, t)))
    df = pd.DataFrame(rows)
    df.to_csv(B.D / "backtest_phase2_A.csv", index=False)
    print("A. SCHEDULE-AWARE FORECASTS (points per team-week)")
    out = []
    for h, g in df.groupby("h"):
        r = dict(h=h, n=len(g))
        for name in ("m1", "blind", "persist", "hybrid"):
            e = g[name] - g.act
            r[name + "_MAE"] = e.abs().mean()
            r[name + "_bias"] = e.mean()
        # within-team swings: does the forecast explain which weeks are big or small for a team?
        for name in ("m1", "blind", "persist", "hybrid"):
            gg = g.assign(f=g[name] - g.groupby("team")[name].transform("mean"), a=g.act - g.groupby("team").act.transform("mean"))
            r[name + "_swing_r"] = np.corrcoef(gg.f, gg.a)[0, 1] if gg.f.std() > 0 else float("nan")
        out.append(r)
    print(pd.DataFrame(out).round(2).T.to_string(header=False))
    return df


def part_b(df=None):
    print("\nB. PLAYOFF WEEKS", PO, "- ranking the 12 teams by their real playoff-week points")
    if not PO:
        print("   no playoff weeks in the data")
        return
    act = {t: sum(actual_pts(mp, t) for mp in PO) for t in TEAMS}
    for anchor in (16, 19):
        res = {}
        for t in TEAMS:
            ps = [project(t, anchor, mp) for mp in PO]
            if any(p[0] is None for p in ps):
                continue
            res[t] = dict(m1=sum(p[0] for p in ps), blind=sum(p[1] for p in ps), persist=actual_pts(anchor, t) * len(PO), hybrid=sum(hybrid(t, anchor, mp) for mp in PO), act=act[t])
        d = pd.DataFrame(res).T
        ks = ("m1", "blind", "persist", "hybrid")
        sp = {k: d[k].corr(d.act, method="spearman") for k in ks}
        mae = {k: (d[k] - d.act).abs().mean() for k in ks}
        print(f"   from week {anchor}: rank correlation with real playoff points  " + "  ".join(f"{k} {sp[k]:.2f}" for k in ks) + " | MAE (3 weeks)  " + "  ".join(f"{k} {mae[k]:.0f}" for k in ks) + f"  (n={len(d)})")


# ---------------- C: add timing
def with_join(pl, join, dates):
    q = dict(pl)
    q["sched"] = {dd: (pl["sched"].get(dd, False) and dd >= join) for dd in dates}
    q["join"] = join
    return q


def realized2(roster, dates, cap):
    """day-by-day replay, real points, oracle availability; a player who joins mid-week cannot play before his join date"""
    c, total = 0, 0.0
    for i, d in enumerate(dates):
        if c >= cap:
            break
        for p in roster:
            p["played"] = ((p["n"], d) in B.played_fp) and (p.get("join") is None or d >= p["join"])
            p["pts"] = B.played_fp.get((p["n"], d), 0.0)
        opts, first = [], None
        for j, dd in enumerate(dates[i:]):
            cands, acc, vals, cnt = B.day_opts(roster, dd, d, True)
            opts.append((vals, cnt))
            if j == 0:
                first = (cands, acc)
        m = B.plan_value(opts, c, cap)[1]
        cands, acc = first
        total += sum(cands[acc[k]][2]["pts"] for k in range(m))
        c += m
    return total


def part_c():
    rows, played = [], set()
    rostered_by_sp = {int(sp): {B.norm(e[1]) for r in recs for e in r["e"]} for sp, recs in B.league.items()}
    for mp in REG:
        if mp < 2:
            continue
        sps, dates = mp_sps[mp], dates_of(mp)
        d0, cap = dates[0], B.cap_for(len(dates))
        taken = rostered_by_sp[sps[0]]
        pool0 = []
        for n, sub in B.series.items():
            if n in taken or n not in B.pos_by_name:
                continue
            past = sub[sub.GAME_DATE.dt.date < d0]
            if len(past) < 8 or past.GAME_DATE.dt.date.max() < d0 - timedelta(days=10):
                continue
            pl = B.mk_player(n, d0, dates, B.pos_by_name[n])
            pl["level"] = B.FA_ANCHOR + 0.6 * (pl["level"] - B.FA_ANCHOR)
            if sum(pl["sched"].values()) == 0:
                continue
            pool0.append(pl)
        pool0.sort(key=lambda p: -(p["level"] * sum(p["sched"].values())))
        pool0 = pool0[:15]
        for t in TEAMS:
            roster = build_roster(roster_at(mp, t), d0, dates)
            if len(roster) < 8:
                continue
            base_e = B.expected_value(roster, dates, cap)
            lv = sorted(roster, key=lambda p: -p["level"])
            prot = {p["n"] for p in lv[:B.KEEPER_PROTECT]} | {p["n"] for p in roster if p["level"] >= B.PROTECT_LEVEL}
            best = None
            for f in pool0:
                drops = ([None] if len(roster) < 15 else []) + [p for p in roster if p["n"] not in prot]
                for dr in drops:
                    wk = B.expected_value([p for p in roster if p is not dr] + [f], dates, cap) - base_e
                    net = wk - (0 if dr is None else max(0.0, dr["level"] - f["level"]) * B.GAMES_PER_WEEK * B.ROS_WEEKS)
                    if best is None or net > best[0]:
                        best = (net, f, dr)
            if not best or best[0] < B.MIN_NET:
                continue
            net, f, dr = best
            base_r = [p for p in roster if p is not dr]
            real_base = realized2([dict(p) for p in base_r], dates, cap)
            for k, dk in enumerate(dates):
                f2 = with_join(f, dk, dates)
                pred = B.expected_value(base_r + [f2], dates, cap) - B.expected_value(base_r, dates, cap)
                real = realized2([dict(p) for p in base_r] + [dict(f2)], dates, cap) - real_base
                rows.append(dict(mp=mp, team=t, day=k, pred=pred, real=real))
        print("  week", mp, "done", len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(B.D / "backtest_phase2_C.csv", index=False)
    return summarize_c(df)


def summarize_c(df):
    print("\nC. ADD TIMING (best suggested add per team-week; gain vs not adding)")
    print(f"   {df.groupby(['mp', 'team']).ngroups} team-weeks with a suggested add; {len(df)} (add, day) values")
    print("   by day of the week (mean predicted vs realized gain):")
    print(df.groupby("day").agg(predicted=("pred", "mean"), realized=("real", "mean"), n=("real", "size")).round(1).T.to_string())
    print(f"   predicted vs realized across (add, day) pairs: correlation {df.pred.corr(df.real):.2f}")
    pol = {"always day 1": [], "model timing": [], "wait 1 day": [], "oracle": []}
    for _, g in df.groupby(["mp", "team"]):
        g = g.sort_values("day")
        p, r = g.pred.to_numpy(), g.real.to_numpy()
        pol["always day 1"].append(r[0])
        pol["wait 1 day"].append(r[min(1, len(r) - 1)])
        k = int(np.argmax(p))
        pol["model timing"].append(r[k] if p[k] - p[0] >= 5 else r[0])
        pol["oracle"].append(r.max())
    for k, v in pol.items():
        print(f"   {k:14s} realized gain {np.mean(v):+6.1f} pts/wk")
    wait = df.groupby(["mp", "team"]).apply(lambda g: (g.sort_values("day").pred.iloc[0] - g.sort_values("day").pred.iloc[1]) if len(g) > 1 else 0.0)
    real_wait = df.groupby(["mp", "team"]).apply(lambda g: (g.sort_values("day").real.iloc[0] - g.sort_values("day").real.iloc[1]) if len(g) > 1 else 0.0)
    print(f"   cost of waiting one day: predicted {wait.mean():.1f} vs realized {real_wait.mean():.1f}; correlation {wait.corr(real_wait):.2f}")


def part_d():
    g = B.g
    g = g.assign(d=g.GAME_DATE.dt.date)
    avg = g.groupby("n").fp.mean()
    rot = set(avg[avg >= 24].index)
    rows = []
    d_first = B.DATE0.date()
    tot = {t: sorted(ds) for t, ds in B.team_days.items()}
    played = {}
    for r in g[["n", "d", "TEAM_ABBREVIATION"]].itertuples():
        played.setdefault(r.n, set()).add(r.d)
    for n in rot:
        s = g[g.n == n]
        if len(s) < 25:
            continue
        for tm, sub in s.groupby("TEAM_ABBREVIATION"):
            first, last = sub.d.min(), sub.d.max()
            days = [dd for dd in tot.get(tm, []) if first <= dd <= last]
            wk = {}
            for dd in days:
                w = (dd - d_first).days // 7
                wk.setdefault(w, []).append(dd)
            for w, dl in wk.items():
                rows.append(dict(sched=len(dl), played=sum(dd in played[n] for dd in dl)))
    df = pd.DataFrame(rows)
    df = df[df.sched.between(2, 5)]
    print("\nD. Play rate of rotation players (avg 24+ pts) by NBA games in the week")
    g2 = df.groupby("sched").agg(player_weeks=("played", "size"), played=("played", "sum"), sched_games=("sched", "sum"))
    g2["play_rate"] = g2.played / g2.sched_games
    print(g2[["player_weeks", "play_rate"]].round(3).T.to_string())


if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "ABCD"
    print(f"regular weeks {REG}, playoff weeks {PO}, teams {len(TEAMS)}")
    if "A" in which:
        part_a()
    if "B" in which:
        part_b()
    if "D" in which:
        part_d()
    if "C" in which:
        part_c()
