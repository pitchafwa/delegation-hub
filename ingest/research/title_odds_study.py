"""Title odds study (2026-11-07, Tommy: "I'm not sure I buy a 51% chance to win the title").

Replays this league's 2025-26 season (backtest_phase2's data: real rosters each morning, real box scores, real matchups) and rebuilds, at the start of every
regular-season week, the forecast the playoff simulation uses: the cap-aware solver on that morning's roster for every remaining week, times the in-season
"beat-the-model" factor (league factor x team factor shrunk with K = 5, as build_schedule_plan.py does).

A. INJURED PLAYERS.  Three ways to count a player who is out at the start of the week (on IR, or inside a reported absence and missing his team's games):
     none    not counted at all
     flat    the live rule: nothing for the next 2 weeks, then 60% of the normal play rate
     curve   the measured return curve (official reports 2021-22..2024-25 only, so 2025-26 is out of sample): P(back within k more team games | out s games
             so far, body-part group, tier), times 0.9 for the first weeks back
   Scored on how close each team's forecast for the rest of the season is to what it really scored.
B. HOW MUCH A TEAM DRIFTS.  The simulation draws one season-long shock per team (SD tau) plus weekly noise (SD sigma_e). From the replay: tau^2 = the average
   product of a team's forecast misses in two different later weeks (what the misses share), sigma_e^2 = the average squared miss minus tau^2. By stage of season.
C. OUTCOMES.  The simulation run at every week's start with tau 60-250: scored on who really made the playoffs, got a bye, and won the title (Brier, log loss).
   One season, one champion: C is a sanity check on B, not a fit.
Run from ingest/:  python research/title_odds_study.py
"""
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import backtest_phase2 as P
import backtest_weekly_planner as B
from playoff_sim import simulate

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
REG_LAST = 19
ALL_W = [w for w in P.ALL_MP if w >= 2]
ANCHORS = [w for w in P.REG if w <= 16]

# ---------------- return curves from the official reports, 2021-22..2024-25 (out of sample for 2025-26)
ep = pd.read_pickle(R / "data" / "injury_episodes.pkl")
rows_ = pd.read_pickle(R / "data" / "injury_rows_05pm.pkl")
season_last = rows_.groupby("season").gd.max().to_dict()
ep = ep[ep.n_out >= 1].copy()
ep["censored"] = ep.apply(lambda r: (season_last[r.season] - r.end).days <= 10, axis=1)
ep["tier2"] = ep.tier.replace({"recovery": "major", "management": "moderate", "other": "moderate"})
fit = ep[ep.season.isin(["2021-22", "2022-23", "2023-24", "2024-25"])]
SB = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 10000)]
KS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 70]


def sbin(s):
    return next((i for i, (a, b) in enumerate(SB) if a <= s <= b), len(SB) - 1)


_curve_cache = {}


def curve_for(group, tier, s):
    si = sbin(max(1, s))
    for key in ((group, tier), ("all", tier), ("all", "all")):
        if (key, si) in _curve_cache:
            return _curve_cache[(key, si)]
        d = fit
        if key[0] != "all":
            d = d[d.group == key[0]]
        if key[1] != "all":
            d = d[d.tier2 == key[1]]
        a, b = SB[si]
        d = d[d.n_out >= a]
        if len(d) < 40:
            continue
        # pooled over the s values in the bin: games still to miss = n_out - s, using each episode's own s at the bin's lower edge (simple and close enough)
        rem = d.n_out - a
        cur = [float(((rem <= k) & ~d.censored).mean()) for k in KS]
        _curve_cache[(key, si)] = cur
        return cur
    return _curve_cache.get((("all", "all"), si))


def p_back(cur, k):
    if k <= 0:
        return 0.0
    pk, pp = 0, 0.0
    for kk, p in zip(KS, cur):
        if k <= kk:
            return pp + (p - pp) * (k - pk) / (kk - pk)
        pk, pp = kk, p
    return cur[-1]


# ---------------- who is out at an anchor date, and for how long so far
eps25 = ep[ep.season == "2025-26"]
ep_by = {k: g for k, g in eps25.groupby("player_key")}


def team_games(tm, a, b):
    """team games on dates a <= d < b"""
    return sum(1 for d in B.team_days.get(tm, ()) if a <= d < b)


def injury_state(e, d0):
    """None if healthy; else (group, tier, games out so far) for a player out at d0 (IR slot, or inside a reported absence and missed his team's last 2 games)"""
    n = B.norm(e[1])
    tm = B.team_on(n, d0)
    g = ep_by.get(n.replace(" ", ""))
    cur = None
    if g is not None:
        c_ = g[(g.start.dt.date <= d0) & (g.end.dt.date >= d0 - timedelta(days=1))]
        if len(c_):
            cur = c_.iloc[-1]
    last_games = sorted(d for d in B.team_days.get(tm, ()) if d < d0)[-2:] if tm else []
    missed2 = len(last_games) == 2 and all((n, d) not in B.played_fp for d in last_games)
    if e[2] != 13 and not (cur is not None and missed2):
        return None
    # games out so far: team games since his last game played
    past = sorted((d for d in B.team_days.get(tm, ()) if d < d0), reverse=True) if tm else []
    s = 0
    for d in past:
        if (n, d) in B.played_fp:
            break
        s += 1
    grp, tier = (cur.group, cur.tier2) if cur is not None else ("all", "all")
    return grp, tier, max(1, s)


# ---------------- forecasts
def avail_fn(rule, state, d0, anchor, tm):
    if state is None:
        return lambda dd, w: 1.0
    if rule == "none":
        return lambda dd, w: 0.0
    if rule == "flat":
        return lambda dd, w: 0.0 if w < anchor + 2 else 0.6
    grp, tier, s = state
    cur = curve_for(grp, tier, s)
    return lambda dd, w: p_back(cur, team_games(tm, d0, dd + timedelta(days=1)) - 0.5) * 0.9


_roster_cache = {}


def roster(team, anchor, rule):
    key = (team, anchor, rule)
    if key in _roster_cache:
        return _roster_cache[key]
    d0 = P.dates_of(anchor)[0]
    ents = P.W[(anchor, team)][P.mp_sps[anchor][0]]
    out = []
    for e in ents:
        n = B.norm(e[1])
        tm = B.team_on(n, d0)
        st = injury_state(e, d0)
        if st is None and e[2] == 13:
            st = ("all", "all", 1)
        pl = {"key": e[0], "n": n, "elig": B.elig_from(B.pos_by_name.get(n), e[5]), "level": B.level(n, d0), "team": tm, "inj": st,
              "avail": avail_fn(rule, st, d0, anchor, tm)}
        out.append(pl)
    _roster_cache[key] = out
    return out


def solve(rost, w):
    dates = P.dates_of(w)
    cap = B.cap_for(len(dates))
    opts = []
    for dd in dates:
        cands = []
        for p in rost:
            if not p["level"] or not p["team"] or dd not in B.team_days.get(p["team"], ()):
                continue
            a = p["avail"](dd, w)
            if a <= 0:
                continue
            pf = B.p_future(p["level"]) * a
            cands.append((pf * p["level"], p["elig"], p, pf))
        cands.sort(key=lambda x: -x[0])
        acc = B.greedy_accept(cands)
        vals, cnt = [0.0], [0.0]
        for ci in acc:
            vals.append(vals[-1] + cands[ci][0])
            cnt.append(cnt[-1] + cands[ci][3])
        opts.append((vals, cnt))
    return B.plan_value(opts, 0, cap)[0]


act = {(t, w): P.actual_pts(w, t) for t in P.TEAMS for w in ALL_W if (w, t) in P.W}
RULES = ["none", "flat", "curve"]
raw = {}
for rule in RULES:
    for c in ANCHORS:
        for t in P.TEAMS:
            if (c, t) not in P.W:
                continue
            ro = roster(t, c, rule)
            for w in ALL_W:
                if w >= c:
                    raw[(rule, t, c, w)] = solve(ro, w)
    print(f"forecasts built: {rule}", flush=True)

# in-season factor, as live: start-of-week forecasts of past full 7-day weeks
FULL = [w for w in P.REG]
fc_rows = []
fcast = {}
for rule in RULES:
    for c in ANCHORS:
        past = [w for w in FULL if w < c]
        pool = [(act[(t, w)], raw[(rule, t, w, w)]) for t in P.TEAMS for w in past if (rule, t, w, w) in raw and raw[(rule, t, w, w)] > 0]
        lf = sum(a for a, _ in pool) / sum(f for _, f in pool) if pool else None
        for t in P.TEAMS:
            mine = [w for w in past if (rule, t, w, w) in raw and raw[(rule, t, w, w)] > 0]
            if lf and mine:
                tf = sum(act[(t, w)] for w in mine) / sum(raw[(rule, t, w, w)] for w in mine)
                mult = lf * (tf / lf) ** (len(mine) / (len(mine) + 5))
            else:
                mult = 0.95
            for w in ALL_W:
                if (rule, t, c, w) in raw:
                    f = mult * raw[(rule, t, c, w)]
                    fcast[(rule, t, c, w)] = f
                    if w > c and (t, w) in act:
                        L = P.LEN[w]
                        fc_rows.append(dict(rule=rule, team=t, c=c, w=w, L=L, f=f, a=act[(t, w)], r7=(act[(t, w)] - f) * 7 / L))
df = pd.DataFrame(fc_rows)
reg = df[(df.w <= REG_LAST) & (df.L == 7)]

print("\nA. INJURED PLAYERS: miss in a team's average score per week over the rest of the regular season (points per 7 days)")
for rule in RULES:
    g = reg[reg.rule == rule].groupby(["team", "c"]).agg(m=("r7", "mean"), n=("r7", "size")).reset_index()
    g = g[g.n >= 3]
    wk = reg[reg.rule == rule]
    print(f"   {rule:6s} rest-of-season miss: mean abs {g.m.abs().mean():6.1f}  bias {g.m.mean():+6.1f}   single-week miss: mean abs {wk.r7.abs().mean():6.1f}")
    for lo, hi, lab in ((2, 6, "weeks 2-6"), (7, 11, "weeks 7-11"), (12, 16, "weeks 12-16")):
        gg = g[(g.c >= lo) & (g.c <= hi)]
        print(f"          from {lab}: rest-of-season mean abs miss {gg.m.abs().mean():6.1f} (n {len(gg)})")

print("\nB. SEASON-LONG DRIFT (tau) and weekly noise (sigma_e), per 7 days; league-wide miss each week removed (the simulation has no shared shock)")
best_rule = None
for rule in RULES:
    d = reg[reg.rule == rule].copy()
    d["r"] = d.r7 - d.groupby(["c", "w"]).r7.transform("mean")
    for lo, hi, lab in ((2, 6, "from weeks 2-6"), (7, 16, "from weeks 7-16"), (2, 16, "all")):
        prods, sq = [], []
        for (t, c), g in d[(d.c >= lo) & (d.c <= hi)].groupby(["team", "c"]):
            r = g.r.values
            if len(r) < 2:
                continue
            s = r.sum()
            prods.append((s * s - (r * r).sum()) / (len(r) * (len(r) - 1)))
            sq.append((r * r).mean())
        t2, v = float(np.mean(prods)), float(np.mean(sq))
        print(f"   {rule:6s} {lab:16s} tau {math.sqrt(max(t2, 0)):6.1f}   sigma_e {math.sqrt(max(v - t2, 0)):6.1f}   (pairs from {len(prods)} team-checkpoints)")

# ---------------- C. outcomes
import build_playoff_odds as O
teams_es, games = O.pull(2026)
from espn_api.basketball import League
import config
_lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
_raw = _lg.espn_request.league_get(params={"view": ["mTeam"]})
seed = {t["id"]: t.get("playoffSeed") for t in _raw["teams"]}
final = {t["id"]: t.get("rankCalculatedFinal") for t in _raw["teams"]}
ids = sorted(teams_es)
assert set(ids) == set(P.TEAMS), (ids, P.TEAMS)
made_y = np.array([seed[t] <= 6 for t in ids], float)
bye_y = np.array([seed[t] <= 2 for t in ids], float)
title_y = np.array([final[t] == 1 for t in ids], float)
champ = next(t for t in ids if final[t] == 1)
print(f"\nC. OUTCOMES (2025-26: champion {teams_es[champ]}; playoffs {', '.join(teams_es[t] for t in ids if seed[t] <= 6)})")
lens = dict(P.LEN)


def ll(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


for rule in ("flat", "curve"):
    print(f"   injured rule: {rule}")
    res = {}
    for tau in (60, 100, 150, 200, 250):
        sc = {"made": [], "bye": [], "title": [], "champ_p": [], "fav": []}
        for c in ANCHORS:
            wins, pf = O.standings(games, upto=c - 1)
            for t in ids:
                wins.setdefault(t, 0)
                pf.setdefault(t, 0.0)
            pairs = [(g["week"], g["a"], g["b"]) for g in games if g["week"] >= c]
            mu = {(t, w): fcast[(rule, t, c, w)] for t in ids for w in ALL_W if (rule, t, c, w) in fcast}
            r = simulate(ids, wins, pf, pairs, mu, lens, REG_LAST, tau=float(tau), sigma_e=175.0, bias=0.0, n=6000, seed=3)
            sc["made"].append(ll(r["made"], made_y))
            sc["bye"].append(ll(r["bye"], bye_y))
            sc["title"].append(ll(r["title"], title_y))
            sc["champ_p"].append(float(r["title"][ids.index(champ)]))
            sc["fav"].append(float(r["title"].max()))
        res[tau] = sc
        print(f"     tau {tau:3d}: log loss made {np.mean(sc['made']):.3f}  bye {np.mean(sc['bye']):.3f}  title {np.mean(sc['title']):.4f}   "
              f"champion's avg title odds {np.mean(sc['champ_p']):.0%}   favourite's avg {np.mean(sc['fav']):.0%}  (max {np.max(sc['fav']):.0%})")
    if rule == "curve":
        print("     champion's title odds by checkpoint (tau 100 / 150 / 200):", " ".join(f"w{c}:{res[100]['champ_p'][i]:.0%}/{res[150]['champ_p'][i]:.0%}/{res[200]['champ_p'][i]:.0%}" for i, c in enumerate(ANCHORS)))

json.dump({"rows": len(df)}, open(R / "data" / "title_odds_study.json", "w"))
