"""The replay pieces of title_odds_study.py for other studies: who is out at a checkpoint and the measured return curve (fit on 2021-22..2024-25 reports),
roster(team, anchor) with each player's availability factor (rule "curve" = what the live forecast does since 2026-11-07)."""
from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import backtest_phase2 as P
import backtest_weekly_planner as B

R = Path(__file__).resolve().parent

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


def roster(team, anchor, rule="curve"):
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


