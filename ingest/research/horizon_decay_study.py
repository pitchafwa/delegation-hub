"""Later-week forecasts run high (forecast_horizon_study.py, 2026-11-07): a player healthy today plays fewer of his games, and scores a little less, the further out
the week. Does a decay by weeks ahead and level fix the forecast, and does it tell teams apart better?

1. FIT on NBA box scores 2010-11..2023-24 (game_logs_unified.csv, sit-outs included), nothing from 2025-26: at a weekly checkpoint, every player with 8+ games
   who played one of his team's last 2 games, recency-weighted level L (half-life 8 games). For each of his team's later games up to Apr 1 (the fantasy season):
   real fantasy points vs what the solver assumes (p_future(L) x L), by weeks ahead h and level band. Decay factor f(h, band) = ratio(h) / ratio(0)
   (the in-season factor already calibrates same-week forecasts). Written to horizon_decay.json.
2. TEST on this league's 2025-26 replay, same forecasts as title_odds_study.py (curve rule for injured players = live) with:
     live      no decay (today)
     full      every player's later games x f(h, band)
     relative  only the level part: f(h, band) / f(h, all levels), so stars lose relative to role players but the league total is unchanged
   Scored on the rest-of-season miss (raw and with the league-wide miss removed), the drift tau, and playoff / bye / title log loss.
Run from ingest/:  python research/horizon_decay_study.py
"""
import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
BANDS = [(20, 30), (30, 40), (40, 50), (50, 999)]
HMAX = 16


def band_of(L):
    return next((i for i, (a, b) in enumerate(BANDS) if a <= L < b), None)


def p_future(level):
    return 0.61 if level < 15 else (0.81 if level < 20 else 0.85)


# ---------------- 1. fit
g = pd.read_csv(R / "data" / "game_logs_unified.csv")
g = g[g.SEASON <= "2023-24"].copy()
g["GAME_DATE"] = pd.to_datetime(g.GAME_DATE)
g["fp"] = g.PTS + 1.5 * g.REB + 2 * g.AST + 3 * g.STL + 3 * g.BLK + g.FG3M + 2 * g.FTM - g.FTA - g.TOV
g["pl"] = g.MIN > 0
team_dates = {k: np.sort(v.GAME_DATE.unique()) for k, v in g.groupby(["SEASON", "TEAM"])}
S_real = np.zeros((HMAX + 1, len(BANDS)))
S_exp = np.zeros((HMAX + 1, len(BANDS)))
S_n = np.zeros((HMAX + 1, len(BANDS)))
for (season, pid), pg in g.groupby(["SEASON", "PLAYER_ID"]):
    pg = pg.sort_values("GAME_DATE")
    yr = int(season[:4]) + 1
    cutoff = np.datetime64(f"{yr}-04-01")
    # his schedule: every game of whichever team he was with at the time (team of his latest row on or before the date)
    rd = pg.GAME_DATE.to_numpy()
    rt = pg.TEAM.to_numpy()
    sched = []
    for tm in pd.unique(rt):
        for d in team_dates[(season, tm)]:
            i = np.searchsorted(rd, d, side="right") - 1
            cur = rt[i] if i >= 0 else rt[0]
            if cur == tm:
                sched.append(d)
    sched = np.array(sorted(set(sched)))
    if len(sched) < 20:
        continue
    fp_by = dict(zip(rd, pg.fp.to_numpy()))
    pl_by = dict(zip(rd, pg.pl.to_numpy()))
    played = np.array([bool(pl_by.get(d, False)) for d in sched])
    fp = np.array([fp_by.get(d, 0.0) if pl_by.get(d, False) else 0.0 for d in sched])
    start = sched[0]
    for k in range(3, 30):
        d0 = start + np.timedelta64(7 * k, "D")
        if d0 >= cutoff:
            break
        past = sched < d0
        pm = past & played
        if pm.sum() < 8:
            continue
        lastn = np.where(past)[0][-2:]
        if not played[lastn].any():
            continue
        f_ = fp[pm]
        w = 0.5 ** (np.arange(len(f_))[::-1] / 8.0)
        L = float((f_ * w).sum() / w.sum())
        b = band_of(L)
        if b is None:
            continue
        fut = (sched >= d0) & (sched < cutoff)
        h = np.minimum(((sched[fut] - d0) / np.timedelta64(7, "D")).astype(int), HMAX)
        np.add.at(S_real[:, b], h, fp[fut])
        np.add.at(S_exp[:, b], h, p_future(L) * L)
        np.add.at(S_n[:, b], h, 1)
ratio = S_real / S_exp
ratio_all = S_real.sum(1) / S_exp.sum(1)
f = ratio / ratio[0:1, :]
f_all = ratio_all / ratio_all[0]
print("1. FIT (box scores 2010-11..2023-24): real / assumed points per scheduled game, by weeks ahead; decay factor = vs the same week")
print("   weeks ahead " + " ".join(f"{'L' + str(a) + '-' + (str(b) if b < 999 else ''):>12s}" for a, b in BANDS) + "         all")
for h in range(HMAX + 1):
    print(f"   {h:2d}{'+' if h == HMAX else ' '}        " + " ".join(f"{ratio[h, b]:.2f} ({f[h, b]:.2f})" for b in range(len(BANDS))) + f"   {ratio_all[h]:.2f} ({f_all[h]:.2f})   n {int(S_n[h].sum()):,}")
# smooth: monotone non-increasing in h per band (cumulative minimum), capped at 1 for h >= 1
fs = np.minimum.accumulate(np.minimum(f, 1.0), axis=0)
fs[0, :] = 1.0
fsa = np.minimum.accumulate(np.minimum(f_all, 1.0))
fsa[0] = 1.0
json.dump({"bands": BANDS, "hmax": HMAX, "f": fs.round(4).tolist(), "f_all": fsa.round(4).tolist(),
           "note": "decay of real vs assumed points per scheduled game by weeks ahead (rows) and level band (cols), relative to the same week; box scores 2010-11..2023-24 (horizon_decay_study.py)"},
          open(R / "horizon_decay.json", "w"))
print("wrote horizon_decay.json")

# ---------------- 2. test on the 2025-26 replay
sys.path.insert(0, R.as_posix())
import backtest_phase2 as P
import backtest_weekly_planner as B
from playoff_sim import simulate
import title_odds_curve as TC

REG_LAST = 19
ALL_W = [w for w in P.ALL_MP if w >= 2]
ANCHORS = [w for w in P.REG if w <= 16]


def factor(variant, h, L):
    if variant == "live" or h <= 0:
        return 1.0
    b = band_of(L)
    if b is None:
        return 1.0
    hh = min(h, HMAX)
    if variant == "full":
        return float(fs[hh, b])
    return float(fs[hh, b] / fsa[hh])


def solve(rost, w, anchor, variant):
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
            pf = B.p_future(p["level"]) * a * factor(variant, w - anchor, p["level"])
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
VARS = ["live", "full", "relative"]
raw = {}
for c in ANCHORS:
    for t in P.TEAMS:
        if (c, t) not in P.W:
            continue
        ro = TC.roster(t, c)
        for w in ALL_W:
            if w >= c:
                for v in VARS:
                    if v != "live" and w == c:
                        raw[(v, t, c, w)] = raw[("live", t, c, w)]
                    else:
                        raw[(v, t, c, w)] = solve(ro, w, c, v)
print("\n2. TEST (2025-26 replay): forecasts built", flush=True)
fcast, rows = {}, []
for v in VARS:
    for c in ANCHORS:
        past = [w for w in P.REG if w < c]
        pool = [(act[(t, w)], raw[(v, t, w, w)]) for t in P.TEAMS for w in past if (v, t, w, w) in raw and raw[(v, t, w, w)] > 0]
        lf = sum(a for a, _ in pool) / sum(f_ for _, f_ in pool) if pool else None
        for t in P.TEAMS:
            mine = [w for w in past if (v, t, w, w) in raw and raw[(v, t, w, w)] > 0]
            mult = lf * ((sum(act[(t, w)] for w in mine) / sum(raw[(v, t, w, w)] for w in mine)) / lf) ** (len(mine) / (len(mine) + 5)) if lf and mine else 0.95
            for w in ALL_W:
                if (v, t, c, w) in raw:
                    fc_ = mult * raw[(v, t, c, w)]
                    fcast[(v, t, c, w)] = fc_
                    if w > c and (t, w) in act and w <= REG_LAST and P.LEN[w] == 7:
                        rows.append(dict(v=v, team=t, c=c, w=w, h=w - c, r7=act[(t, w)] - fc_))
df = pd.DataFrame(rows)
df["r"] = df.r7 - df.groupby(["v", "c", "w"]).r7.transform("mean")
print(f"   {'variant':9s} {'bias':>7s} {'rest-of-season miss':>20s} {'same, league miss removed':>26s} {'tau wk2-6':>10s} {'tau wk7+':>9s}  bias by weeks ahead (1 / 2-3 / 4-6 / 7+)")
for v in VARS:
    d = df[df.v == v]
    gm = d.groupby(["team", "c"]).agg(m=("r7", "mean"), mr=("r", "mean"), n=("r7", "size")).reset_index()
    gm = gm[gm.n >= 3]
    taus = []
    for lo, hi in ((2, 6), (7, 16)):
        prods = []
        for _, gg in d[(d.c >= lo) & (d.c <= hi)].groupby(["team", "c"]):
            r = gg.r.values
            if len(r) >= 2:
                s = r.sum()
                prods.append((s * s - (r * r).sum()) / (len(r) * (len(r) - 1)))
        taus.append(math.sqrt(max(np.mean(prods), 0)))
    hb = [d[(d.h >= a) & (d.h <= b)].r7.mean() for a, b in ((1, 1), (2, 3), (4, 6), (7, 99))]
    print(f"   {v:9s} {d.r7.mean():+7.1f} {gm.m.abs().mean():20.1f} {gm.mr.abs().mean():26.1f} {taus[0]:10.1f} {taus[1]:9.1f}  " + " / ".join(f"{x:+.0f}" for x in hb))

import build_playoff_odds as O
teams_es, games = O.pull(2026)
from espn_api.basketball import League
import config
_raw = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID).espn_request.league_get(params={"view": ["mTeam"]})
seed = {t["id"]: t.get("playoffSeed") for t in _raw["teams"]}
final = {t["id"]: t.get("rankCalculatedFinal") for t in _raw["teams"]}
ids = sorted(teams_es)
Y = {"made": np.array([seed[t] <= 6 for t in ids], float), "bye": np.array([seed[t] <= 2 for t in ids], float), "title": np.array([final[t] == 1 for t in ids], float)}
champ = ids.index(next(t for t in ids if final[t] == 1))
PAR = json.load(open(R / "playoff_odds_params.json"))
from playoff_sim import season_tau


def ll(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


print("   outcomes (live drift settings, every checkpoint weeks 2-16):")
for v in VARS:
    sc = {k: [] for k in ("made", "bye", "title", "cp")}
    for c in ANCHORS:
        wins, pf = O.standings(games, upto=c - 1)
        for t in ids:
            wins.setdefault(t, 0)
            pf.setdefault(t, 0.0)
        pairs = [(gm_["week"], gm_["a"], gm_["b"]) for gm_ in games if gm_["week"] >= c]
        mu = {(t, w): fcast[(v, t, c, w)] for t in ids for w in ALL_W if (v, t, c, w) in fcast}
        r = simulate(ids, wins, pf, pairs, mu, dict(P.LEN), REG_LAST, tau=season_tau(PAR, c), sigma_e=PAR["sigma_e"], bias=0.0, n=6000, seed=3)
        for k in ("made", "bye", "title"):
            sc[k].append(ll(r[k], Y[k]))
        sc["cp"].append(float(r["title"][champ]))
    print(f"     {v:9s} log loss made {np.mean(sc['made']):.3f}  bye {np.mean(sc['bye']):.3f}  title {np.mean(sc['title']):.4f}   champion's avg title odds {np.mean(sc['cp']):.0%}")
