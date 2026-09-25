"""Usage flow: how much of a missing rotation player's production lands on each teammate (see RESEARCH_usage_flow.md).

  Delta_j = share_j * ( kappa[tier_j] * V + lambda[tier_j] * Vsim_j )              (fantasy points per game, for teammate j when he plays)
    V        = sum over absent teammates x of  p_out_x * fp_x                       (fp = fantasy points per game; only x averaging 12+ minutes)
    Vsim_j   = the same sum weighted by positional similarity of x and j (probabilities of playing guard / forward / center, from style)
    share_j  = fp_j^0.5 / sum over teammates i (1 - p_out_i) * fp_i^0.5             (only players averaging 6+ minutes)
    tier_j   = teammate's minutes tier (<15, 15-22, 22-30, 30+ mpg): low-minute players capture the most per share, starters the least
Parameters were fit on 14 seasons of box scores (2010-11 to 2023-24) in usage_flow_study2.py and tested out of sample.
Also: absence_survival(streak, k) = chance a player who has been out `streak` listings is still out k games later.
"""
import json
from pathlib import Path

import numpy as np

_M = json.load(open(Path(__file__).resolve().parent / "usage_flow_model.json"))
KAPPA, LAM, GAMMA = _M["kappa"], _M["lam"], _M["share_gamma"]
TIER_MPG = _M["tier_mpg"]
_CLASSES = _M["pos_classes"]
_COEF = np.array(_M["pos_coef"])
_INT = np.array(_M["pos_intercept"])
CAL_BINS, CAL = _M["cal_bins"], _M["cal"]
HAZ = {k: v for k, v in _M["hazard"].items()}
_BINS, _LABELS = _M["hazard_bins"], _M["hazard_labels"]


def pos_probs(reb36, ast36, blk36, stl36, tp36):
    """P(center), P(forward), P(guard) from per-36 rebounds, assists, blocks, steals, threes (multinomial model fit on box-score starters)"""
    z = _COEF @ np.array([reb36, ast36, blk36, stl36, tp36]) + _INT
    z = np.exp(z - z.max())
    p = z / z.sum()
    d = dict(zip(_CLASSES, p))
    return d["C"], d["F"], d["G"]


def tier_of(mpg):
    return int(np.searchsorted(TIER_MPG, mpg, side="right"))


def uplifts(players):
    """players: list of dicts for ONE NBA team on ONE day with keys
         id, fp (baseline fantasy points/game), mpg, pos (pC, pF, pG), p_out (0..1 chance he is out that day)
       returns {id: expected extra fantasy points per game if he plays}"""
    act = [p for p in players if p["mpg"] >= 6]
    if not act:
        return {}
    denom = sum((1 - p["p_out"]) * max(p["fp"], 0.5) ** GAMMA for p in act)
    absent = [p for p in players if p["p_out"] > 0 and p["mpg"] >= 12]
    for p in absent:
        p["_pe"] = p["p_out"] * p.get("w", 1.0)          # effective chance: long absences are partly inside the recent-form baseline already
    V = sum(p["_pe"] * p["fp"] for p in absent)
    out = {}
    if V <= 0 or denom <= 0:
        return {p["id"]: 0.0 for p in players}
    for j in players:
        if j["mpg"] < 6:
            out[j["id"]] = 0.0
            continue
        share = max(j["fp"], 0.5) ** GAMMA / denom
        vsim = sum(x["_pe"] * x["fp"] * sum(a * b for a, b in zip(x["pos"], j["pos"])) for x in absent if x["id"] != j["id"])
        vtot = V - (j["_pe"] * j["fp"] if (j["mpg"] >= 12 and j["p_out"] > 0) else 0.0)
        t = tier_of(j["mpg"])
        raw = share * (KAPPA[t] * vtot + LAM[t] * vsim)
        out[j["id"]] = max(0.0, raw * CAL[int(np.searchsorted(CAL_BINS, j["fp"], side="right"))])       # calibrated out of time: the raw model overshoots, most for stars
    return out


def absence_survival(streak, k):
    """P(still out k games from now | he has been listed Out `streak` times in a row): product of the per-step hazards"""
    p, s = 1.0, streak
    for _ in range(k):
        lab = next(l for l, hi in zip(_LABELS, _BINS[1:]) if s <= hi)
        p *= HAZ[lab]
        s += 1
    return p


def plan_boosts(days, team_players, absent, plays, team_game_no, w_recent=0.45):
    """Expected extra fantasy points per game for every teammate on each day.
       days            list of dates (objects with .isoformat())
       team_players    {team: [dict(id, fp, mpg, pos, name)]} for each NBA team (players who might play)
       absent          list of dict(id, team, status0 'Out'|'Doubtful'|'Questionable', streak int, listed {date_iso: status}, p_today for Q/D)
       plays(team, d)  whether the team plays that day;  team_game_no(team, d) = games the team plays between today and d (0 = today)
       returns {(id, date_iso): {"delta": float, "because": [names]}}"""
    out = {}
    by_team = {}
    for a in absent:
        by_team.setdefault(a["team"], []).append(a)
    for team, alist in by_team.items():
        roster = team_players.get(team, [])
        if not roster:
            continue
        for d in days:
            if not plays(team, d):
                continue
            k = team_game_no(team, d)
            ds = d.isoformat()
            pmap, why = {}, {}
            for a in alist:
                listed = a["listed"].get(ds)
                if listed == "Out":
                    p = 1.0
                elif listed == "Doubtful":
                    p = 0.97
                elif listed == "Questionable":
                    p = 0.5
                elif a["status0"] == "Out":
                    p = absence_survival(a["streak"], k) if k > 0 else 1.0
                elif k == 0:
                    p = a.get("p_today", 0.0)
                else:
                    p = 0.0
                if p > 0.02:
                    pmap[a["id"]] = (p, min(1.0, a["streak"] / 15.0) if a["status0"] == "Out" else 0.0)
            if not pmap:
                continue
            pl = []
            for r in roster:
                q = dict(r)
                p, sk = pmap.get(r["id"], (0.0, 0.0))
                q["p_out"], q["w"] = p, 1.0 - w_recent * sk
                pl.append(q)
            up = uplifts(pl)
            names = [r["name"] for r in roster if r["id"] in pmap and r["mpg"] >= 12]
            for r in roster:
                v = up.get(r["id"], 0.0)
                if v >= 0.3:
                    out[(r["id"], ds)] = {"delta": float(v), "because": names}
    return out
