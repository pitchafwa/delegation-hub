"""Trades Command Center data (2026-11-09) -> dashboard/trade_board.json.  Runs every morning in refresh_all.py (laptop and the cloud test alike).

  players   every rostered player: owner, levels, DELCO vs season average, expected points split into regular-season and fantasy-playoff weeks (availability-aware,
            the analyzer's own play chances), keeper asset, league price (trade_value.LeaguePerception) and TAGS with a one-line reason.
  quick     the instant trade estimate the page runs for any target (trade_proxy_study.py, 2026-11-09: with playoff games weighted, the full analyzer's best package was in its
            top 5 for 66 of 66 targets and its #1 lost 0.5 on average vs a spread of 24):
                now    = a x (regular-season points gained) + b x (fantasy-playoff points gained)      points: level x play chance per remaining day, 0.85 below your 10th man,
                                                                                                         forced drops (your worst healthy player), open spots refilled by a free agent,
                                                                                                         the streaming-spot bonus (trade_value.stream_bonus)
                future = w_future x lambda x 3.4 x league slope x (change in keeper strength + picks' value)   keeper strength as trade_value.keeper_strength with a shared
                                                                                                                 seeded random generator (mulberry32) so the page gets the same numbers
            a and b are re-fit EVERY MORNING against the full analyzer (about 150 packages, title-odds simulation), because they move with the standings; fit quality is stored.
  targets   for each other-team player, the best package by the quick estimate that his owner would at least consider (bands as the analyzer) = his Target score.
  ideas     the best few packages across the league, each run through the full analyzer.
Tag cutoffs from trade_tag_study.py (15 seasons, data/trade_tag_study.log).
  python research/build_trade_board.py [--fit 150] [--ideas 12]        (from ingest/)
"""
import itertools
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import trade_value as TV  # noqa: E402

ARGS = sys.argv[1:]
N_FIT = int(ARGS[ARGS.index("--fit") + 1]) if "--fit" in ARGS else 150
N_IDEAS = int(ARGS[ARGS.index("--ideas") + 1]) if "--ideas" in ARGS else 12
T0 = time.time()
random.seed(5)
DRAWS = 300
GIVE_POOL, OTHER_POOL, MAX_PICK_ROUND = 12, 8, 2
DEFAULT_AB = (0.0, 0.029)        # study fit (regular, playoff) if the morning fit fails
# trade_tag_study.py: players 5+ under DELCO averaged about 0.83 x the gap more over their next 20 games; 7+ over gave back about 0.42 x the gap (and stayed above DELCO)
BUY_GAP, BUY_BACK = -5.0, 0.83
SELL_GAP, SELL_BACK = 7.0, 0.42
PERF_GAP = 5.0                   # under / overperforming vs ESPN's preseason number
ROLE_JUMP, ROLE_KEEP = 4.0, 0.33
PO_GAMES = 12                    # games in the fantasy-playoff weeks (most players get 10-11)
KEEPER_RANK = 60                 # 12 teams x 5 keepers


def key(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)


# ---------------- shared random numbers (the page runs the same generator)
def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt


_Z = {}


def zdraws(pid):
    if pid not in _Z:
        r = mulberry32(int(pid))
        z = np.empty(DRAWS)
        for i in range(DRAWS):
            u1, u2 = max(r(), 1e-12), r()
            z[i] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        _Z[pid] = z
    return _Z[pid]


_NOISY = {}


def noisy(p):
    k = (p["id"], p["asset_f"])
    if k not in _NOISY:
        z = zdraws(p["id"])
        _NOISY[k] = np.maximum(p["asset_f"] * (1 + TV.ASSET_NOISE * z) + 0.8 * z, 0)
    return _NOISY[k]


def ks_next(players):
    """next-season points above the cutoff of the soft top 5 (TV.keeper_next on the shared draws)"""
    ps = [p for p in players if p["asset_f"] > 0]
    if not ps:
        return 0.0
    M = np.stack([noisy(p) for p in ps])
    top = np.argsort(-M, axis=0, kind="stable")[:5]
    v1 = np.array([p.get("v1", 0.0) for p in ps])
    return float(v1[top].sum(axis=0).mean())


def ks(players):
    """keeper strength (trade_value.keeper_strength's formula) on the shared draws"""
    rows = [noisy(p) for p in players if p["asset_f"] > 0]
    if not rows:
        return 0.0
    M = np.sort(np.stack(rows), axis=0)[::-1]
    return float(M[:5].sum(axis=0).mean() + TV.CHIP_WEIGHT * M[5:9].sum(axis=0).mean())


def main():
    W = TV.World()
    inp = TV.sim_inputs(W)
    ids = inp[0]
    LP = TV.LeaguePerception(W)
    me_ab = W.wp["my_abbrev"]
    me_id = W.by_abbrev[me_ab]
    ctx = W.ctx or {}
    s_next = ctx.get("slope_mean", TV.DEFAULT_SLOPE)
    post = (ctx.get("teams", {}).get(str(me_id)) or {}).get("posture", "balanced")
    w_now, w_fut = TV.POSTURE[post]
    c_fut = w_fut * TV.FUTURE_LAMBDA * TV.GAMES_PER_WEEK_PER_ASSET_PT * s_next
    c_next = max(w_now, w_fut) * TV.GAMES_PER_WEEK_PER_ASSET_PT * TV.next_slope(W, me_id) if TV.NEXT_SPLIT else None     # next season like this one (trade_value.NEXT_SPLIT)
    po_set = set(W.playoff_weeks)
    weeks = list(range(W.first, W.last + 1))
    reg_wk = sum(W.weeks[w].get("days", 7) / 7.0 for w in weeks if w not in po_set)
    po_wk = sum(W.weeks[w].get("days", 7) / 7.0 for w in weeks if w in po_set)
    rd = {int(x["id"]): x for x in json.load(open(TV.HUB / "redraft_data.json", encoding="utf-8"))["players"]}
    fs = json.load(open(TV.HUB / "form_split.json", encoding="utf-8")).get("players", {})
    late = json.load(open(TV.HUB / "late_season.json", encoding="utf-8")).get("teams", {})

    # ---------------- every rostered player
    def split(pl):
        r = q = 0.0
        for w in weeks:
            for d in W.wk_days(w):
                v = W.p_play(pl, d, w) * pl["level"]
                if w in po_set:
                    q += v
                else:
                    r += v
        return r, q

    rosters = {tid: TV.fix_roster(W, W.roster(tid), str(tid))[0] for tid in W.teams}
    P = {}
    for tid, ro in rosters.items():
        for p in ro:
            er, ep = split(p)
            p.update({"tid": tid, "er": er, "ep": ep, "pv": LP.pv(p["id"])})
            P[p["id"]] = p
    t5 = {tid: LP.top5(tid) for tid in W.teams}
    af_rank = {pid: i + 1 for i, pid in enumerate(sorted(P, key=lambda i: -P[i]["asset_f"]))}
    fa_sorted = sorted((f["level"] for f in W.fa_pool), reverse=True)
    fa_level = fa_sorted[2] if len(fa_sorted) >= 3 else 24.0

    # ---------------- picks (next draft), from the simulation's draft-position odds
    mu0 = {(t, w): v for t in W.teams for w, v in W.base_proj[t].items()}
    base = TV.run_odds(W, inp, mu0, 20000)
    # what a point is worth in title odds, measured on the simulation itself: +DELTA a week in your regular-season weeks, then in your playoff weeks (same random numbers).
    # 2026-11-09: fitting these two weights on ~145 full checks alone was unstable (regular/playoff points move together), so the simulation sets them and the checks only
    # fit one overall scale.
    DELTA = 25.0
    ids_i = ids.index(me_id)
    b40 = TV.run_odds(W, inp, mu0, 40000)
    rw = [w for w in W.base_proj[me_id] if w >= W.first and w not in po_set]
    pw = [w for w in W.base_proj[me_id] if w in po_set]
    tr_ = TV.run_odds(W, inp, {k: v + (DELTA if k[0] == me_id and k[1] in rw else 0) for k, v in mu0.items()}, 40000)
    tp_ = TV.run_odds(W, inp, {k: v + (DELTA if k[0] == me_id and k[1] in pw else 0) for k, v in mu0.items()}, 40000)
    A0 = w_now * 100 * (float(tr_["title"][ids_i]) - float(b40["title"][ids_i])) / (DELTA * max(1, len(rw)))
    B0 = w_now * 100 * (float(tp_["title"][ids_i]) - float(b40["title"][ids_i])) / (DELTA * max(1, len(pw)))
    print(f"title points per point: regular {A0:.5f}, playoffs {B0:.5f}", flush=True)
    owners = TV.pick_owner_map(W)
    l2v = TV.level_to_market()
    picks = {}
    for t in W.teams:
        ab = W.abbrev_of[t]
        for rnd in range(1, MAX_PICK_ROUND + 1):
            probs = TV.pick_slot_probs(base["draft_pos"][ids.index(t)], rnd)
            eq, _ = TV.pick_value(probs, l2v)
            en, el = TV.pick_split(probs)
            cum, lo, hi = 0.0, None, None
            for s in sorted(probs):
                cum += probs[s]
                if lo is None and cum >= 0.1:
                    lo = s
                if hi is None and cum >= 0.9:
                    hi = s
            k = f"PICK:{ab}:{rnd}"
            picks[k] = {"k": k, "orig": ab, "round": rnd, "owner": owners.get((t, rnd), t), "eq": eq, "en": en, "el": el, "pv": TV.pick_pv(probs),
                        "lo": (lo - 1) % 12 + 1, "hi": (hi - 1) % 12 + 1, "pick": True, "name": f"{ab} round {rnd}"}

    # ---------------- the quick estimate (the page runs the same function)
    R_ME = rosters[me_id]
    ks0 = ks(R_ME)
    ksn0 = ks_next(R_ME)
    sb0 = TV.stream_bonus(R_ME)

    def spots(ro):
        return len(ro) - min(TV.IR_SLOTS, sum(1 for p in ro if p["out"]))

    # How much of a player's expected points reach YOUR lineup depends on where he ranks on your roster (10 slots a day, the 40-start cap, and injured players returning
    # for the playoff weeks). Measured every morning on your real roster with the analyzer's own lineup solver: remove each player, see what the lineup loses, divide by
    # his expected points; by rank (regular-season and playoff weeks separately), never rising down the ranks. 2026-11-09: a flat 0.85-below-the-10th-man rule credited
    # 2-for-1 depth deals with ~190 playoff points the lineups did not use.
    def wsplit(ro):
        wk = W.weekly(ro)
        return sum(v for w, v in wk.items() if w not in po_set), sum(v for w, v in wk.items() if w in po_set)
    V0 = wsplit(R_ME)
    marg = []
    for p in R_ME:
        r, q = wsplit([x for x in R_ME if x is not p])
        marg.append((p, (V0[0] - r) / p["er"] if p["er"] > 1 else None, (V0[1] - q) / p["ep"] if p["ep"] > 1 else None))

    def curve(idx, key):
        rows = sorted(marg, key=lambda m: -m[0][key])
        f, last = [], 1.0
        for m in rows:
            v = m[idx] if m[idx] is not None else last
            v = min(max(v, 0.0), 1.1, last if f else 1.1)
            f.append(round(v, 4))
            last = v
        return f
    F_REG, F_PO = curve(1, "er"), curve(2, "ep")
    print("lineup share by rank: regular", F_REG, " playoffs", F_PO, flush=True)
    fa_e = (fa_level * 0.9 * 3.3 * reg_wk, fa_level * 0.9 * 3.3 * po_wk)

    def seg(ro, key, f):
        vals = sorted((p[key] for p in ro), reverse=True)
        return sum(v * f[min(i, len(f) - 1)] for i, v in enumerate(vals))

    open0 = TV.ROSTER_SPOTS - spots(R_ME)
    FA = lambda k: {"id": -k - 1, "level": fa_level, "out": False, "asset_f": 0.0, "er": fa_e[0], "ep": fa_e[1]}
    base0 = R_ME + [FA(k) for k in range(max(0, open0))]
    B_REG, B_PO = seg(base0, "er", F_REG), seg(base0, "ep", F_PO)

    def parts(give, get):
        """(regular points, playoff points, keeper+pick change, players dropped, free agents added)"""
        gp = [x for x in give if x.get("pick")]
        gv = [x for x in give if not x.get("pick")]
        rp = [x for x in get if x.get("pick")]
        rv = [x for x in get if not x.get("pick")]
        gids = {x["id"] for x in gv}
        new = [p for p in R_ME if p["id"] not in gids] + rv
        op = TV.ROSTER_SPOTS - spots(new)
        ndrop = max(0, -op) - max(0, -open0)
        drops = []
        if ndrop > 0:
            got = {x["id"] for x in rv}
            drops = sorted((p for p in new if not p["out"] and p["id"] not in got), key=lambda p: p["level"])[:ndrop]
            new = [p for p in new if p not in drops]
        filled = new + [FA(k) for k in range(max(0, op))]
        dsb = TV.stream_bonus(filled) - sb0
        reg = seg(filled, "er", F_REG) - B_REG + dsb * reg_wk
        po = seg(filled, "ep", F_PO) - B_PO + dsb * po_wk
        if c_next is None:
            fv = c_fut * (ks(new) - ks0 + sum(x["eq"] for x in rp) - sum(x["eq"] for x in gp))
        else:
            dn = ks_next(new) - ksn0
            fv = c_next * (dn + sum(x["en"] for x in rp) - sum(x["en"] for x in gp)) + c_fut * (ks(new) - ks0 - dn + sum(x["el"] for x in rp) - sum(x["el"] for x in gp))
        return reg, po, fv, drops, max(0, op) - max(0, open0)

    def accept(give, get, tid):
        return LP.accept([x["id"] for x in give if not x.get("pick")], [x["id"] for x in get if not x.get("pick")], tid,
                         [x["pv"] for x in give if x.get("pick")], [x["pv"] for x in get if x.get("pick")])

    my_pool = sorted((p for p in R_ME if not p["ir"]), key=lambda p: -p["pv"])[:GIVE_POOL] + [x for x in picks.values() if x["owner"] == me_id]

    def packages(T):
        tid = T["tid"]
        others = sorted((p for p in rosters[tid] if p["id"] != T["id"] and not p["ir"]), key=lambda p: -p["pv"])[:OTHER_POOL] + [x for x in picks.values() if x["owner"] == tid]
        others = [o for o in others if o["pv"] <= T["pv"]]          # he stays the main piece: a throw-in, not a bigger star, so his score is about him
        gets = [(T,)] + [(T, o) for o in others]
        for n in (1, 2, 3):
            for G in itertools.combinations(my_pool, n):
                if sum(1 for x in G if x.get("pick")) > 1:
                    continue
                for R in gets:
                    yield G, R

    def label(x):
        return x["k"] if x.get("pick") else x["name"]

    # ---------------- morning fit of a, b against the full analyzer
    targets = [p for tid, ro in rosters.items() if tid != me_id for p in ro if not p["ir"] and (p["pv"] >= 3 or p["level"] >= 28)]
    sample, ab = [], (A0, B0)
    fit_targets = random.sample(targets, min(len(targets), max(1, N_FIT // 5)))
    for T in fit_targets:
        cands = []
        for G, R in packages(T):
            if accept(G, R, T["tid"])["band"] == "unlikely":
                continue
            r, q, dk, _, _ = parts(G, R)
            cands.append((ab[0] * r + ab[1] * q + dk, G, R, r, q, dk))
        if not cands:
            continue
        cands.sort(key=lambda c: -c[0])
        chosen = cands[:3] + random.sample(cands[3:], min(2, len(cands) - 3))
        for _, G, R, r, q, dk in chosen:
            try:
                res = TV.evaluate(me_ab, W.abbrev_of[T["tid"]], [label(x) for x in G], [label(x) for x in R], n=10000, world=W, inp=inp)
            except SystemExit:
                continue
            sample.append({"t": T["id"], "r": r, "q": q, "dk": dk, "now": res["me"]["value"]["now"], "total": res["me"]["value"]["total"]})
    fit = {"n": len(sample), "a": ab[0], "b": ab[1], "sim_a": A0, "sim_b": B0, "scale": 1.0, "fallback": True}
    if len(sample) >= 30:
        def scale(rows):
            x = np.array([A0 * s_["r"] + B0 * s_["q"] for s_ in rows])
            y = np.array([s_["now"] for s_ in rows])
            return max(0.0, float(x @ y / (x @ x))) if (x @ x) > 0 else 1.0
        tl = sorted({s_["t"] for s_ in sample})
        half = set(tl[::2])
        tr = [s_ for s_ in sample if s_["t"] in half]
        te = [s_ for s_ in sample if s_["t"] not in half]
        k_tr = scale(tr)
        pred = [k_tr * (A0 * s_["r"] + B0 * s_["q"]) + s_["dk"] for s_ in te]
        rk = lambda v: np.argsort(np.argsort(v))
        rho = float(np.corrcoef(rk(pred), rk([s_["total"] for s_ in te]))[0, 1]) if len(te) > 3 else None
        k = scale(sample)
        ab = (k * A0, k * B0)
        fit = {"n": len(sample), "a": ab[0], "b": ab[1], "sim_a": A0, "sim_b": B0, "scale": round(k, 3), "fallback": False, "holdout_rank_corr": round(rho, 3) if rho is not None else None,
               "mean_abs_err": round(float(np.mean([abs(ab[0] * s_["r"] + ab[1] * s_["q"] + s_["dk"] - s_["total"]) for s_ in sample])), 2)}
    print(f"fit: {fit}  ({time.time() - T0:.0f}s)", flush=True)

    def value(G, R):
        r, q, dk, drops, _ = parts(G, R)
        return ab[0] * r + ab[1] * q + dk, drops

    # ---------------- target scores
    best, alts = {}, {}
    for T in targets:
        found = []
        for G, R in packages(T):
            acc = accept(G, R, T["tid"])
            if acc["band"] == "unlikely":
                continue
            v, drops = value(G, R)
            found.append((v, G, R, acc["band"], drops))
        if found:
            found.sort(key=lambda f: -f[0])
            top = found[0]
            best[T["id"]] = {"v": round(top[0], 2), "give": [label(x) for x in top[1]], "get": [label(x) for x in top[2]], "band": top[3], "drops": [p["name"] for p in top[4]]}
            alts[T["id"]] = [(f[0], [label(x) for x in f[1]], [label(x) for x in f[2]]) for f in found[:8]]
    print(f"targets scored: {len(best)}  ({time.time() - T0:.0f}s)", flush=True)

    # ---------------- lineup check (2026-11-09, Tommy approved): the quick estimate ignores positions (Amen + Peterson + Clingan for Jokic: quick +18, full +8), so each
    # target's best CHECK_TOP different packages go through the full analyzer and his Target score is the best FULL value. The page shows that check on the trade sheet.
    CHECK_TOP = 2
    full_res = {}
    for pid in list(best):
        tried, seen_core = [], set()
        for v, g, r in alts[pid]:
            if len(tried) >= CHECK_TOP:
                break
            core = (tuple(sorted(x for x in g if not x.startswith("PICK:"))), tuple(sorted(r)))
            if core in seen_core:
                continue
            seen_core.add(core)
            try:
                res = TV.evaluate(me_ab, W.abbrev_of[P[pid]["tid"]], g, r, n=10000, world=W, inp=inp)
            except SystemExit:
                continue
            tried.append((res["me"]["value"]["total"], v, g, r, res))
        if not tried:
            continue
        def compact(res):
            m, t = res["me"], res["them"]
            return {"me": {k: m[k] for k in ("odds_before", "odds_after", "d_title_pp", "keeper_strength", "drops", "adds", "season_points")} | {"value": m["value"]},
                    "them": {"abbrev": t["abbrev"], "odds_before": t["odds_before"], "odds_after": t["odds_after"], "value": t["value"]}}
        tot, v, g, r, res = max(tried, key=lambda t: t[0])
        best[pid] = {"v": round(tot, 2), "quick": round(v, 2), "give": g, "get": r, "band": res["market"]["band"], "drops": res["me"]["drops"], "full": compact(res),
                     "checked": [{"give": g_, "get": r_, "full": compact(x_)} for _, _, g_, r_, x_ in tried]}
        full_res[pid] = res
    print(f"lineup-checked: {len(full_res)}  ({time.time() - T0:.0f}s)", flush=True)

    # ---------------- ideas: the best checked packages across the league
    ideas, used, seen = [], {}, set()
    for pid in sorted(full_res, key=lambda i: -best[i]["v"]):
        b_ = best[pid]
        if len(ideas) >= N_IDEAS or b_["v"] <= 0.3:
            break
        key_ = (tuple(sorted(b_["give"])), tuple(sorted(b_["get"])))
        if key_ in seen or any(used.get(n, 0) >= 2 for n in b_["give"]):
            continue
        seen.add(key_)
        for n in b_["give"]:
            used[n] = used.get(n, 0) + 1
        res = full_res[pid]
        ideas.append({"target": pid, "partner": W.abbrev_of[P[pid]["tid"]], "give": b_["give"], "get": b_["get"], "quick": b_["quick"], "me": res["me"], "them": res["them"], "market": res["market"], "verdict": res["verdict"]})
    ideas.sort(key=lambda i: -i["me"]["value"]["total"])
    print(f"ideas: {len(ideas)}  ({time.time() - T0:.0f}s)", flush=True)

    # ---------------- tags
    def tags(p):
        out = []
        r = rd.get(p["id"]) or {}
        delco, avg, n = r.get("ppg"), r.get("s_avg"), r.get("n_season") or 0
        if delco and avg is not None and n >= 5 and r.get("sa_season", r.get("gs")) == r.get("gs"):
            gap = avg - delco
            if gap <= BUY_GAP:
                out.append(["Buy low", f"Averaging {avg:.1f}, DELCO {delco:.1f}. Players this far under have averaged about {BUY_BACK * -gap:.0f} more over their next 20 games."])
            elif gap >= SELL_GAP:
                out.append(["Sell high", f"Averaging {avg:.1f}, DELCO {delco:.1f}. Hot starts like this have given back about {SELL_BACK * gap:.0f} over the next 20 games."])
            espn = r.get("espn_ppg") or 0
            if espn > 0:
                d = avg - espn
                if d <= -PERF_GAP:
                    out.append(["Underperforming", f"{-d:.1f} under his preseason projection of {espn:.1f}."])
                elif d >= PERF_GAP:
                    out.append(["Overperforming", f"{d:.1f} over his preseason projection of {espn:.1f}."])
        if (r.get("gpo") or 0) >= PO_GAMES and p["level"] >= 25 and (late.get(p["team"]) or {}).get("core_mult", 1.0) >= 0.98:
            out.append(["Playoff target", f"{r['gpo']} games in the fantasy-playoff weeks (most players get 10-11)."])
        adv = p.get("advice") or {}
        if p["out"] and p["level"] >= 30 and (adv.get("p_back_playoffs") or 0) >= 0.75:
            g = adv.get("plan_games")
            out.append(["Injury target", (f"Out about {g:.0f} more games; " if g else "Out now; ") + f"{100 * adv['p_back_playoffs']:.0f}% chance he is back for the fantasy playoffs."])
        f = fs.get(key(p["name"])) or {}
        if (f.get("n") or 0) >= 5 and f.get("m3") is not None and f.get("m") is not None and f["m3"] - f["m"] >= ROLE_JUMP:
            out.append(["Rising role", f"Minutes up {f['m3'] - f['m']:.1f} over his last 3 games ({f['m3']:.0f} vs {f['m']:.0f}). About a third of a jump like this usually holds."])
        if af_rank[p["id"]] <= KEEPER_RANK and p["asset_f"] > 0:
            out.append(["Keeper asset", f"#{af_rank[p['id']]} in keeper value among rostered players."])
        if p["id"] in t5.get(p["tid"], set()) and p["tid"] != me_id:
            out.append(["Their keeper", f"One of {W.abbrev_of[p['tid']]}'s 5 most valuable players, so expect to pay extra."])
        tpost = (ctx.get("teams", {}).get(str(p["tid"])) or {}).get("posture")
        if tpost == "rebuild" and p["tid"] != me_id:
            out.append(["Seller", f"{W.abbrev_of[p['tid']]} looks like a rebuilding team."])
        return out

    # value to you of each of your own players (what losing him for nothing costs, by the quick estimate)
    my_cost = {p["id"]: round(-value([p], [])[0], 2) for p in R_ME}

    players = []
    for pid, p in P.items():
        r = rd.get(pid) or {}
        players.append({"id": pid, "n": p["name"], "tid": p["tid"], "t": W.abbrev_of[p["tid"]], "nba": p["team"], "pos": r.get("pos") or "/".join(s for s in p["slots"] if s in ("PG", "SG", "SF", "PF", "C")),
                        "slots": p["slots"], "lvl": round(p["level"], 1), "er": round(p["er"], 1), "ep": round(p["ep"], 1), "af": round(p["asset_f"], 2), "v1": round(p.get("v1", 0.0), 2), "out": p["out"], "ir": p["ir"],
                        "pv": round(p["pv"], 2), "t5": pid in t5.get(p["tid"], set()), "delco": r.get("ppg"), "avg": r.get("s_avg"), "gp": r.get("n_season"), "espn": r.get("espn_ppg"),
                        "gpo": r.get("gpo"), "age": p.get("age"), "back": r.get("back"), "tags": tags(p), "best": best.get(pid), "cost": my_cost.get(pid)})
    players.sort(key=lambda x: -((x["best"] or {}).get("v") or -99))
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "me": me_ab, "me_id": me_id, "posture": post,
           "params": {"a": ab[0], "b": ab[1], "c_fut": c_fut, "c_next": c_next, "f_reg": F_REG, "f_po": F_PO, "fa_level": fa_level, "fa_er": fa_e[0], "fa_ep": fa_e[1], "reg_wk": reg_wk, "po_wk": po_wk,
                      "spots": TV.ROSTER_SPOTS, "ir": TV.IR_SLOTS, "stream7": TV.STREAM_SPOT_7, "asset_noise": TV.ASSET_NOISE, "chip": TV.CHIP_WEIGHT, "draws": DRAWS,
                      "lp": TV.LEAGUE_PV, "give_pool": GIVE_POOL, "other_pool": OTHER_POOL},
           "fit": fit, "order": {str(tid): [p["id"] for p in ro] for tid, ro in rosters.items()}, "picks": list(picks.values()), "players": players, "ideas": ideas, "seconds": round(time.time() - T0)}
    for k in out["picks"]:
        k["eq"], k["pv"], k["en"], k["el"] = round(k["eq"], 2), round(k["pv"], 2), round(k["en"], 2), round(k["el"], 2)
    (TV.HUB / os.environ.get("TB_OUT", "trade_board.json")).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote trade_board.json: {len(players)} players, {len(best)} targets, {len(ideas)} ideas in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
