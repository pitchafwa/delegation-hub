"""Trade proxy study (2026-11-09, Tommy: pick a target, get a handful of trade ideas quickly; full title-odds simulation only when looking deeper).

Question: which quick estimate ranks trade packages the way the full analyzer (trade_value.evaluate: lineups for every week + playoff simulation) does?
The keeper ("future") half of the analyzer does not use the simulation, so every proxy computes it exactly; the proxies differ only in the season half.

  P1  QUICK      find_trades.quick_me: rest-of-season points (level x games left, 0.85 below your 10th man), drops/free-agent refill, streaming spot, x your title slope
  P2  QUICK-PO   P1 with playoff-week games split out and weighted (weight fitted, leave-one-team-out)
  P3  LINEUP     the analyzer's own weekly lineups (W.weekly, after the same forced drops / free-agent fills) with no simulation: regular and playoff point changes,
                 two weights fitted leave-one-team-out
Sample: your roster vs every other team; targets = each team's top 6 by league perception; packages 1-1, 2-1, 3-1, 1-2, 2-2 from your top 12, kept only if the
other owner would see them as at least 'possible'; up to 22 per target (P1's best 8 + a random 14). Full analyzer at 20,000 simulations; noise floor from 20,000 vs 80,000.
Metrics: rank correlation with the full value overall and within each target; for each target, how much full value you lose by taking each proxy's top pick instead
of the true best, and how often the true best is in the proxy's top 5.
  python research/trade_proxy_study.py        (from ingest/)        writes research/logs/trade_proxy_study.json
"""
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import trade_value as TV  # noqa: E402

random.seed(11)
NSIM = 20000
T0 = time.time()
W = TV.World()
inp = TV.sim_inputs(W)
LP = TV.LeaguePerception(W)
me_ab = W.wp["my_abbrev"]
me_id = W.by_abbrev[me_ab]
R_ME, _, _ = TV.fix_roster(W, W.roster(me_id), "me")
mine = [p for p in R_ME if not p["ir"]]
tenth = sorted((p["level"] for p in mine), reverse=True)[9] if len(mine) >= 10 else 0.0
ctx = W.ctx or {}
slope = ctx.get("slope", {}).get(str(me_id), TV.DEFAULT_SLOPE)
s_next = ctx.get("slope_mean", TV.DEFAULT_SLOPE)
fa_level = sorted((f["level"] for f in W.fa_pool), reverse=True)[2] if len(W.fa_pool) >= 3 else 24.0
weeks = list(range(W.first, W.last + 1))
nweeks = len(weeks)
po = set(W.playoff_weeks)
k_me = TV.keeper_strength(R_ME)


def g_split(team):
    """(regular-season games left, playoff games) for an NBA team"""
    reg = sum(1 for w in weeks if w not in po for d in W.wk_days(w) if W.plays(team, d))
    pog = sum(1 for w in weeks if w in po for d in W.wk_days(w) if W.plays(team, d))
    return reg, pog


GS = {}


def gs(p):
    if p["team"] not in GS:
        GS[p["team"]] = g_split(p["team"])
    return GS[p["team"]]


def quick_parts(give, get):
    """(regular points, playoff points) change for you, P1 style"""
    def c(p):
        f = p["level"] * (1.0 if p["level"] >= tenth else 0.85)
        r, q = gs(p)
        return f * r, f * q
    reg = po_ = 0.0
    for p in get:
        a, b = c(p); reg += a; po_ += b
    for p in give:
        a, b = c(p); reg -= a; po_ -= b
    extra = len(get) - len(give)
    if extra > 0:
        for p in sorted(mine, key=lambda p: p["level"])[:extra]:
            a, b = c(p); reg -= a; po_ -= b
    elif extra < 0:
        reg_w = sum(1 for w in weeks if w not in po)
        reg += -extra * fa_level * 0.85 * 3.3 * reg_w
        po_ += -extra * fa_level * 0.85 * 3.3 * (nweeks - reg_w)
    gid = {g["id"] for g in give}
    new = [p for p in R_ME if p["id"] not in gid] + list(get)
    filled = new + [{"id": -k, "level": fa_level, "out": False, "asset_f": 0.0} for k in range(max(0, -extra))]
    sb = (TV.stream_bonus(filled) - TV.stream_bonus(R_ME))
    reg += sb * (nweeks - len(po)); po_ += sb * len(po)
    return reg, po_, new


def lineup_parts(give, get):
    gid = {g["id"] for g in give}
    new = [p for p in R_ME if p["id"] not in gid] + list(get)
    new, _, _ = TV.fix_roster(W, new, "me")
    b, a = W.weekly(R_ME), W.weekly(new)
    s0, s1 = TV.stream_bonus(R_ME), TV.stream_bonus(new)
    d = {w: (a[w] + s1 * W.weeks[w].get("days", 7) / 7.0) - (b[w] + s0 * W.weeks[w].get("days", 7) / 7.0) for w in weeks}
    return sum(v for w, v in d.items() if w not in po), sum(v for w, v in d.items() if w in po)


rows = []
targets = []
for ab, tid in W.by_abbrev.items():
    if tid == me_id:
        continue
    theirs = sorted([p for p in W.roster(tid) if not p["ir"]], key=lambda p: -LP.pv(p["id"]))[:9]
    for T in theirs[:6]:
        targets.append((ab, tid, T, [p for p in theirs if p["id"] != T["id"]]))
my_pool = sorted(mine, key=lambda p: -LP.pv(p["id"]))[:12]
print(f"{len(targets)} targets, load {time.time() - T0:.0f}s", flush=True)

for ti, (ab, tid, T, others) in enumerate(targets):
    pk = []
    for n in (1, 2, 3):
        for G in itertools.combinations(my_pool, n):
            pk.append((G, (T,)))
    for G in my_pool:
        for O in others:
            pk.append(((G,), (T, O)))
    for G in itertools.combinations(my_pool, 2):
        for O in others[:4]:
            pk.append((G, (T, O)))
    ok = []
    for G, Rr in pk:
        acc = LP.accept([p["id"] for p in G], [p["id"] for p in Rr], tid)
        if acc["band"] == "unlikely":
            continue
        reg, po_, new = quick_parts(G, Rr)
        ok.append((G, Rr, acc, reg, po_, new))
    if not ok:
        continue
    ok.sort(key=lambda o: -(o[3] + o[4]))
    pick = ok[:8] + random.sample(ok[8:], min(14, len(ok) - 8))
    for G, Rr, acc, reg, po_, new in pick:
        res = TV.evaluate(me_ab, ab, [p["name"] for p in G], [p["name"] for p in Rr], n=NSIM, world=W, inp=inp)
        lr, lp = lineup_parts(G, Rr)
        rows.append({"target": f"{ab}:{T['name']}", "team": ab, "give": [p["name"] for p in G], "get": [p["name"] for p in Rr], "shape": f"{len(G)}-{len(Rr)}",
                     "band": acc["band"], "q_reg": reg, "q_po": po_, "l_reg": lr, "l_po": lp, "full_now": res["me"]["value"]["now"], "fut": res["me"]["value"]["future"],
                     "full": res["me"]["value"]["total"], "w_now": res["me"]["value"]["w_now"]})
    print(f"[{ti + 1}/{len(targets)}] {ab} {T['name']}: {len(ok)} acceptable, ran {len(pick)}  ({time.time() - T0:.0f}s)", flush=True)

# noise floor: 40 random rows again at 80,000
noise = []
for r in random.sample(rows, min(40, len(rows))):
    res = TV.evaluate(me_ab, r["team"], r["give"], r["get"], n=80000, world=W, inp=inp)
    noise.append((r["full"], res["me"]["value"]["total"]))


def spearman(a, b):
    a, b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


wn = rows[0]["w_now"] if rows else 1.0
scale = wn * slope / nweeks              # points over the season -> title pp (P1's own conversion), times the posture weight


def fit(xr, xp, y, keys):
    """least squares y_now ~ a*reg + b*po, leave-one-team-out; returns out-of-sample predictions"""
    xr, xp, y = map(np.array, (xr, xp, y))
    pred = np.zeros(len(y))
    for k in set(keys):
        tr = np.array([kk != k for kk in keys])
        X = np.stack([xr[tr], xp[tr]], 1)
        coef, *_ = np.linalg.lstsq(X, y[tr], rcond=None)
        pred[~tr] = coef[0] * xr[~tr] + coef[1] * xp[~tr]
    X = np.stack([xr, xp], 1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pred, coef


keys = [r["team"] for r in rows]
fut = np.array([r["fut"] for r in rows])
y_now = np.array([r["full_now"] for r in rows])
full = np.array([r["full"] for r in rows])
P = {}
P["P1 quick"] = scale * np.array([r["q_reg"] + r["q_po"] for r in rows]) + fut
p2, c2 = fit([r["q_reg"] for r in rows], [r["q_po"] for r in rows], y_now, keys)
P["P2 quick+playoff wt"] = p2 + fut
p3, c3 = fit([r["l_reg"] for r in rows], [r["l_po"] for r in rows], y_now, keys)
P["P3 lineup, no sim"] = p3 + fut
P["future only"] = fut

out = {"n_rows": len(rows), "n_targets": len({r["target"] for r in rows}), "sims": NSIM,
       "noise_floor": {"spearman_20k_vs_80k": round(spearman([a for a, _ in noise], [b for _, b in noise]), 3),
                       "mean_abs_diff": round(float(np.mean([abs(a - b) for a, b in noise])), 3)},
       "coef": {"P2": [float(x) for x in c2], "P3": [float(x) for x in c3], "P1_scale": scale}, "proxies": {}}
by_t = {}
for i, r in enumerate(rows):
    by_t.setdefault(r["target"], []).append(i)
for name, pr in P.items():
    within, regret, hit5, hit3 = [], [], [], []
    for t, ix in by_t.items():
        if len(ix) < 5:
            continue
        f, q = full[ix], pr[ix]
        within.append(spearman(f, q))
        best = int(np.argmax(f))
        regret.append(float(f[best] - f[int(np.argmax(q))]))
        top5 = set(np.argsort(-q)[:5].tolist())
        hit5.append(best in top5)
        hit3.append(len(set(np.argsort(-f)[:3].tolist()) & top5) / 3)
    sign = [(a > 0) == (b > 0) for a, b in zip(full, pr) if abs(a) >= 0.5]
    out["proxies"][name] = {"spearman_all": round(spearman(full, pr), 3), "spearman_within_target_median": round(float(np.median(within)), 3),
                            "best_in_top5": round(float(np.mean(hit5)), 3), "share_of_true_top3_in_top5": round(float(np.mean(hit3)), 3),
                            "regret_median": round(float(np.median(regret)), 3), "regret_mean": round(float(np.mean(regret)), 3),
                            "regret_p90": round(float(np.percentile(regret, 90)), 3), "sign_agree_material": round(float(np.mean(sign)), 3)}
out["full_value_spread_within_target_median"] = round(float(np.median([full[ix].max() - full[ix].min() for ix in by_t.values() if len(ix) >= 5])), 3)
out["seconds"] = round(time.time() - T0)
Path(Path(__file__).resolve().parent / "logs").mkdir(exist_ok=True)
(Path(__file__).resolve().parent / "logs" / "trade_proxy_study.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
