"""DELCO riser study (Tommy, 2026-11-25). Open finding from trade_tag_study.py: players averaging 5+ over season DELCO kept beating it by +3.5 to +6.5 over their next
20 games. Is season DELCO (the "this season" number behind long_level, trade values, keeper values this season, the season forecast) too slow, and what fixes it best?

Checkpoints: after game n (n = 3, 5, 8, 12, 16, 20, 25, 30, 40, 50) of each player-season 2011-12..2025-26 (DELCO opening at 15+, 10+ games after the checkpoint).
  cur = season DELCO after game n; sofar = fantasy points per game so far; l10 = last up-to-10 games; mins = minutes so far / last 5 vs DELCO's expected minutes.
Targets: next 30 games' average (10+ needed) and the rest-of-season average.
Candidates (fit 2011-12..2020-21, judged on 2021-22..2025-26, never seen in fitting):
  F0 DELCO as is      F1 + constant      F2 + a + b x (so far - DELCO) by stage      F3 F2 + recent form (last 10 vs so far) + minutes trend      F4 F3 split by gap sign
Run from ingest/:  .venv/Scripts/python research/delco_riser_study.py   (writes data/delco_riser_study.log, data/delco_riser_model.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import form_common as FC  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
OUT = open(D / "delco_riser_study.log", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")


g = FC.load_all()
_td = ((g.pts >= 10).astype(int) + (g.reb >= 10) + (g.ast >= 10) + (g.stl >= 10) + (g.blk >= 10)) >= 3
g = g.assign(fp=g.fp - 3.0 * _td.astype(float))[["pid", "date", "fp"]].copy()      # DELCO's number leaves out the triple-double bonus (added separately); so does this
g["date"] = pd.to_datetime(g["date"]).dt.normalize()
g = g.groupby(["pid", "date"], as_index=False).fp.max()
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "SEASON", "MIN", "pre", "after", "afterMIN", "preMIN"], low_memory=False)
k["date"] = pd.to_datetime(k.GAME_DATE).dt.normalize()
k = k[k.MIN > 0]
m = k.merge(g, left_on=["PLAYER_ID", "date"], right_on=["pid", "date"]).sort_values(["PLAYER_ID", "date"])
del g, k

CHECKS = (3, 5, 8, 12, 16, 20, 25, 30, 40, 50)
rows = []
for (pid, season), x in m.groupby(["PLAYER_ID", "SEASON"], sort=False):
    if season < "2011-12" or season > "2025-26" or len(x) < 13:
        continue
    fp_, after, pre, mins, amin = x.fp.to_numpy(float), x["after"].to_numpy(float), x["pre"].to_numpy(float), x.MIN.to_numpy(float), x.afterMIN.to_numpy(float)
    if np.isnan(pre[0]) or pre[0] < 15:
        continue
    N = len(fp_)
    for n in CHECKS:
        if N - n < 10 or np.isnan(after[n - 1]):
            continue
        rows.append({"pid": pid, "season": season, "n": n, "open": pre[0], "cur": after[n - 1], "sofar": fp_[:n].mean(), "l10": fp_[max(0, n - 10):n].mean(),
                     "l5": fp_[max(0, n - 5):n].mean(), "min_sofar": mins[:n].mean(), "min_l5": mins[max(0, n - 5):n].mean(), "amin": amin[n - 1],
                     "f30": fp_[n:n + 30].mean(), "ros": fp_[n:].mean(), "nfut": N - n})
P = pd.DataFrame(rows)
P["gap"] = P.sofar - P.cur
P["trend"] = P.l10 - P.sofar
P["mgap"] = P.min_sofar - P.amin
P["mtrend"] = P.min_l5 - P.min_sofar
P["stage"] = pd.cut(P.n, [0, 6, 14, 24, 99], labels=["3-5", "8-12", "16-20", "25-50"])
P["pos"] = (P.gap > 0).astype(float)
TR = P[P.season <= "2020-21"].copy()
TE = P[P.season >= "2021-22"].copy()
say(f"checkpoints {len(P):,} (fit {len(TR):,}, test {len(TE):,}), player-seasons {P.groupby(['pid', 'season']).ngroups:,}\n")

say("A. Is DELCO biased? next 30 minus DELCO by season (all checkpoints)")
for s, x in P.groupby("season"):
    say(f"   {s}  n={len(x):5d}  mean {np.mean(x.f30 - x.cur):+.2f}   rest of season {np.mean(x.ros - x.cur):+.2f}")
say("")
say("B. Next 30 minus DELCO by gap (so far minus DELCO) and stage, all seasons")
for st, xs in P.groupby("stage", observed=True):
    out = []
    for lo, hi in ((-99, -5), (-5, -2), (-2, 2), (2, 5), (5, 99)):
        x = xs[(xs.gap >= lo) & (xs.gap < hi)]
        out.append(f"[{lo:+d},{hi:+d}) {np.mean(x.f30 - x.cur):+.2f} (n={len(x)})")
    say(f"   games {st:6s} " + "  ".join(out))
say("")


def design(df, kind, stage):
    cols = [np.ones(len(df))]
    if kind >= 2:
        cols.append(df.gap.to_numpy())
    if kind >= 3:
        cols += [df.trend.to_numpy(), df.mgap.to_numpy(), df.mtrend.to_numpy()]
    if kind >= 4:
        cols += [df.gap.to_numpy() * df.pos.to_numpy()]
    return np.column_stack(cols)


def fit(df, kind, target, lam=50.0):
    co = {}
    for st, x in df.groupby("stage", observed=True):
        X = design(x, kind, st)
        y = (x[target] - x.cur).to_numpy()
        A = X.T @ X + lam * np.diag([0] + [1] * (X.shape[1] - 1))
        co[str(st)] = np.linalg.solve(A, X.T @ y)
    return co


def predict(df, kind, co):
    out = df.cur.to_numpy().copy()
    if kind == 0:
        return out
    for st, x in df.groupby("stage", observed=True):
        idx = df.index.get_indexer(x.index)
        out[idx] += design(x, kind, st) @ co[str(st)]
    return out


def errs(df, pred, target):
    e = df[target].to_numpy() - pred
    w = np.minimum(df.nfut.to_numpy(), 30) if target == "f30" else df.nfut.to_numpy()
    return np.sqrt(np.average(e ** 2, weights=w)), np.average(np.abs(e), weights=w), np.average(e, weights=w)


say("C. Out of sample (test 2021-22..2025-26): error of each candidate (RMSE / MAE / mean miss), weighted by games")
names = {0: "F0 DELCO as is", 1: "F1 + constant", 2: "F2 + gap", 3: "F3 + gap, form, minutes", 4: "F4 F3 split by gap sign"}
res = {}
for target in ("f30", "ros"):
    say(f"  target: {'next 30 games' if target == 'f30' else 'rest of season'}")
    for kind in range(5):
        co = fit(TR, kind, target) if kind else None
        pr = predict(TE, kind, co)
        r, a, b = errs(TE, pr, target)
        sub = []
        for lab, mask in (("risers 5+", TE.gap >= 5), ("slumpers 5+", TE.gap <= -5), ("rest", TE.gap.abs() < 5)):
            rr, aa, bb = errs(TE[mask], pr[mask.to_numpy()], target)
            sub.append(f"{lab}: {aa:.2f} ({bb:+.2f})")
        say(f"    {names[kind]:26s} RMSE {r:.3f}  MAE {a:.3f}  bias {b:+.2f}   |  " + "  ".join(sub))
        res[(target, kind)] = (r, a, co)
    say("")

say("D. By stage, next 30 games MAE (test): F0 vs F2 vs F3")
co2, co3 = fit(TR, 2, "f30"), fit(TR, 3, "f30")
p0, p2, p3 = predict(TE, 0, None), predict(TE, 2, co2), predict(TE, 3, co3)
for st, x in TE.groupby("stage", observed=True):
    mk = (TE.stage == st).to_numpy()
    say(f"   games {st:6s} n={mk.sum():5d}   F0 {errs(x, p0[mk], 'f30')[1]:.3f}   F2 {errs(x, p2[mk], 'f30')[1]:.3f}   F3 {errs(x, p3[mk], 'f30')[1]:.3f}")
say("")

# refit on all seasons for production, both kinds, so the chosen one can be written out
final = {}
for kind in (2, 3):
    co = fit(P, kind, "f30")
    final[f"F{kind}"] = {st: [round(float(v), 4) for v in c] for st, c in co.items()}
say("coefficients refit on 2011-26 (next 30 games; columns: const, gap[, trend, mgap, mtrend]):")
for kk, v in final.items():
    for st, c in v.items():
        say(f"   {kk} games {st:6s} {c}")
json.dump({"note": "delco_riser_study.py: season DELCO + correction by stage (games played this season). See data/delco_riser_study.log.", "stages": [[0, 6], [7, 14], [15, 24], [25, 999]],
           "models": final}, open(D / "delco_riser_model.json", "w"), indent=1)
say("\nwrote data/delco_riser_model.json")

# E. The constant: DELCO's overall low bias fell from about +2 (2011-21) to +0.2 (2025-26). A constant fit on old seasons overshoots now. Compare, out of sample:
#    F3 as fit (constant from 2011-21), F3n (no constant: gap/form/minutes only), F3r (constant = last 2 completed seasons' F3n miss, rolling), per test season.
say("\nE. The constant, per test season (next 30 games: MAE / mean miss)")


def fit_nc(df, target, lam=50.0):
    co = {}
    for st, x in df.groupby("stage", observed=True):
        X = design(x, 3, st)[:, 1:]
        y = (x[target] - x.cur).to_numpy()
        co[str(st)] = np.r_[0.0, np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)]
    return co


coN = fit_nc(TR, "f30")
pN = predict(TE, 3, coN)
pN_all = predict(P, 3, coN)
for s in sorted(TE.season.unique()):
    mk = (TE.season == s).to_numpy()
    x = TE[mk]
    prev = [q for q in sorted(P.season.unique()) if q < s][-2:]
    pm = P.season.isin(prev).to_numpy()
    c_roll = float(np.mean(P[pm].f30.to_numpy() - pN_all[pm]))
    say(f"   {s}   F0 {errs(x, p0[mk], 'f30')[1]:.3f} ({errs(x, p0[mk], 'f30')[2]:+.2f})   F3 {errs(x, p3[mk], 'f30')[1]:.3f} ({errs(x, p3[mk], 'f30')[2]:+.2f})"
        f"   F3n {errs(x, pN[mk], 'f30')[1]:.3f} ({errs(x, pN[mk], 'f30')[2]:+.2f})   F3r {errs(x, pN[mk] + c_roll, 'f30')[1]:.3f} ({errs(x, pN[mk] + c_roll, 'f30')[2]:+.2f}, const {c_roll:+.2f})")
r_, a_, b_ = errs(TE, pN, "f30")
say(f"   all test: F3n MAE {a_:.3f} bias {b_:+.2f}; risers 5+ {errs(TE[TE.gap >= 5], pN[(TE.gap >= 5).to_numpy()], 'f30')[1]:.2f}; slumpers 5+ {errs(TE[TE.gap <= -5], pN[(TE.gap <= -5).to_numpy()], 'f30')[1]:.2f}")
coNA = fit_nc(P, "f30")
json.dump({"note": "delco_riser_study.py: season DELCO + correction by stage (games played this season). See data/delco_riser_study.log.", "stages": [[0, 6], [7, 14], [15, 24], [25, 999]],
           "models": {**final, "F3_train": {st: [round(float(v), 4) for v in c] for st, c in fit(TR, 3, "f30").items()}, "F3n": {st: [round(float(v), 4) for v in c] for st, c in coNA.items()}}}, open(D / "delco_riser_model.json", "w"), indent=1)
say("F3n refit on 2011-26: " + "; ".join(f"{st} {[round(float(v), 3) for v in c]}" for st, c in coNA.items()))
