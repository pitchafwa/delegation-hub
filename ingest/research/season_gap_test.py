"""Season-vs-career gap test (2026-10-29, Tommy: Ty Jerome's keeper trajectory starts ~16 below his season DELCO after a clear role change).
Does the keeper trajectory (asset_value_v2.py unified path: veterans E_h = w*ridge_h + (1-w)*Kalman_h, w = 0.75 for h=1, 0.5 after) get more accurate
if it moves toward season-mode DELCO when that is above career-mode DELCO at the end of the base season?

PART 1, held-out keeper paths: engine_blend_test.py's held-out rows (data/_engine_blend_rows.pkl: origins 2016-24, horizons 1-4, both engines rebuilt
with only data known at the origin; players with 20+ games and 10+ mpg who rate as fantasy-relevant) plus season and career DELCO at the end of the
origin season (data/delco_pregame_all.csv.gz). Candidates: E + b_h * gap+ (gap = season - career, positive part only), E + b_h * gap (both ways),
and E + b_h * gap+ capped. b_h fitted on origins 2016-2019, graded on origins 2020-2024 (and the reverse, as a check).
PART 2, the same question for players PART 1 cannot see (fewer than 20 games in the base season, like Jerome's 15): next-season actual vs season and
career DELCO at the end of the base season, by gap.
PART 3, in-season: after a player's 5th game, which predicts the rest of that season.
Run from ingest/:  .venv/Scripts/python research/season_gap_test.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
rm = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))

k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "SEASON", "after", "career"]).sort_values("GAME_DATE")
end = k.groupby(["PLAYER_ID", "SEASON"]).agg(S=("after", "last"), C=("career", "last"), n=("after", "size")).reset_index()
end["origin"] = end["SEASON"].str[:4].astype(int)

# ---------------- PART 1
E = pd.read_pickle(D / "_engine_blend_rows.pkl")
E = E.merge(end[["PLAYER_ID", "origin", "S", "C"]], on=["PLAYER_ID", "origin"], how="left").dropna(subset=["S", "C"])
E["w"] = np.where(E["h"] == 1, 0.75, 0.5)
E["prod"] = E["w"] * E["B_ridge"] + (1 - E["w"]) * E["A_kalman"]
E["gap"] = E["S"] - E["C"]
E["gp_"] = E["gap"].clip(lower=0)
print(f"PART 1: held-out keeper paths, {len(E)} rows (origins {E.origin.min()}-{E.origin.max()}, horizons 1-4)")
print(f"  gap = season DELCO - career DELCO at the end of the base season: share >= +5: {(E.gap >= 5).mean():.0%}, <= -5: {(E.gap <= -5).mean():.0%}\n")


def fit_b(tr, col):
    x, y = tr[col].to_numpy(), (tr["y"] - tr["prod"]).to_numpy()
    return float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0


for lab, trm, tem in (("fit 2016-19, graded 2020-24", E.origin <= 2019, E.origin >= 2020), ("fit 2020-24, graded 2016-19", E.origin >= 2020, E.origin <= 2019)):
    print(f"  {lab}:")
    print(f"  {'h':>2} {'n':>5} {'today':>7} {'+gap+':>7} {'b':>5} {'+gap':>7} {'b':>5} | gap>=+5: {'n':>4} {'today':>7} {'+gap+':>7} | gap<=-5: {'n':>4} {'today':>7} {'+gap':>7}")
    for h in (1, 2, 3, 4):
        tr, te = E[trm & (E.h == h)], E[tem & (E.h == h)].copy()
        b1, b2 = fit_b(tr, "gp_"), fit_b(tr, "gap")
        te["c1"], te["c2"] = te["prod"] + b1 * te["gp_"], te["prod"] + b2 * te["gap"]
        up, dn = te[te.gap >= 5], te[te.gap <= -5]
        print(f"  {h:2d} {len(te):5d} {rm(te.y, te['prod']):7.2f} {rm(te.y, te.c1):7.2f} {b1:5.2f} {rm(te.y, te.c2):7.2f} {b2:5.2f} |"
              f"          {len(up):4d} {rm(up.y, up['prod']):7.2f} {rm(up.y, up.c1):7.2f} |          {len(dn):4d} {rm(dn.y, dn['prod']):7.2f} {rm(dn.y, dn.c2):7.2f}")
print("\n  bias (actual - today's path) by gap, all held-out rows:")
E["gb"] = pd.cut(E["gap"], [-99, -5, -2, 2, 5, 10, 99])
print(E.groupby(["gb", "h"], observed=True).apply(lambda t: pd.Series({"n": len(t), "bias": (t.y - t["prod"]).mean()})).unstack("h").round(1).to_string())

# ---------------- PART 2 + 3: actual fantasy points from box scores
g = pd.read_csv(D / "game_logs_unified.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA"])
g["fp"] = g.PTS + 1.5 * g.REB + 2 * g.AST + 3 * g.STL + 3 * g.BLK + g.FG3M + 2 * g.FTM - g.FTA - g.TOV
g = g[g.MIN > 0]
seasons = sorted(k.SEASON.unique())
nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
act = g.groupby(["PLAYER_ID", "SEASON"]).agg(y=("fp", "mean"), gp=("fp", "size")).reset_index()
e2 = end.assign(S2=end.SEASON.map(nxt)).merge(act.rename(columns={"SEASON": "S2"}), on=["PLAYER_ID", "S2"]).dropna()
e2 = e2[(e2.gp >= 15) & (e2.S2 < "2026-27") & (e2.C >= 15)]
e2["gap"] = e2.S - e2.C
print("\nPART 2: next season vs season / career DELCO at the end of the base season (career 15+, 15+ games next season)")
for lab, m in (("base season under 20 games", e2.n < 20), ("base season 20+ games", e2.n >= 20)):
    print(f"  {lab}:")
    for lo, hi in ((-99, -5), (-5, 2), (2, 5), (5, 10), (10, 99)):
        s = e2[m & (e2.gap >= lo) & (e2.gap < hi)]
        if len(s) < 10:
            print(f"    gap {lo:>3} to {hi:<3} n={len(s)}")
            continue
        w = ((s.y - s.C) * (s.S - s.C)).sum() / ((s.S - s.C) ** 2).sum()
        print(f"    gap {lo:>3} to {hi:<3} n={len(s):5d}  actual minus career {(s.y - s.C).mean():+5.1f}  minus season {(s.y - s.S).mean():+5.1f}  best weight on season {w:.2f}")

rows = []
for (pid, s), sub in k[k.SEASON < "2026-27"].groupby(["PLAYER_ID", "SEASON"]):
    if len(sub) >= 25:
        a = sub.iloc[4]
        rows.append((pid, s, a.GAME_DATE, a.after, a.career))
r = pd.DataFrame(rows, columns=["PLAYER_ID", "SEASON", "d5", "S", "C"])
gg = g.merge(r, on=["PLAYER_ID", "SEASON"])
gg = gg[gg.GAME_DATE > gg.d5].groupby(["PLAYER_ID", "SEASON"]).fp.agg(["mean", "size"]).reset_index().rename(columns={"mean": "y", "size": "n"})
d3 = r.merge(gg, on=["PLAYER_ID", "SEASON"])
d3 = d3[(d3.n >= 15) & (d3.C >= 15)]
d3["gap"] = d3.S - d3.C
print("\nPART 3: after a player's 5th game, the rest of that season (career 15+)")
for lo, hi in ((-99, -5), (-5, 2), (2, 5), (5, 10), (10, 99)):
    s = d3[(d3.gap >= lo) & (d3.gap < hi)]
    if len(s) < 10:
        print(f"  gap {lo:>3} to {hi:<3} n={len(s)}")
        continue
    w = ((s.y - s.C) * (s.S - s.C)).sum() / ((s.S - s.C) ** 2).sum()
    print(f"  gap {lo:>3} to {hi:<3} n={len(s):5d}  actual minus career {(s.y - s.C).mean():+5.1f}  minus season {(s.y - s.S).mean():+5.1f}  best weight on season {w:.2f}")

# ---------------- PART 4 (in-season): the keeper path's number for a season (made before it, held-out rows h=1 / h=2) vs season DELCO after N games of
# that season. Does this season's DELCO beat the path for the REST of this season (h=1), and does it improve NEXT season (h=2)?
print("\nPART 4: in-season. Path = the keeper path's number made before the season; season = season DELCO after N games of it")
kk = k.copy()
kk["origin"] = kk["SEASON"].str[:4].astype(int)
kk["i"] = kk.groupby(["PLAYER_ID", "SEASON"]).cumcount() + 1
fp_ = g.copy()
fp_["origin"] = fp_["SEASON"].str[:4].astype(int)
for N in (5, 10, 20):
    at = kk[kk.i == N][["PLAYER_ID", "origin", "GAME_DATE", "after", "career"]].rename(columns={"origin": "tgt", "GAME_DATE": "dN", "after": "SN", "career": "CN"})
    for h in (1, 2):
        t = E[E.h == h].copy()
        t["tgt"] = t["origin"] + 1                       # the season the in-season games come from (the one after the origin)
        t = t.merge(at, on=["PLAYER_ID", "tgt"])
        if h == 1:                                       # rest of that season after game N
            rest = fp_.merge(at[["PLAYER_ID", "tgt", "dN"]].rename(columns={"tgt": "origin"}), on=["PLAYER_ID", "origin"])
            rest = rest[rest.GAME_DATE > rest.dN].groupby(["PLAYER_ID", "origin"]).fp.agg(["mean", "size"]).reset_index().rename(columns={"origin": "tgt", "mean": "y2", "size": "n2"})
            t = t.merge(rest, on=["PLAYER_ID", "tgt"])
            t = t[t.n2 >= 10]
            t["y"] = t["y2"]
        t["gp2"] = t["SN"] - t["prod"]
        best = None
        for wv in np.arange(0, 1.01, 0.1):
            e_ = rm(t.y, t["prod"] + wv * t.gp2)
            best = (e_, wv) if best is None or e_ < best[0] else best
        up = t[t.gp2 >= 8]
        wu = ((up.y - up["prod"]) * up.gp2).sum() / (up.gp2 ** 2).sum() if len(up) else float("nan")
        print(f"  after game {N:2d}, {'rest of this season' if h == 1 else 'NEXT season       '}: n={len(t):4d}  path rmse {rm(t.y, t['prod']):5.2f}  season DELCO {rm(t.y, t.SN):5.2f}"
              f"  best mix {best[0]:5.2f} at {best[1]:.1f} on season | season 8+ above path: n={len(up):3d} path bias {(up.y - up['prod']).mean():+5.1f}, best weight {wu:.2f}")

# ---------------- PART 5: the same blend, both directions and three seasons out (path + w * (season DELCO - path))
print("\nPART 5: best weight on season DELCO by direction and horizon (h=1 rest of this season, h=2 next season, h=3 the one after)")
for N in tuple(int(x) for x in os.environ.get("GAP_N", "5,20").split(",")):
    at = kk[kk.i == N][["PLAYER_ID", "origin", "GAME_DATE", "after"]].rename(columns={"origin": "tgt", "GAME_DATE": "dN", "after": "SN"})
    for h in (1, 2, 3):
        t = E[E.h == h].copy()
        t["tgt"] = t["origin"] + 1
        t = t.merge(at, on=["PLAYER_ID", "tgt"])
        if h == 1:
            rest = fp_.merge(at[["PLAYER_ID", "tgt", "dN"]].rename(columns={"tgt": "origin"}), on=["PLAYER_ID", "origin"])
            rest = rest[rest.GAME_DATE > rest.dN].groupby(["PLAYER_ID", "origin"]).fp.agg(["mean", "size"]).reset_index().rename(columns={"origin": "tgt", "mean": "y2", "size": "n2"})
            t = t.merge(rest, on=["PLAYER_ID", "tgt"])
            t = t[t.n2 >= 10]
            t["y"] = t["y2"]
        t["d"] = t["SN"] - t["prod"]
        bw = lambda s: ((s.y - s["prod"]) * s.d).sum() / (s.d ** 2).sum() if len(s) else float("nan")
        up, dn, mid = t[t.d >= 5], t[t.d <= -5], t[t.d.abs() < 5]
        ws = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
        errs = [rm(t.y, t["prod"] + w * t.d) for w in ws]
        print(f"  game {N:2d} h={h}: n={len(t):4d} best w {ws[int(np.argmin(errs))]} rmse {rm(t.y, t['prod']):.2f} -> {min(errs):.2f} | 5+ above path: n={len(up):3d} w {bw(up):.2f}"
              f" | 5+ below: n={len(dn):3d} w {bw(dn):.2f} | within 5: w {bw(mid):.2f}")
