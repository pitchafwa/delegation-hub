"""Study 4: how much of a missing rotation player's production do his teammates pick up, and who gets it?

Uses data/usage_flow_panel.pkl (usage_flow_panel.py).  Within each teammate-season we compare games with more vs less production vacated by absent rotation teammates
(player-season fixed effects, so a teammate's own level does not matter):
    fp_jg - mean_js(fp) = kappa * x_jg + e,     x_jg = V_g * share_jg
    V_g     = fantasy points per game (previous-15 mean) of the rotation players absent in game g
    share_jg = teammate j's share of the ACTIVE team's production (his season mean ^ gamma / the sum over players who played)
kappa = the fraction of the vacated production that shows up in the teammate line.  gamma > 1 means the best remaining players capture more than their proportional share.
Then an out-of-sample forecast test (fit on 2010-11..2019-20, test 2020-21..2023-24): predict a teammate's fantasy points with and without the absence adjustment.
Run from ingest/:  uv run python research/usage_flow_study.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
t = pd.read_pickle(D / "usage_flow_panel.pkl")
t = t[(t.mp15 >= 6) & (t["min"] > 0)].copy()
t["season"] = t.season_year
sm = t.groupby(["personId", "season"]).fp.transform("mean")
t["bs"] = sm
mn = t.groupby(["personId", "season"])["min"].transform("mean")
t["bs_min"] = mn
print(f"{len(t):,} teammate-games; {t.personId.nunique():,} players")


def shares(gamma):
    a = t.bs.clip(lower=0.5) ** gamma
    return a / a.groupby([t.gameId, t.teamTricode]).transform("sum")


def demean(x, keys):
    return x - x.groupby(keys).transform("mean")


keys = [t.personId, t.season]
y_dm = demean(t.fp, keys)
print("\nA. capture of vacated fantasy production (player-season fixed effects); kappa = share of the vacated points that reaches a teammate, by concentration gamma")
for gamma in (0.0, 0.5, 1.0, 1.5, 2.0):
    x = t.V_fp * shares(gamma)
    x_dm = demean(x, keys)
    kappa = (x_dm * y_dm).sum() / (x_dm ** 2).sum()
    resid = y_dm - kappa * x_dm
    r2 = 1 - (resid ** 2).sum() / (y_dm ** 2).sum()
    print(f"   gamma {gamma:3.1f}: kappa {kappa:.3f}  R2 of within-player variation {r2:.4f}")
best_gamma = 1.0
for gamma in (1.0,):
    pass

print("\nB. minutes: share of vacated minutes that reaches a teammate, by teammate role (season minutes tier)")
tiers = pd.cut(t.bs_min, [0, 15, 22, 30, 60], labels=["<15 mpg", "15-22", "22-30", "30+"])
for gamma in (1.0,):
    xs = t.V_min * (t.bs_min.clip(lower=1) ** gamma / (t.bs_min.clip(lower=1) ** gamma).groupby([t.gameId, t.teamTricode]).transform("sum"))
    xdm = demean(xs, keys)
    ymin = demean(t["min"], keys)
    print(f"   all: kappa_min {(xdm * ymin).sum() / (xdm ** 2).sum():.3f}")
    for tier in tiers.cat.categories:
        m = tiers == tier
        print(f"   {tier:8s}: kappa_min {(xdm[m] * ymin[m]).sum() / (xdm[m] ** 2).sum():.3f} (n={m.sum():,})")

print("\nC. fantasy points by size of the absent player (largest absentee's previous-15 minutes) and teammate tier: mean change vs his own season average")
big = t.merge(pd.DataFrame(), how="left") if False else t
ab = pd.read_pickle(D / "usage_flow_absences.pkl")
mx = ab.groupby(["gameId", "teamTricode"]).agg(max_mp=("mp15", "max"), max_fp=("fp15", "max")).reset_index()
t = t.merge(mx, on=["gameId", "teamTricode"], how="left")
t["dfp"] = t.fp - t.bs
t["absent_cat"] = pd.cut(t.max_mp.fillna(0), [-1, 0.1, 20, 28, 33, 60], labels=["none", "12-20 mpg", "20-28", "28-33", "33+ (star)"])
t["role"] = pd.cut(t.bs, [-1, 12, 20, 30, 100], labels=["<12 fp/g", "12-20", "20-30", "30+ fp/g"])
piv = t.groupby(["absent_cat", "role"], observed=True).dfp.agg(["mean", "size"]).round(2)
print(piv["mean"].unstack().to_string())
print("n:"); print(piv["size"].unstack().to_string())

print("\nD. out-of-sample forecast test: fit kappa on 2010-11..2019-20, predict 2020-21..2023-24 teammate games")
tr = t.season <= "2019-20"
te = ~tr
gamma = 1.0
t["x"] = t.V_fp * shares(gamma)
kappa = ((demean(t.x, keys)[tr]) * (y_dm[tr])).sum() / (demean(t.x, keys)[tr] ** 2).sum()
# baseline forecast: the teammate's trailing-30-game mean; adjusted forecast adds kappa x (x_now - mean of x over the same 30 games)
t = t.sort_values(["personId", "gd", "gameId"]).reset_index(drop=True)
g = t.groupby("personId")
t["tr_fp"] = g.fp.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
t["tr_x"] = g.x.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
t["pred0"] = t.tr_fp
t["pred1"] = t.tr_fp + kappa * (t.x - t.tr_x)
te = (t.season > "2019-20") & t.tr_fp.notna()
for name, m in (("all teammate games", te), ("games with a rotation teammate absent", te & (t.V_min > 0)), ("a 28+ mpg player absent", te & (t.max_mp >= 28)), ("a 33+ mpg star absent", te & (t.max_mp >= 33))):
    e0, e1 = (t.fp - t.pred0)[m].abs(), (t.fp - t.pred1)[m].abs()
    print(f"   {name:40s} n={m.sum():6,}  MAE baseline {e0.mean():.3f} -> adjusted {e1.mean():.3f}  ({(1 - e1.mean() / e0.mean()) * 100:+.1f}%)   bias baseline {(t.pred0 - t.fp)[m].mean():+.2f} adjusted {(t.pred1 - t.fp)[m].mean():+.2f}")
print(f"   kappa used {kappa:.3f}")
t.to_pickle(D / "usage_flow_panel2.pkl")
