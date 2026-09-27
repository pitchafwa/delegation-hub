"""Does DELCO account for a player changing teams/situations, and does it matter? (2026-09-27, at Tommy's request)

DELCO's core trajectory (kalman_vor.py's build_trajectory) is a pure per-player, per-stat time series: each stat's Kalman posterior advances only by
age (annual_slope(stat, age)). No team, role, or situation feature enters it anywhere -- a trade only shows up lagging, once real games are observed
with the new team. The only team-situation feature anywhere in this project is for INCOMING ROOKIES (team_context.py's "available usage" measure,
built 2026-09-24) -- a different problem (a rookie's first team) from a veteran changing teams mid-career, which is untouched.
(The now-superseded v1 valuation model, before the Kalman rebuild, DID have a team-change feature in its regression -- VALUATION_RESEARCH.md's
"known limitations" note about it belongs to that retired model, not DELCO.)

This tests whether that actually costs accuracy: for real player-seasons 2010-2024 (rotation players, 20+ mpg, 40+ GP in both a season and the
next), does simple persistence (what DELCO's per-stat tracking effectively reduces to, age drift aside) predict next season's rate (fantasy points
per minute) worse for players who changed teams than for those who didn't?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F

sys.stdout.reconfigure(encoding="utf-8")
g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
g = g[g.season != "2025-26"].copy()
g["fp"] = F.fp(g)
per = g.groupby(["pid", "season"]).agg(min_sum=("min", "sum"), fp_sum=("fp", "sum"), gp=("min", "size"), mpg=("min", "mean"),
                                        team_last=("team", "last"), team_first=("team", "first")).reset_index()
per = per[(per.gp >= 40) & (per.mpg >= 20)].copy()
per["rate"] = per.fp_sum / per.min_sum
seasons = sorted(per.season.unique())
nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
idx = per.set_index(["pid", "season"])

rows = []
for (pid, season), r in idx.iterrows():
    if season not in nxt or (pid, nxt[season]) not in idx.index:
        continue
    r1 = idx.loc[(pid, nxt[season])]
    rows.append(dict(pid=pid, season=season, rate0=r.rate, rate1=r1.rate, mpg0=r.mpg, mpg1=r1.mpg, changed=int(r.team_last != r1.team_first)))
P = pd.DataFrame(rows)
print(f"{len(P)} rotation-to-rotation player-season pairs, {P.changed.sum()} team changes ({P.changed.mean():.1%})")

print("\n=== persistence-only prediction error, by whether he changed teams ===")
P["err"] = P.rate1 - P.rate0
for grp, sub in P.groupby("changed"):
    print(f"  changed={grp}  n={len(sub):4d}  RMSE {np.sqrt((sub.err**2).mean()):.4f}  mean signed err {sub.err.mean():+.4f}")

print("\n=== does his role/minutes shift more after a team change? ===")
P["dmpg"] = P.mpg1 - P.mpg0
for grp, sub in P.groupby("changed"):
    print(f"  changed={grp}  mean |minutes change| {sub.dmpg.abs().mean():.2f} mpg")

print("\n=== residual after controlling for his own prior rate (isolates the team-change effect from 'he was just declining anyway') ===")
mu, sd = P[["rate0"]].mean(), P[["rate0"]].std()
m = Ridge(alpha=1.0).fit((P[["rate0"]] - mu) / sd, P.rate1)
P["res"] = P.rate1 - m.predict((P[["rate0"]] - mu) / sd)
for grp, sub in P.groupby("changed"):
    print(f"  changed={grp}  n={len(sub):4d}  RMSE {np.sqrt((sub.res**2).mean()):.4f}  mean residual {sub.res.mean():+.4f}")
