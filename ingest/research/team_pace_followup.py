"""Pace follow-up (2026-09-27, at Tommy's request): team-to-team pace variation (not year-to-year league drift), its persistence, and whether
converting a team-changer's rate through his old/new team's actual pace (instead of assuming a flat per-minute rate) helps predict his production
after a move. See RESEARCH_gamelevel.md's "Pace follow-up" section for the numbers and interpretation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F

sys.stdout.reconfigure(encoding="utf-8")
tb = pd.read_pickle(F.D.parent / "gamelevel" / "team_box_features.pkl")
from team_abbr import canon
tb["team"] = tb.TEAM_ABBREVIATION.map(canon)

season_pace = tb.groupby(["team", "season"]).poss.mean().reset_index()
print("=== cross-team spread of full-season pace, by season ===")
for s, sub in season_pace.groupby("season"):
    print(f"  {s}: min {sub.poss.min():.1f}  mean {sub.poss.mean():.1f}  max {sub.poss.max():.1f}  sd {sub.poss.std():.2f}  spread {sub.poss.max()-sub.poss.min():.1f}")

print("\n=== year-to-year stickiness ===")
seasons = sorted(season_pace.season.unique())
sp = season_pace.pivot(index="team", columns="season", values="poss")
for i in range(1, len(seasons)):
    a, b = seasons[i - 1], seasons[i]
    sub = sp[[a, b]].dropna()
    print(f"  {a} -> {b}: r={sub[a].corr(sub[b]):+.3f}")

print("\n=== within-season stability ===")
tb2 = tb.sort_values(["team", "season", "GAME_DATE"]).copy()
tb2["g_idx"] = tb2.groupby(["team", "season"]).cumcount()
for N in (5, 10, 15, 20):
    first = tb2[tb2.g_idx < N].groupby(["team", "season"]).poss.mean().rename("first")
    rest = tb2[tb2.g_idx >= N].groupby(["team", "season"]).poss.mean().rename("rest")
    both = pd.concat([first, rest], axis=1).dropna()
    print(f"  first {N:2d} games vs rest of season: r={both['first'].corr(both['rest']):+.3f}")

# ---------------- does pace-adjusting a team-changer's rate help predict his production after the move? ----------------
g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
g = g[g.season != "2025-26"].copy()
g["fp"] = F.fp(g)
per = g.groupby(["pid", "season"]).agg(min_sum=("min", "sum"), fp_sum=("fp", "sum"), gp=("min", "size"), mpg=("min", "mean"),
                                        team_last=("team", "last"), team_first=("team", "first")).reset_index()
per = per[(per.gp >= 40) & (per.mpg >= 20)].copy()
per["rate"] = per.fp_sum / per.min_sum
seasons2 = sorted(per.season.unique())
nxt = {s: seasons2[i + 1] for i, s in enumerate(seasons2[:-1])}
idx = per.set_index(["pid", "season"])
team_pace = tb.groupby(["team", "season"]).poss.mean().to_dict()

rows = []
for (pid, season), r in idx.iterrows():
    if season not in nxt or (pid, nxt[season]) not in idx.index:
        continue
    r1 = idx.loc[(pid, nxt[season])]
    changed = int(r.team_last != r1.team_first)
    p0, p1 = team_pace.get((r.team_last, season)), team_pace.get((r1.team_first, nxt[season]))
    if p0 is None or p1 is None:
        continue
    rows.append(dict(pid=pid, season=season, rate0=r.rate, rate1=r1.rate, changed=changed, pace0=p0, pace1=p1))
P = pd.DataFrame(rows)
P["pred_flat"] = P.rate0
P["pred_pace"] = P.rate0 * (P.pace1 / P.pace0)
print("\n=== flat persistence vs pace-adjusted prediction of next season's rate ===")
for grp, sub in P.groupby("changed"):
    e_flat, e_pace = sub.rate1 - sub.pred_flat, sub.rate1 - sub.pred_pace
    print(f"  changed={grp}  n={len(sub)}  flat RMSE {np.sqrt((e_flat**2).mean()):.4f}  pace-adjusted RMSE {np.sqrt((e_pace**2).mean()):.4f}")
ch = P[P.changed == 1]
print(f"\n  pace ratio (pace1/pace0) for changers: mean {(ch.pace1/ch.pace0).mean():.3f}  sd {(ch.pace1/ch.pace0).std():.3f}")
print(f"  correlation(pace ratio, actual rate ratio) for changers: {(ch.rate1/ch.rate0).corr(ch.pace1/ch.pace0):+.3f}")
