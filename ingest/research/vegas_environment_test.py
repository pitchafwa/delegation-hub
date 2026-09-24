"""Does the betting market add information about a player's fantasy output BEYOND his own recent form?

For every player-game (players who actually played) we predict league-scored fantasy points from a rolling baseline
(his previous 20 games) and ask whether pre-game Vegas information -- team implied total, game total (pace proxy), size of
the spread (blowout risk), favourite status -- plus rest/home explains the RESIDUAL. Fit on earlier seasons, scored on the
latest, so the answer is out-of-sample. Conditional on playing: injuries/DNPs are a separate question (games missed).

League formula: PTS + 1.5 REB + 2 AST + 3 STL + 3 BLK + FG3M + 2 FTM - FTA - TOV + 3 TD3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
FIX = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}

lg = pd.read_csv(R / "data" / "game_logs_unified.csv", usecols=["PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_DATE", "TEAM", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
lg["fp"] = lg.PTS + 1.5 * lg.REB + 2 * lg.AST + 3 * lg.STL + 3 * lg.BLK + lg.FG3M + 2 * lg.FTM - lg.FTA - lg.TOV + 3 * lg.TD3
lg["date"] = pd.to_datetime(lg.GAME_DATE)
od = pd.read_csv(R / "data" / "game_odds.csv")
od["home"] = od.home.replace(FIX)
od["away"] = od.away.replace(FIX)
od["date"] = pd.to_datetime(od.date)
seasons = {int(s) for s in od.season.unique()}
lg = lg[lg.SEASON.map(lambda s: int(s[:4]) + 1 in seasons)].copy()

# rolling baselines (previous 20 games, shifted so the current game never leaks)
lg = lg.sort_values(["PLAYER_ID", "date"])
g = lg.groupby("PLAYER_ID")
lg["base_fp"] = g.fp.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
lg["base_min"] = g.MIN.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
lg["prev_date"] = g.date.shift(1)
lg["b2b"] = ((lg.date - lg.prev_date).dt.days == 1).astype(int)
lg["rest_days"] = (lg.date - lg.prev_date).dt.days.clip(upper=7)

# join odds from the player's team perspective
h = od.rename(columns={"home": "TEAM"}).assign(is_home=1, spread=lambda x: x.spread_home)
a = od.rename(columns={"away": "TEAM"}).assign(is_home=0, spread=lambda x: -x.spread_home)
tg = pd.concat([h[["date", "TEAM", "is_home", "spread", "total", "season"]], a[["date", "TEAM", "is_home", "spread", "total", "season"]]])
m = lg.merge(tg, on=["date", "TEAM"], how="inner")
m = m.dropna(subset=["base_fp", "base_min"])
m = m[(m.base_min >= 15)]
m["implied"] = m.total / 2 - m.spread / 2               # team's implied points (spread<0 = favourite)
m["fav"] = (m.spread < 0).astype(int)
m["abs_spread"] = m.spread.abs()
m["blowout"] = np.clip(m.abs_spread - 7, 0, None)         # extra spread beyond 7 (where the blowout risk starts to bite)
m["tot_dev"] = m.total - m.total.mean()
m["imp_dev"] = m.implied - m.implied.mean()
m["y"] = m.fp - m.base_fp
m["ymin"] = m.MIN - m.base_min
# role weighting: a starter (>=28 mpg baseline) vs a bench player
m["starter"] = (m.base_min >= 28).astype(int)
print(f"{len(m):,} player-games, seasons {sorted(m.season.unique())}, {m.PLAYER_ID.nunique()} players")

FEATS = {"tot_dev": "game total (pace/env), per point", "imp_dev": "team implied total, per point", "blowout": "spread beyond 7 (pts of extra spread)",
         "fav": "favourite", "is_home": "home", "b2b": "second night of a back-to-back"}


def fit(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, df.y.to_numpy(), rcond=None)
    return beta


def pred(df, cols, beta):
    return np.column_stack([np.ones(len(df))] + [df[c] for c in cols]) @ beta


last = max(m.season)
tr, te = m[m.season < last], m[m.season == last]
if len(tr) < 5000:
    cut = m.date.quantile(.6)
    tr, te = m[m.date <= cut], m[m.date > cut]
print(f"train {len(tr):,} (earlier seasons)  test {len(te):,} (season {last})")
mae0 = np.abs(te.y).mean()
rmse0 = np.sqrt((te.y ** 2).mean())
print(f"baseline only (rolling 20-game average): MAE {mae0:.2f}  RMSE {rmse0:.2f} fantasy pts/game")
for name, cols in [("game total only", ["tot_dev"]), ("implied total only", ["imp_dev"]), ("spread/blowout only", ["blowout", "fav"]),
                   ("rest/home only", ["b2b", "is_home"]), ("all Vegas (implied, total, blowout, fav)", ["imp_dev", "tot_dev", "blowout", "fav"]),
                   ("Vegas + rest/home", ["imp_dev", "tot_dev", "blowout", "fav", "b2b", "is_home"])]:
    b = fit(tr, cols)
    e = te.y - pred(te, cols, b)
    print(f"  + {name:42s} MAE {np.abs(e).mean():.2f}  RMSE {np.sqrt((e ** 2).mean()):.2f}  (RMSE gain {100 * (1 - np.sqrt((e ** 2).mean()) / rmse0):.2f}%)")

print("\n== effect sizes (all data), fantasy pts per game / minutes ==")
for grp, sub in [("everyone (>=15 mpg)", m), ("starters (>=28 mpg)", m[m.starter == 1]), ("bench/role (15-28 mpg)", m[m.starter == 0])]:
    cols = ["imp_dev", "tot_dev", "blowout", "b2b", "is_home"]
    b = fit(sub, cols)
    X = np.column_stack([np.ones(len(sub))] + [sub[c] for c in cols])
    ym = sub.ymin.to_numpy()
    bm, *_ = np.linalg.lstsq(X, ym, rcond=None)
    print(f"{grp:24s} n={len(sub):6,d}  fp: " + "  ".join(f"{c}={v:+.3f}" for c, v in zip(cols, b[1:])) + "   | minutes: " + "  ".join(f"{c}={v:+.3f}" for c, v in zip(cols, bm[1:])))

print("\n== average fantasy points vs own baseline, by spread bucket (favourite's players) ==")
m["bucket"] = pd.cut(m.abs_spread, [-0.1, 3, 6, 9, 12, 30], labels=["0-3", "3-6", "6-9", "9-12", "12+"])
t = m.assign(role=np.where(m.starter == 1, "starter", "role")).groupby(["role", "fav", "bucket"], observed=True).agg(n=("y", "size"), fp_vs_base=("y", "mean"), min_vs_base=("ymin", "mean")).round(2)
print(t.to_string())
print("\n== by game total bucket ==")
m["tb"] = pd.qcut(m.total, 5)
print(m.groupby("tb", observed=True).agg(n=("y", "size"), fp_vs_base=("y", "mean"), min_vs_base=("ymin", "mean")).round(2).to_string())
