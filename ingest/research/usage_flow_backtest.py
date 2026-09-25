"""Out-of-time test of usage flow on 2024-25 and 2025-26 (the model was fit on 2010-11..2023-24).
For every game with one or more rotation players officially listed OUT (5:30 PM report, injury/illness), forecast each teammate's fantasy points as
  baseline (recency-weighted mean of his previous games)   vs   baseline + usage-flow uplift
and score against what he actually scored.  Baselines use only games before that date; absences come from the pre-game report (known in advance).
Run from ingest/:  uv run python research/usage_flow_backtest.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import injury_common as C
import usage_flow as U
sys.stdout.reconfigure(encoding="utf-8")
D = C.D
g = pd.concat([pd.read_csv(D / "game_logs" / f"nba_api_{s}.csv") for s in ("2024-25", "2025-26")], ignore_index=True)
g["gd"] = pd.to_datetime(g.GAME_DATE)
cats = (g[["PTS", "REB", "AST", "STL", "BLK"]] >= 10).sum(axis=1)
g["fp"] = g.PTS + 1.5 * g.REB + 2 * g.AST + 3 * g.STL + 3 * g.BLK + g.FG3M + 2 * g.FTM - g.FTA - g.TOV + 3 * (cats >= 3)
g["key"] = g.PLAYER_NAME.map(C.key_of_log)
g["TEAM"] = g.TEAM_ABBREVIATION
g = g[g.MIN > 0].sort_values(["key", "gd"]).reset_index(drop=True)
# trailing (pre-game) quantities per player across both seasons
grp = g.groupby("key")
w = lambda s: s.shift(1).ewm(halflife=8, min_periods=5).mean()
g["b_fp"] = grp.fp.transform(w)
g["b_mpg"] = grp.MIN.transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())
# per-36 style from the previous 60 games (vectorised)
for c in ("REB", "AST", "BLK", "STL", "FG3M"):
    num = grp[c].transform(lambda s: s.shift(1).rolling(60, min_periods=10).sum())
    den = grp.MIN.transform(lambda s: s.shift(1).rolling(60, min_periods=10).sum())
    g[c + "36"] = num / den * 36
g = g.dropna(subset=["b_fp", "b_mpg", "REB36"])
# report absences: Out with an injury/illness reason on the 5:30 PM report
rep = pd.read_pickle(D / "injury_rows_05pm.pkl")
rep = rep[(rep.status == "Out") & rep.kind.isin(["injury", "illness"]) & rep.season.isin(["2024-25", "2025-26"])].drop_duplicates(["player_key", "game_date"])
# baseline for each absent player as of the game date: last known values before that date
last = g[["key", "gd", "b_fp", "b_mpg", "REB36", "AST36", "BLK36", "STL36", "FG3M36", "TEAM"]].sort_values("gd")
rep = rep.assign(gd=pd.to_datetime(rep.game_date)).sort_values("gd")
ab = pd.merge_asof(rep, last.rename(columns={"key": "player_key", "gd": "lgd"}), left_on="gd", right_on="lgd", by="player_key", direction="backward", tolerance=pd.Timedelta(days=45))
ab = ab.dropna(subset=["b_fp"])
ab = ab[ab.b_mpg >= 12]
print(f"{len(ab):,} rotation absences from the reports; {len(g):,} teammate player-games")
ab["pos"] = [U.pos_probs(r.REB36, r.AST36, r.BLK36, r.STL36, r.FG3M36) for r in ab.itertuples()]
byteam = {k: v for k, v in ab.groupby(["gd", "TEAM"])}
rows = []
for (gd, team), x in g.groupby(["gd", "TEAM"]):
    A = byteam.get((gd, team))
    pl = [dict(id=r.key, fp=r.b_fp, mpg=r.b_mpg, pos=U.pos_probs(r.REB36, r.AST36, r.BLK36, r.STL36, r.FG3M36), p_out=0.0) for r in x.itertuples()]
    if A is not None:
        pl += [dict(id=r.player_key, fp=r.b_fp, mpg=r.b_mpg, pos=r.pos, p_out=1.0) for r in A.itertuples() if r.player_key not in set(x.key)]
        up = U.uplifts(pl)
        mx, V, n_abs = A.b_mpg.max(), A.b_fp.sum(), len(A)
    else:
        up, mx, V, n_abs = {}, 0.0, 0.0, 0
    for r in x.itertuples():
        rows.append(dict(gd=gd, team=team, key=r.key, fp=r.fp, base=r.b_fp, mpg=r.b_mpg, up=up.get(r.key, 0.0), n_abs=n_abs, V=V, max_mpg=mx))
R = pd.DataFrame(rows).sort_values(["key", "gd"]).reset_index(drop=True)
# de-uplifted baseline: the recent-games baseline already contains whatever uplift those recent games had, so take it out and add back today's
R["fp_clean"] = R.fp - R.up
R["base2"] = R.groupby("key").fp_clean.transform(lambda s_: s_.shift(1).ewm(halflife=8, min_periods=5).mean())
R = R[(R.mpg >= 6) & R.base2.notna()]
R["adj"] = R.base2 + R.up
sel = R.n_abs > 0
print(f"{len(R):,} teammate games; {sel.sum():,} with a rotation absence")
def rep_(name, m):
    e0, e1 = (R.fp - R.base)[m].abs(), (R.fp - R.adj)[m].abs()
    print(f"   {name:46s} n={m.sum():6,}  MAE {e0.mean():.3f} -> {e1.mean():.3f} ({(1 - e1.mean() / e0.mean()) * 100:+.1f}%)  bias {(R.base - R.fp)[m].mean():+.2f} -> {(R.adj - R.fp)[m].mean():+.2f}  mean uplift {R.up[m].mean():+.2f}")
rep_("ALL teammate games", R.base >= 0)
rep_("games with a rotation absence", sel)
for lo, hi in ((6, 12), (12, 20), (20, 30), (30, 100)):
    rep_(f"  teammate baseline {lo}-{hi} fp/g (absence games)", sel & (R.base >= lo) & (R.base < hi))
rep_("a 28+ mpg player out", R.max_mpg >= 28)
rep_("a 33+ mpg star out", R.max_mpg >= 33)
rep_("two or more rotation players out", R.n_abs >= 2)
print("fantasy-relevant teammates (baseline 20+ fp/g), by size of the absence: actual minus (de-uplifted) baseline vs the model's uplift")
R["cat"] = pd.cut(R.max_mpg, [-1, 0.1, 20, 28, 33, 60], labels=["none", "12-20", "20-28", "28-33", "33+"])
R["act_gain"] = R.fp - R.base2
s2 = R[R.base >= 20].groupby("cat", observed=True).apply(lambda x: pd.Series({"n": len(x), "actual_minus_clean_base": x.act_gain.mean(), "model_uplift": x.up.mean()}), include_groups=False)
print(s2.round(2).to_string())
R.to_pickle(D / "usage_flow_backtest.pkl")
