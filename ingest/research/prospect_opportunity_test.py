"""How much does a rookie's team opportunity matter, and for how long?
Uses the shared healthy-roster measure (team_context.py): open_fp = production that LEFT the team minus veterans who ARRIVED.
Leave-one-draft-class-out ridge on draft slot + age + talent (+ #1-pick tier), with and without opportunity, by seasons since draft.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from team_context import prep_panel, team_context  # noqa: E402

D = ROOT / "data"
pp = prep_panel(pd.read_csv(D / "breakout_panel_ctx2.csv"))
gl = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "TEAM"])
gl["sy"] = gl["SEASON"].str[:4].astype(int)
first = gl.sort_values("GAME_DATE").groupby(["PLAYER_ID", "sy"]).first().reset_index()
abbr_id = pp.dropna(subset=["team_id"]).drop_duplicates(["team", "yr"]).set_index(["team", "yr"])["team_id"].to_dict()
E = pd.read_pickle(D / "_prospect_engine_rows.pkl")
E["is1"] = (E["pick"] == 1).astype(float)
rows = []
for cls in sorted(E["cls"].unique()):
    if cls - 1 < pp["yr"].min():
        continue
    ctx = team_context(pp, cls - 1)
    for r in first[first["sy"] == cls].itertuples():
        tid = abbr_id.get((r.TEAM, cls))
        if tid is not None and int(tid) in ctx.index:
            rows.append(dict(pid=r.PLAYER_ID, cls=cls, open=float(ctx.loc[int(tid), "open_fp"])))
C = pd.DataFrame(rows).drop_duplicates(["pid", "cls"])
S = E[["pid", "cls", "pick", "lp", "dage", "talent", "is1", "k", "y"]].merge(C, on=["pid", "cls"])
BASE = ["lp", "dage", "talent", "is1"]
rm = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))


def loco(feats, sub):
    out = pd.Series(np.nan, index=sub.index)
    for cls in sub["cls"].unique():
        for k in range(5):
            te, tr = (sub["cls"] == cls) & (sub["k"] == k), (sub["cls"] != cls) & (sub["k"] == k)
            if te.sum() == 0 or tr.sum() < 30:
                continue
            X = sub.loc[tr, feats]
            sc = StandardScaler().fit(X)
            out[te] = Ridge(alpha=5).fit(sc.transform(X), sub.loc[tr, "y"]).predict(sc.transform(sub.loc[te, feats]))
    return out


print(f"rookies with a known team situation: {S['pid'].nunique()}; open_fp sd = {S['open'].std():.0f} (fantasy pts per game)")
for label, sub in [("PICKS 1-15", S[S["pick"] <= 15]), ("ALL PICKS", S)]:
    pr0, pr1 = loco(BASE, sub), loco(BASE + ["open"], sub)
    print(f"\n{label}: held-out RMSE without -> with opportunity, and the effect of +1 sd of opportunity on pts/g")
    for k in range(5):
        t = sub[sub["k"] == k]
        sc = StandardScaler().fit(t[BASE + ["open"]])
        m = Ridge(alpha=5).fit(sc.transform(t[BASE + ["open"]]), t["y"])
        print(f"  {'rookie year' if k == 0 else f'year {k + 1}':11s} n={len(t):3d}  {rm(t['y'], pr0[sub['k'] == k]):5.2f} -> {rm(t['y'], pr1[sub['k'] == k]):5.2f}   +1 sd opportunity = {m.coef_[-1]:+5.2f} pts/g")
