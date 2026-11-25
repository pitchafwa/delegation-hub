"""Riser correction inside the veterans' in-season update (2026-11-25). asset_value_v2.py sets a veteran's this-season number to path + w x (season DELCO - path)
(w 0.7; 1.0 when DELCO is 5+ above the path; 0.6 when 5+ below; season_gap_test.py PART 5). delco_riser_study.py found season DELCO itself too slow.
Rest of this season after game N (5, 10, 20, 30), held-out keeper paths (data/_engine_blend_rows.pkl, h=1), target seasons 2021-22..2024-25 only (the riser
coefficients used here were fit on 2011-21 alone, F3_train). Candidates:
  A current rule on DELCO      B corrected DELCO alone      C current rule on corrected DELCO      D best single w on corrected DELCO (fit on the same rows: an upper bound)
Run from ingest/:  .venv/Scripts/python research/riser_path_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
RM = json.load(open(D / "delco_riser_model.json"))
CO = RM["models"]["F3_train"]
LABS = ["3-5", "8-12", "16-20", "25-50"]

g = pd.read_csv(D / "game_logs_unified.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA"])
g = g[g.MIN > 0]
g["fp"] = g.PTS + 1.5 * g.REB + 2 * g.AST + 3 * g.STL + 3 * g.BLK + g.FG3M + 2 * g.FTM - g.FTA - g.TOV
g["GAME_DATE"] = pd.to_datetime(g.GAME_DATE).dt.normalize()
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "SEASON", "MIN", "after", "afterMIN"])
k = k[k.MIN > 0]
k["GAME_DATE"] = pd.to_datetime(k.GAME_DATE).dt.normalize()
m = k.merge(g[["PLAYER_ID", "GAME_DATE", "fp"]], on=["PLAYER_ID", "GAME_DATE"]).sort_values(["PLAYER_ID", "GAME_DATE"])
m["tgt"] = m.SEASON.str[:4].astype(int) + 1 - 1          # season start year
E = pd.read_pickle(D / "_engine_blend_rows.pkl")
E = E[E.h == 1].copy()
E["prod"] = 0.75 * E["B_ridge"] + 0.25 * E["A_kalman"]          # the keeper path for the next season (asset_value_v2: w 0.75 at h=1)
E["tgt"] = E["origin"] + 1
E = E[E.tgt >= 2021]
want = set(zip(E.PLAYER_ID, E.tgt))

rows = []
for (pid, season), x in m.groupby(["PLAYER_ID", "SEASON"], sort=False):
    t = int(season[:4])
    if (pid, t) not in want:
        continue
    f, mn, af, am = x.fp.to_numpy(float), x.MIN.to_numpy(float), x["after"].to_numpy(float), x.afterMIN.to_numpy(float)
    for N in (5, 10, 20, 30):
        if len(f) - N < 10 or np.isnan(af[N - 1]):
            continue
        st = next(i for i, (lo, hi) in enumerate(RM["stages"]) if lo <= N <= hi)
        c = CO[LABS[st]]
        adj = c[0] + c[1] * (f[:N].mean() - af[N - 1]) + c[2] * (f[max(0, N - 10):N].mean() - f[:N].mean()) + c[3] * (mn[:N].mean() - am[N - 1]) + c[4] * (mn[max(0, N - 5):N].mean() - mn[:N].mean())
        rows.append({"PLAYER_ID": pid, "tgt": t, "N": N, "S": af[N - 1], "Sc": af[N - 1] + adj, "y": f[N:].mean(), "n2": len(f) - N})
R = pd.DataFrame(rows).merge(E[["PLAYER_ID", "tgt", "prod"]], on=["PLAYER_ID", "tgt"])
rm = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
ma = lambda a, b: float(np.abs(a - b).mean())


def rule(path, s):
    d = s - path
    w = np.where(d >= 5, 1.0, np.where(d <= -5, 0.6, 0.7))
    return path + w * d


print(f"rows {len(R)} (players {R.PLAYER_ID.nunique()}, seasons {sorted(R.tgt.unique())})")
for N, t in R.groupby("N"):
    A, B, C = rule(t["prod"], t.S), t.Sc, rule(t["prod"], t.Sc)
    ws = np.arange(0, 1.01, 0.1)
    e = [rm(t.y, t["prod"] + w * (t.Sc - t["prod"])) for w in ws]
    up = (t.Sc - t.S) >= 2
    print(f"game {N:2d} n={len(t):4d}  RMSE/MAE  A rule+DELCO {rm(t.y, A):.2f}/{ma(t.y, A):.2f}  B corrected alone {rm(t.y, B):.2f}/{ma(t.y, B):.2f}"
          f"  C rule+corrected {rm(t.y, C):.2f}/{ma(t.y, C):.2f}  D best w {ws[int(np.argmin(e))]:.1f} {min(e):.2f}"
          f"  | correction 2+ up n={up.sum()}: A miss {np.mean(t.y[up] - A[up]):+.2f}  C miss {np.mean(t.y[up] - C[up]):+.2f}")
print("\nE: flat w 0.7 on corrected DELCO (the rule's base weight, without the extra push for 5+ above the path)")
for N, t in R.groupby("N"):
    A, C = rule(t["prod"], t.S), rule(t["prod"], t.Sc)
    F = t["prod"] + 0.7 * (t.Sc - t["prod"])
    up, dn = (t.Sc - t.S) >= 2, (t.Sc - t.S) <= -2
    print(f"game {N:2d}  A {rm(t.y, A):.2f}/{ma(t.y, A):.2f}  C {rm(t.y, C):.2f}/{ma(t.y, C):.2f}  E {rm(t.y, F):.2f}/{ma(t.y, F):.2f}"
          f"  | up n={up.sum()} miss A {np.mean(t.y[up] - A[up]):+.2f} C {np.mean(t.y[up] - C[up]):+.2f} E {np.mean(t.y[up] - F[up]):+.2f}"
          f"  | down n={dn.sum()} miss A {np.mean(t.y[dn] - A[dn]):+.2f} E {np.mean(t.y[dn] - F[dn]):+.2f}")
