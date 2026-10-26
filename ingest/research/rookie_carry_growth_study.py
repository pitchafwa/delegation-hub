"""Rookie carry-forward study (Tommy, 2026-10-26, prompted by Cameron Boozer: 51 pts/g early, prospect projection 38, peak only 50).
Live rule (asset_value_v2.py): every later season = prospect path + carry(n) x (early pts/g - rookie projection), carry 0.28 (1 game) .. 0.70 (41),
fit on season two only and applied as the SAME flat amount to seasons 3+. Question: does a rookie who beats his projection keep pulling away from his
draft-slot peers in seasons 3-5 (carry grows), hold a fixed lead (flat, as live), or fall back (carry fades)? And is it different for big positive
surprises and for the youngest rookies?
Data: data/_prospect_engine_rows.pkl (leave-one-class-out prospect projections and actual pts/g, seasons 1-5, classes 2010-2025), rookie games
(data/kalman_input.csv). Target: pts/g in season k+1 for players present (10+ games, 8+ min); presence itself checked separately.
Run from ingest/:  .venv/Scripts/python research/rookie_carry_growth_study.py   (writes data/rookie_carry_growth_study.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
CHECKS = [5, 12, 41, 82]                 # 82 = his whole rookie season

eng = pd.read_pickle(D / "_prospect_engine_rows.pkl")
lam = 0.4 * np.clip((16 - eng["pick"]) / 6.0, 0, 1)
eng["prior"] = np.where(eng["A_cal"].notna(), (1 - lam) * eng["B"] + lam * eng["A_cal"], eng["B"])
P = eng.pivot_table(index="pid", columns="k", values=["prior", "y"])
meta = eng.drop_duplicates("pid").set_index("pid")[["cls", "pick", "dage"]]

g = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
g = g[g["MIN"] > 0]
g["fp"] = g["PTS"] + 1.5 * g["REB"] + 2 * g["AST"] + 3 * g["STL"] + 3 * g["BLK"] + g["FG3M"] + 2 * g["FTM"] - g["FTA"] - g["TOV"] + 3 * g["TD3"]
g["sy"] = g["SEASON"].str[:4].astype(int)
g = g.merge(meta["cls"], left_on="PLAYER_ID", right_index=True)
g = g[g["sy"] == g["cls"]].sort_values(["PLAYER_ID", "GAME_DATE"])
early = {n: g.groupby("PLAYER_ID")["fp"].apply(lambda s: s.iloc[:n].mean() if len(s) >= min(n, 41) + (10 if n < 82 else 0) else np.nan) for n in CHECKS}

# the proposed alternative to test: the live rule and four shapes for seasons 3-5
#   flat:    later season = prior_k + c x S                         (live; c fit on season two)
#   by year: later season = prior_k + c_k x S                       (c_k fit for each season separately -- does it grow?)
#   ratio:   later season = prior_k x (1 + c x S / prior_0)         (he grows like his peers but from a scaled-up level)
def wls(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


out = {}
for n in CHECKS:
    S = early[n] - P[("prior", 0)]
    print(f"\n=== surprise measured after {'his whole rookie season' if n == 82 else f'{n} games'} ===")
    rows = []
    for k in range(1, 5):
        d = pd.DataFrame({"y": P[("y", k)], "pk": P[("prior", k)], "p0": P[("prior", 0)], "S": S}).join(meta).dropna(subset=["y", "pk", "S"])
        d = d[d["cls"] + k <= 2025]
        if len(d) < 40:
            continue
        b = wls(d[["pk", "S"]].to_numpy(), d["y"].to_numpy())
        # bigger surprises / youngest rookies: separate slopes
        Sp, Sm = d["S"].clip(lower=0), d["S"].clip(upper=0)
        bpm = wls(np.column_stack([d["pk"], Sp, Sm]), d["y"].to_numpy())
        young = (d["dage"] < 20).astype(float)
        by = wls(np.column_stack([d["pk"], d["S"], d["S"] * young]), d["y"].to_numpy())
        top = d[d["S"] >= 8]
        rows.append({"season": k + 1, "players": len(d), "carry": b[2], "prior wt": b[1], "carry if ahead": bpm[2], "carry if behind": bpm[3],
                     "extra if under 20": by[3], "8+ ahead: n": len(top), "8+ ahead: lead kept": (top["y"] - top["pk"]).mean() / top["S"].mean() if len(top) else np.nan})
    t = pd.DataFrame(rows).set_index("season")
    print(t.round(2).to_string())
    out[n] = t.reset_index().to_dict("records")

    # held-out (leave one class out): flat carry from season two vs a carry for each season, scored on seasons 3-5
    d_all = []
    for k in range(1, 5):
        d = pd.DataFrame({"y": P[("y", k)], "pk": P[("prior", k)], "p0": P[("prior", 0)], "S": S, "k": k}).join(meta).dropna(subset=["y", "pk", "S"])
        d_all.append(d[d["cls"] + k <= 2025])
    A = pd.concat(d_all)
    err = {"prospect path only": [], "flat (live)": [], "by season": [], "ratio": []}
    for c in A["cls"].unique():
        tr, te = A[A["cls"] != c], A[(A["cls"] == c) & (A["k"] >= 2)]
        if not len(te):
            continue
        t2 = tr[tr["k"] == 1]
        bf = wls(t2[["pk", "S"]].to_numpy(), t2["y"].to_numpy())
        bk = {k: wls(tr[tr["k"] == k][["pk", "S"]].to_numpy(), tr[tr["k"] == k]["y"].to_numpy()) for k in range(1, 5)}
        br = wls(np.column_stack([tr["pk"], tr["pk"] * tr["S"] / tr["p0"].clip(lower=8)]), tr["y"].to_numpy())
        b0 = {k: wls(tr[tr["k"] == k][["pk"]].to_numpy(), tr[tr["k"] == k]["y"].to_numpy()) for k in range(1, 5)}
        for r in te.itertuples():
            err["prospect path only"].append(b0[r.k][0] + b0[r.k][1] * r.pk - r.y)
            err["flat (live)"].append(bk[r.k][0] + bk[r.k][1] * r.pk + bf[2] * r.S - r.y)
            err["by season"].append(bk[r.k][0] + bk[r.k][1] * r.pk + bk[r.k][2] * r.S - r.y)
            err["ratio"].append(br[0] + br[1] * r.pk + br[2] * r.pk * r.S / max(r.p0, 8) - r.y)
    print("held-out typical miss, seasons 3-5 (pts/g):", {k: round(float(np.sqrt(np.mean(np.square(v)))), 3) for k, v in err.items()})
    out[f"{n}_heldout"] = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in err.items()}

# presence: are big early surprises more likely to still be in the league (and playing) in seasons 3-5?
S = early[82] - P[("prior", 0)]
pres = pd.DataFrame({"S": S}).join(meta)
for k in range(2, 5):
    ok = pres["cls"] + k <= 2025
    present = P[("y", k)].notna()
    q = pd.qcut(S[ok], 4, labels=["bottom", "2nd", "3rd", "top"])
    print(f"season {k + 1} present, by rookie surprise quarter:", present[ok].groupby(q).mean().round(2).to_dict())
json.dump({str(k): v for k, v in out.items()}, open(D / "rookie_carry_growth_study.json", "w"), indent=1, default=float)
