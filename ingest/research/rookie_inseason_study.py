"""Rookie in-season study (Tommy, 2026-10-25): how much should a rookie's first NBA games move his projection away from the draft-based (prospect engine) projection?

Background: once a rookie plays, the hub stopped using the prospect engine and fell back to a weak Kalman start plus his first 2-3 games (Peterson 37.2 -> 22.9,
Lendeborg 23.0 -> 42.8). Tommy's point: early games DO show role, minutes and usage, but rookies also take time to acclimate. Measure the balance on real history.

Data
  prior    the prospect engine's rookie-season projection, made WITHOUT that draft class (leave-one-class-out rows in data/_prospect_engine_rows.pkl):
           (1 - lam) x B (draft slot / age / talent ridge) + lam x A_cal (calibrated Output B), lam = 0.4 for picks 1-10 tapering to 0 by pick 16 -- the live blend
           (without the team-situation term, which history does not have; the live prior is a little better, so the weights below lean slightly toward the games)
  games    every rookie-season game with minutes (data/kalman_input.csv), league scoring
  target   his fantasy points per game over the REST of the rookie season after N games (weighted by games left), and his second season
Checkpoints N = 1, 2, 3, 5, 8, 12, 20, 30, 41 games; a rookie counts at N if he played at least N + 10 games that season.
Models (fit leave-one-class-out, scored on the held-out class):
  prior only | early games only | prior + early points | prior + early points + early minutes (the role signal) | the same with minutes split into the role and the rate
Also: how much rookies improve over a season (acclimation), and how much of an early surprise carries into season two.
Run from ingest/:  .venv/Scripts/python research/rookie_inseason_study.py   (writes data/rookie_inseason_study.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R_ = Path(__file__).resolve().parent
D = R_ / "data"
CHECKS = [1, 2, 3, 5, 8, 12, 20, 30, 41]

eng = pd.read_pickle(D / "_prospect_engine_rows.pkl")
lam = 0.4 * np.clip((16 - eng["pick"]) / 6.0, 0, 1)
eng["prior"] = np.where(eng["A_cal"].notna(), (1 - lam) * eng["B"] + lam * eng["A_cal"], eng["B"])
p0 = eng[eng["k"] == 0].drop_duplicates("pid").set_index("pid")
p1 = eng[eng["k"] == 1].drop_duplicates("pid").set_index("pid")

g = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
g = g[g["MIN"] > 0]
g["fp"] = g["PTS"] + 1.5 * g["REB"] + 2 * g["AST"] + 3 * g["STL"] + 3 * g["BLK"] + g["FG3M"] + 2 * g["FTM"] - g["FTA"] - g["TOV"] + 3 * g["TD3"]
g["sy"] = g["SEASON"].str[:4].astype(int)

rows = []
for pid, r in p0.iterrows():
    cls = int(r["cls"])
    gl = g[(g["PLAYER_ID"] == pid) & (g["sy"] == cls)].sort_values("GAME_DATE")
    if len(gl) < 11:
        continue
    fp, mn = gl["fp"].to_numpy(float), gl["MIN"].to_numpy(float)
    y2 = float(p1.loc[pid, "y"]) if pid in p1.index and np.isfinite(p1.loc[pid, "y"]) else np.nan
    pr2 = float(p1.loc[pid, "prior"]) if pid in p1.index else np.nan
    for n in CHECKS:
        if len(fp) < n + 10:
            continue
        rows.append(dict(pid=pid, cls=cls, pick=r["pick"], n=n, prior=float(r["prior"]), E=fp[:n].mean(), M=mn[:n].mean(), Rt=fp[:n].sum() / max(mn[:n].sum(), 1),
                         y=fp[n:].mean(), ym=mn[n:].mean(), w=min(len(fp) - n, 60), whole=fp.mean(), first10=fp[:10].mean(), last20=fp[-20:].mean(), ng=len(fp), y2=y2, prior2=pr2))
R = pd.DataFrame(rows)
print(f"{R['pid'].nunique()} rookies (draft classes {R['cls'].min()}-{R['cls'].max()}, college players with 11+ rookie games)\n")

MODELS = {"prior only": ["prior"], "early games only": ["E"], "prior + early pts": ["prior", "E"], "prior + early pts + minutes": ["prior", "E", "M"],
          "prior + minutes + rate": ["prior", "M", "Rt", "MxRt"]}


def wls(X, y, w):
    X1 = np.column_stack([np.ones(len(X)), X])
    W = np.sqrt(w)
    return np.linalg.lstsq(X1 * W[:, None], y * W, rcond=None)[0]


def feats(df, cols):
    df = df.assign(MxRt=df["M"] * df["Rt"])
    return df[cols].to_numpy(float)


res, coefs = [], {}
for n in CHECKS:
    s = R[R["n"] == n]
    line = {"n": n, "rookies": len(s)}
    for name, cols in MODELS.items():
        err, wt = [], []
        for c in s["cls"].unique():
            tr, te = s[s["cls"] != c], s[s["cls"] == c]
            b = wls(feats(tr, cols), tr["y"].to_numpy(), tr["w"].to_numpy(float))
            pred = b[0] + feats(te, cols) @ b[1:]
            err.extend((pred - te["y"]).tolist()); wt.extend(te["w"].tolist())
        err, wt = np.array(err), np.array(wt, float)
        line[name] = float(np.sqrt(np.sum(wt * err ** 2) / wt.sum()))
        coefs[(n, name)] = wls(feats(s, cols), s["y"].to_numpy(), s["w"].to_numpy(float)).tolist()
    res.append(line)
T = pd.DataFrame(res).set_index("n")
print("Error predicting the REST of the rookie season (points per game, held-out draft classes; lower is better):")
print(T.round(2).to_string())

print("\nFitted blend (all classes): rest of season = a + b x prior + c x early pts/g  [+ d x early minutes]")
for n in CHECKS:
    a, b, c = coefs[(n, "prior + early pts")]
    a2, b2, c2, d2 = coefs[(n, "prior + early pts + minutes")]
    print(f"  {n:2d} games: a {a:5.1f}  b {b:.2f}  c {c:.2f}   (share on games {c / (b + c):.0%})     with minutes: a {a2:5.1f} b {b2:.2f} c {c2:.2f} d {d2:+.2f}/min")

# acclimation: do rookies get better as the season goes on?
w = R[R["n"] == 1].drop_duplicates("pid")
w = w[w["ng"] >= 40]
print(f"\nAcclimation ({len(w)} rookies with 40+ games): first 10 games {w['first10'].mean():.1f} pts/g, last 20 {w['last20'].mean():.1f} (+{(w['last20'] - w['first10']).mean():.1f}); "
      f"top-10 picks: {w[w['pick'] <= 10]['first10'].mean():.1f} -> {w[w['pick'] <= 10]['last20'].mean():.1f}")
for n in (2, 5, 12):
    s = R[R["n"] == n]
    print(f"  after {n:2d} games: rest of season minus early average {np.average(s['y'] - s['E'], weights=s['w']):+.1f} pts/g; minutes {np.average(s['ym'] - s['M'], weights=s['w']):+.1f}")

# season two: how much of the early surprise carries over
print("\nSeason two: year-2 pts/g = a + b x year-2 prior + c x (rookie-season surprise so far = early pts/g - rookie prior)")
carry = {}
for n in (5, 12, 20, 41):
    s = R[(R["n"] == n) & R["y2"].notna() & R["prior2"].notna()]
    X = np.column_stack([s["prior2"], s["E"] - s["prior"]])
    b = wls(X, s["y2"].to_numpy(), np.ones(len(s)))
    carry[n] = b.tolist()
    print(f"  after {n:2d} games ({len(s)} rookies): a {b[0]:.1f}  b {b[1]:.2f}  c {b[2]:.2f}")
json.dump({"table": res, "coefs": {f"{k[0]}|{k[1]}": v for k, v in coefs.items()}, "carry": carry}, open(D / "rookie_inseason_study.json", "w"), indent=1)
