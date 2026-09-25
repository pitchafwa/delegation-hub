"""Study 3: a multi-tier injury-risk score for the board, and whether real injury details (episodes, repeats, surgeries) improve it.

Outcome: a MATERIAL ABSENCE next season = missing 20%+ of the games (16+ of 82) for a player who played 20+ games at 15+ mpg the season before.
Features, all known before the season:
  history (2010s on, from season totals): games-missed share in each of the last 3 seasons, age, minutes per game
  injury details (official reports, 2021-22 on): number of long episodes (10+ game days out) in the last two seasons, days out last season,
      repeat same-side episodes of a body part, a surgery/tear/fracture episode of 20+ days last season
Evaluated leave-one-season-out over outcome seasons 2022-23 to 2025-26 (AUC and Brier; lower Brier = better calibrated).
Then writes the coefficients of the chosen model and tier cut points to data/injury_risk_model.json (used by build_hub_data.py).
Run from ingest/:  uv run python research/injury_risk_model.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import injury_common as C

sys.stdout.reconfigure(encoding="utf-8")
sb = pd.read_csv(C.D / "player_season_base.csv")
sb["yr"] = sb.SEASON.str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["len"] = np.where(sb.yr.isin([2019, 2020]), 72, 82)          # 2019-20 and 2020-21 were shortened
sb["missed"] = (1 - sb.GP / sb.len).clip(lower=0)
sb["mpg"] = sb.MIN / sb.GP
S = sb.set_index(["PLAYER_ID", "yr"])
ep = pd.read_pickle(C.D / "injury_episodes.pkl")
# map report player keys to PLAYER_ID through the logs
lg = pd.read_csv(C.D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
lg["player_key"] = lg.PLAYER_NAME.map(C.key_of_log)
k2id = lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
ep["pid"] = ep.player_key.map(k2id)
ep = ep.dropna(subset=["pid"]).copy()
ep["pid"] = ep.pid.astype(int)
ep["yr"] = ep.season.str[:4].astype(int)
_rows = pd.read_pickle(C.D / "injury_rows_05pm.pkl")
SEASON_END = {int(s[:4]): g.gd.max() for s, g in _rows.groupby("season")}
MAJOR_RE = "surg|tear|torn|fract|broken|acl|achilles|rupture"


def report_feats(pid, t):
    """features from reports over seasons t-1 and t-2 (only 2021-22 onward exist)"""
    e = ep[(ep.pid == pid) & (ep.yr.between(t - 2, t - 1)) & (ep.yr >= 2021)]
    e1 = e[e.yr == t - 1]
    sea_end = SEASON_END.get(t - 1)
    live = e1[(e1.end >= sea_end - pd.Timedelta(days=10)) & (e1.n_out >= 5)] if sea_end is not None else e1.iloc[0:0]      # still listed when the season ended
    late_major = int(((live.major == 1) | (live.n_out >= 25)).any()) if len(live) else 0
    late_out = int(len(live) > 0)
    long_ = e[e.n_out >= 10]
    rep = 0
    for (part, side), g in e[(e.n_out >= 3) & (e.side != "")].groupby(["part", "side"]):
        if len(g) >= 2:
            rep += 1
    return dict(long_eps=len(long_), out_days1=float(e1.n_out.sum()), repeats=rep, n_eps=len(e[e.n_out >= 3]), big_recent=int(((e1.n_out >= 20)).sum()), late_out=late_out, late_major=late_major)


import os
THRESH = float(os.environ.get("THRESH", "0.20"))
# early-season availability: games played in the first 35 days of the season (about 16 team games)
_lg = pd.read_csv(C.D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN"])
_lg = _lg[_lg.MIN > 0]
_lg["gd"] = pd.to_datetime(_lg.GAME_DATE)
_lg["yr"] = _lg.SEASON.str[:4].astype(int)
_start = _lg.groupby("yr").gd.min()
_lg = _lg[_lg.gd <= _lg.yr.map(_start) + pd.Timedelta(days=35)]
EARLY = _lg.groupby(["PLAYER_ID", "yr"]).size().to_dict()
rows = []
for t in range(2022, 2026):                       # outcome seasons 2022-23 .. 2025-26
    _p2 = sb[(sb.yr == t - 2) & (sb.mpg >= 15) & (sb.GP >= 20)].PLAYER_ID
    prev = sb[(sb.yr == t - 1) & (sb.GP >= 5) & ((sb.mpg >= 15) | sb.PLAYER_ID.isin(_p2))]
    for r in prev.itertuples():
        pid = r.PLAYER_ID
        cur = S.loc[(pid, t)] if (pid, t) in S.index else None
        if cur is None or cur.GP < 5:
            continue
        m2 = S.loc[(pid, t - 2)].missed if (pid, t - 2) in S.index else np.nan
        m3 = S.loc[(pid, t - 3)].missed if (pid, t - 3) in S.index else np.nan
        f = dict(early=EARLY.get((pid, t), 0), t=t, pid=pid, y=int(cur.missed >= THRESH), miss=float(cur.missed), m1=r.missed, m2=m2, m3=m3, age=r.AGE + 1, mpg=r.mpg)
        f.update(report_feats(pid, t))
        rows.append(f)
D = pd.DataFrame(rows)
for c in ("m2", "m3"):
    D[c + "_na"] = D[c].isna() * 1.0
    D[c] = D[c].fillna(D.m1)
print(f"{len(D)} player-seasons; base rate of a material absence: {D.y.mean():.3f}")
print(D.groupby("t").y.agg(["mean", "size"]).round(3).T.to_string())

BASE = ["m1", "m2", "m3", "age", "mpg", "m2_na", "m3_na"]
INJ = BASE + ["long_eps", "out_days1", "repeats", "big_recent", "late_out", "late_major"]
INJ2 = BASE + ["repeats", "big_recent"]


def cv(cols, name, show=True):
    pr = np.zeros(len(D))
    for t in sorted(D.t.unique()):
        tr, te = D.t != t, D.t == t
        mu, sd = D.loc[tr, cols].mean(), D.loc[tr, cols].std().replace(0, 1)
        m = LogisticRegression(C=0.5, max_iter=3000).fit(((D.loc[tr, cols] - mu) / sd), D.y[tr])
        pr[te.to_numpy()] = m.predict_proba((D.loc[te, cols] - mu) / sd)[:, 1]
    if show:
        print(f"   {name:44s} AUC {roc_auc_score(D.y, pr):.3f}  Brier {brier_score_loss(D.y, pr):.4f}")
    return pr


print("\nleave-one-season-out:")
print(f"   {'constant base rate':44s} AUC 0.500  Brier {brier_score_loss(D.y, np.full(len(D), D.y.mean())):.4f}")
# the current board rule: number of the last 4 seasons with 25%+ missed
D["old_n"] = [sum(1 for k in range(1, 5) if (r.pid, r.t - k) in S.index and S.loc[(r.pid, r.t - k)].missed > 0.25) for r in D.itertuples()]
print(f"   {'current board rule (count of 25%+ missed seasons)':44s} AUC {roc_auc_score(D.y, D.old_n):.3f}")
p_last = cv(["m1"], "last season's missed share only")
p_base = cv(BASE, "history: 3 seasons missed, age, minutes")
p_inj = cv(INJ, "+ injury details (episodes, repeats, big injury)")
cv(INJ2, "+ repeats and big injury only")
for name, c in (("history + still-out-at-season-end", BASE + ["late_out", "late_major"]), ("history + late_major only", BASE + ["late_major"]), ("history + long episodes only", BASE + ["long_eps"]), ("history + repeats only", BASE + ["repeats"]), ("history + big injury only", BASE + ["big_recent"]), ("history + days out only", BASE + ["out_days1"])):
    cv(c, name)
D["p_base"], D["p_inj"] = p_base, p_inj
print("\ncalibration of the injury-detail model (predicted vs actual, quintile bins):")
D["bin"] = pd.qcut(D.p_inj, 5, labels=False)
print(D.groupby("bin").agg(pred=("p_inj", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).T.to_string())
print("\nsame for history-only:")
D["bin0"] = pd.qcut(D.p_base, 5, labels=False)
print(D.groupby("bin0").agg(pred=("p_base", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).T.to_string())

# ---- final fit on all seasons, tiers by predicted probability
cols = INJ
mu, sd = D[cols].mean(), D[cols].std().replace(0, 1)
m = LogisticRegression(C=0.5, max_iter=3000).fit((D[cols] - mu) / sd, D.y)
print("\nfinal model (standardized coefficients, log-odds):")
for c, b in sorted(zip(cols, m.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"   {c:12s} {b:+.2f}")
allp = m.predict_proba((D[cols] - mu) / sd)[:, 1]
cuts = [0.15, 0.25, 0.38, 0.55]
tier = np.digitize(allp, cuts)
print("\ntier table (cut points on predicted probability of missing 20%+ of games):")
print(pd.DataFrame({"tier": tier, "y": D.y}).groupby("tier").y.agg(["mean", "size"]).round(3).to_string())
json.dump({"cols": cols, "mu": mu.to_dict(), "sd": sd.to_dict(), "coef": dict(zip(cols, m.coef_[0])), "intercept": float(m.intercept_[0]), "cuts": cuts, "base_rate": float(D.y.mean())},
          open(C.D / "injury_risk_model.json", "w"), indent=1)

print("EARLY-SEASON availability: played 8 or fewer of the first ~16 team games (missed half+), for the same players")
D["ye"] = (D.early <= 8) * 1.0
print(f"   base rate {D.ye.mean():.3f}")
def cv2(cols, name):
    pr = np.zeros(len(D))
    for t in sorted(D.t.unique()):
        tr, te = D.t != t, D.t == t
        mu2, sd2 = D.loc[tr, cols].mean(), D.loc[tr, cols].std().replace(0, 1)
        m2 = LogisticRegression(C=0.5, max_iter=3000).fit((D.loc[tr, cols] - mu2) / sd2, D.ye[tr])
        pr[te.to_numpy()] = m2.predict_proba((D.loc[te, cols] - mu2) / sd2)[:, 1]
    print(f"   {name:44s} AUC {roc_auc_score(D.ye, pr):.3f}  Brier {brier_score_loss(D.ye, pr):.4f}")
    return pr
cv2(["m1"], "last season's missed share only")
cv2(BASE, "history")
cv2(BASE + ["late_out", "late_major"], "+ still listed out at season end")
cv2(BASE + ["late_major"], "+ late major injury only")
cv2(INJ, "+ all injury details")
print(D.groupby(["late_out", "late_major"]).ye.agg(["mean", "size"]).round(3).to_string())
