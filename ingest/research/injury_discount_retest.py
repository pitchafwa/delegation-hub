"""Re-test (2026-09-27, at Tommy's request): does using the NEW health-confirmed injury data (built 2026-09-26/27: only games a player was actually
listed Out/Doubtful for an injury or illness, not rest/coach's-decision/G-League) -- plus injury type, severity tier, recurring-injury flags, and a
covid-era check -- fix the original objection to folding an injury discount into the Kalman/asset-value projections (a smooth games-missed-based curve
mislabeled Cooper Flagg's normal 70/82-game rookie season the same as Jayson Tatum's real Achilles-recovery season) and improve next-season accuracy.

Outcome: next season's ANY-REASON missed-games fraction (what actually costs a fantasy manager games; continuous RMSE + binary AUC at 25%+).
Models compared, leave-one-season-out over outcome years 2023-24..2025-26 (the only years with 2+ prior seasons of real health-confirmed history):
  A  baseline (what's already in production for the games-left projection): any-reason missed fraction, last 3 seasons, age, mpg
  B  A + this season's HEALTH-CONFIRMED missed fraction (h1) and its severity mix (major-tier share) and a recurring-injury flag
  C  A + two years of health-confirmed history (h1, h2) + severity + recurring
Also: an anecdote check (does the new measure actually separate real injuries from incidental non-injury absences, Flagg/Tatum-style) and a covid
diagnostic (2020-21 predates our injury-report data entirely -- can we say anything about it at all).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import injury_common as C

sys.stdout.reconfigure(encoding="utf-8")
D = C.D

# ---------------- base season table (any-reason GP) ----------------
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb.SEASON.str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["len"] = np.where(sb.yr.isin([2019, 2020]), 72, 82)
sb["missed"] = (1 - sb.GP / sb.len).clip(lower=0)
sb["mpg"] = sb.MIN / sb.GP

# ---------------- health-confirmed missed games (Out/Doubtful, injury/illness/mgmt reason; report coverage from 2021-12) ----------------
r5 = pd.read_pickle(D / "injury_rows_05pm.pkl")
h = r5[r5.status.isin(["Out", "Doubtful"]) & r5.kind.isin(["injury", "illness", "mgmt"])].drop_duplicates(["player_key", "game_date"]).copy()
lg = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
lg["player_key"] = lg.PLAYER_NAME.map(C.key_of_log)
k2id = lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
h["pid"] = h.player_key.map(k2id)
h = h.dropna(subset=["pid"]).copy()
h["pid"] = h.pid.astype(int)
h["yr"] = h.season.str[:4].astype(int)
# severity tier per row (from the reason text) and major-tier flag
tiers = h.reason.map(lambda x: C.parse_reason(x)["tier"])
h["major"] = (tiers == "major").astype(int)
HM = h.groupby(["pid", "yr"]).size().to_dict()
HM_MAJOR = h.groupby(["pid", "yr"]).major.sum().to_dict()
sb["hmissed"] = [(min(HM.get((int(a), int(b)), 0) / l, 1.0) if b >= 2022 else np.nan) for a, b, l in zip(sb.PLAYER_ID, sb.yr, sb.len)]
sb["hmajor"] = [(HM_MAJOR.get((int(a), int(b)), 0) / max(HM.get((int(a), int(b)), 0), 1) if b >= 2022 and HM.get((int(a), int(b)), 0) > 0 else 0.0) for a, b in zip(sb.PLAYER_ID, sb.yr)]

# ---------------- recurring-injury flag (2+ episodes, same body-part group + side, trailing 2 seasons) ----------------
ep = pd.read_pickle(D / "injury_episodes.pkl")
ep["pid"] = ep.player_key.map(k2id)
ep = ep.dropna(subset=["pid"]).copy()
ep["pid"] = ep.pid.astype(int)
ep["yr"] = ep.season.str[:4].astype(int)


def recurring_flag(pid, upto_yr, lookback=2):
    e = ep[(ep.pid == pid) & (ep.yr <= upto_yr) & (ep.yr > upto_yr - lookback) & (ep.n_out >= 3)]
    for (part, side), g in e[e.side != ""].groupby(["part", "side"]):
        if len(g) >= 2:
            return 1
    for part, g in e[e.side == ""].groupby("part"):
        if len(g) >= 3:
            return 1
    return 0


REC = {}
for pid in sb.PLAYER_ID.unique():
    for yr in sb.yr.unique():
        if yr >= 2022:
            REC[(pid, yr)] = recurring_flag(pid, yr)

S = sb.set_index(["PLAYER_ID", "yr"])
print(f"seasons with health-confirmed data: {sb[sb.yr >= 2022].yr.unique()}; {(~sb.hmissed.isna()).sum()} player-seasons")


def gm(pid, y, col):
    if (pid, y) in S.index:
        v = S.loc[(pid, y)][col]
        return float(v) if pd.notna(v) else np.nan
    return np.nan


def feats(pid, t):
    """features predicting outcome season t from seasons before it (t-1, t-2, t-3)"""
    if (pid, t - 1) not in S.index:
        return None
    r = S.loc[(pid, t - 1)]
    m1, age, mpg = float(r.missed), r.AGE + 1, r.mpg
    m2, m3 = gm(pid, t - 2, "missed"), gm(pid, t - 3, "missed")
    h1, h2 = gm(pid, t - 1, "hmissed"), gm(pid, t - 2, "hmissed")
    hmaj1 = gm(pid, t - 1, "hmajor")
    rec = REC.get((pid, t - 1), 0)
    return dict(m1=m1, m2=(m2 if not np.isnan(m2) else m1), m3=(m3 if not np.isnan(m3) else m1),
                m2_na=float(np.isnan(m2)), m3_na=float(np.isnan(m3)), age=age, mpg=mpg,
                h1=(h1 if not np.isnan(h1) else m1), h1_na=float(np.isnan(h1)), h2=(h2 if not np.isnan(h2) else (h1 if not np.isnan(h1) else m1)),
                hmaj1=(hmaj1 if not np.isnan(hmaj1) else 0.0), rec=rec)


rows = []
for t in range(2023, 2026):
    p2 = sb[(sb.yr == t - 2) & (sb.mpg >= 15) & (sb.GP >= 20)].PLAYER_ID
    prev = sb[(sb.yr == t - 1) & (sb.GP >= 5) & ((sb.mpg >= 15) | sb.PLAYER_ID.isin(p2))]
    for pid in prev.PLAYER_ID:
        if (pid, t) not in S.index or S.loc[(pid, t)].GP < 5:
            continue
        f = feats(pid, t)
        if f is None:
            continue
        f.update(t=t, y=int(S.loc[(pid, t)].missed >= 0.25), miss=float(S.loc[(pid, t)].missed))
        rows.append(f)
D_ = pd.DataFrame(rows)
print(f"\n{len(D_)} player-seasons, outcome years {sorted(D_.t.unique())}")

FEATS_A = ["m1", "m2", "m3", "m2_na", "m3_na", "age", "mpg"]
FEATS_B = FEATS_A + ["h1", "h1_na", "hmaj1", "rec"]
FEATS_C = FEATS_A + ["h1", "h1_na", "h2", "hmaj1", "rec"]


def loso(feats):
    mu, sd = D_[feats].mean(), D_[feats].std().replace(0, 1)
    oos_p, oos_m = pd.Series(index=D_.index, dtype=float), pd.Series(index=D_.index, dtype=float)
    for t in D_.t.unique():
        tr, te = D_[D_.t != t], D_[D_.t == t]
        clf = LogisticRegression(C=0.5, max_iter=3000).fit((tr[feats] - mu) / sd, tr.y)
        reg = Ridge(alpha=5.0).fit((tr[feats] - mu) / sd, tr.miss)
        oos_p[te.index] = clf.predict_proba((te[feats] - mu) / sd)[:, 1]
        oos_m[te.index] = reg.predict((te[feats] - mu) / sd)
    auc = roc_auc_score(D_.y, oos_p)
    rmse = float(np.sqrt(np.mean((D_.miss - oos_m) ** 2)))
    return auc, rmse, oos_p, oos_m


print("\n=== leave-one-season-out: predicting NEXT season's any-reason missed-games ===")
for name, feats in (("A  any-reason history only (current production)", FEATS_A), ("B  + 1yr health-confirmed + severity + recurring", FEATS_B), ("C  + 2yr health-confirmed + severity + recurring", FEATS_C)):
    auc, rmse, _, _ = loso(feats)
    print(f"{name:52s} AUC {auc:.3f}   RMSE {rmse:.4f}")

# ---------------- anecdote check: does the health-confirmed measure separate real injuries from incidental absences? ----------------
print("\n=== anecdote check: players with a big ANY-reason miss but near-zero health-confirmed (should NOT be flagged as injury risk) ===")
chk = sb[(sb.yr >= 2022) & (sb.missed >= 0.15) & (sb.hmissed.notna()) & (sb.hmissed <= 0.03) & (sb.mpg >= 8)]
names = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").PLAYER_NAME
for _, r in chk.sort_values("missed", ascending=False).head(8).iterrows():
    print(f"  {names.get(r.PLAYER_ID, r.PLAYER_ID):24s} {r.SEASON}  any-reason missed {r.missed:.0%}  health-confirmed {r.hmissed:.0%}  (GP {r.GP:.0f}, {r.mpg:.0f} mpg)")
print("\n=== the reverse: big health-confirmed miss (a real, serious case) ===")
chk2 = sb[(sb.yr >= 2022) & (sb.hmissed.notna()) & (sb.hmissed >= 0.25) & (sb.mpg >= 20)]
for _, r in chk2.sort_values("hmajor" if "hmajor" in chk2 else "hmissed", ascending=False).head(8).iterrows():
    print(f"  {names.get(r.PLAYER_ID, r.PLAYER_ID):24s} {r.SEASON}  any-reason missed {r.missed:.0%}  health-confirmed {r.hmissed:.0%}  major-tier share {r.hmajor:.0%}")

# ---------------- covid diagnostic: can we say ANYTHING about 2020-21? ----------------
print("\n=== covid-era (2020-21) diagnostic ===")
n_2021_reports = len(r5[r5.season == "2020-21"])
print(f"official injury reports covering 2020-21: {n_2021_reports} rows (report data starts Dec 2021 -- 2020-21 predates it entirely; cannot classify 2020-21 absences as injury vs covid vs rest at all)")
# indirect check: does missed-games in 2020-21 predict the FOLLOWING season (2021-22) as well as other adjacent-season pairs do?
pairs = []
for t in range(2013, 2026):
    p2 = sb[(sb.yr == t - 2) & (sb.mpg >= 15) & (sb.GP >= 20)].PLAYER_ID
    prev = sb[(sb.yr == t - 1) & (sb.GP >= 5) & ((sb.mpg >= 15) | sb.PLAYER_ID.isin(p2))]
    for pid in prev.PLAYER_ID:
        if (pid, t) not in S.index or S.loc[(pid, t)].GP < 5:
            continue
        pairs.append((t, float(S.loc[(pid, t - 1)].missed), float(S.loc[(pid, t)].missed >= 0.25)))
PR = pd.DataFrame(pairs, columns=["t", "m1", "y"])
for t in sorted(PR.t.unique()):
    sub = PR[PR.t == t]
    if len(sub) < 30:
        continue
    try:
        a = roc_auc_score(sub.y, sub.m1)
    except Exception:
        a = float("nan")
    flag = "  <- predicts FROM 2020-21" if t == 2021 else ("  <- FROM lockout 2011-12" if t == 2012 else "")
    print(f"  outcome {t}: m1-only AUC {a:.3f} (n={len(sub)}){flag}")
