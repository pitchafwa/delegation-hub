"""Five-tier injury risk for every player on the board, plus material-injury tags.  Writes data/injury_risk_v4.csv (read by build_hub_data.py).

TIER = predicted chance of missing 30%+ of next season's games, from a logistic model on games-missed history (last 3 seasons), age and minutes per game.
  Out-of-sample (leave-one-season-out, 2022-23 to 2025-26, 1,362 player-seasons): AUC 0.686 vs 0.628 for the old rule (count of seasons with 25%+ missed);
  calibrated (top quintile predicted 73% -> actual 72%).  Injury DETAILS (episode counts, repeats, surgeries, still-out-at-season-end) did not improve the
  season-level prediction beyond history (AUC 0.680-0.686), so they are shown as TAGS, not folded into the score:
    recurring  same body part on the same side, 2+ episodes of 3+ game days in the last two seasons (a prior same-group episode roughly doubles the chance of another;
               same-side recurrences last about 5 games longer)
    returning  still listed out (5+ days, major injury or 25+ days) when last season ended: 31% missed half of the first ~16 games vs 18% otherwise
Players with no games last season are treated as having missed all of them.  Tiers: 0 Low, 1 Below average, 2 Average, 3 Elevated, 4 High.
Run from ingest/ (after injury_recurrence_study.py has written injury_episodes.pkl):  uv run python research/build_injury_risk.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import injury_common as C

sys.stdout.reconfigure(encoding="utf-8")
THRESH_H = 0.20                        # health model: chance of being listed OUT for an injury or illness in 20%+ of games (about 16)
THRESH_T = 0.30                        # availability model (games for the GP projection): chance of missing 30%+ of games for ANY reason
PCTS = [0.15, 0.35, 0.65, 0.85]        # tiers are RELATIVE to other rotation players: bottom 15%, next 20%, middle 30%, next 20%, top 15%
LABELS = ["Low", "Below typical", "Typical", "Elevated", "High"]
sb = pd.read_csv(C.D / "player_season_base.csv")
sb["yr"] = sb.SEASON.str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["len"] = np.where(sb.yr.isin([2019, 2020]), 72, 82)
sb["missed"] = (1 - sb.GP / sb.len).clip(lower=0)          # ANY reason: rest, coach's decision, not on a roster, injury (the old measure; kept only for the games-played projection)
sb["mpg"] = sb.MIN / sb.GP

# ---- CONFIRMED health absences: game days the player was listed OUT/DOUBTFUL on the official injury report for an injury, illness or return-from-injury reason.
# A game a fringe player simply did not play (coach's decision, G League, two-way) is NOT counted. Reports exist from Dec 2021, so full seasons start with 2022-23.
_r5 = pd.read_pickle(C.D / "injury_rows_05pm.pkl")
_h = _r5[_r5.status.isin(["Out", "Doubtful"]) & _r5.kind.isin(["injury", "illness", "mgmt"])].drop_duplicates(["player_key", "game_date"]).copy()
_lg = pd.read_csv(C.D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
_lg["player_key"] = _lg.PLAYER_NAME.map(C.key_of_log)
_k2id = _lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
_h["pid"] = _h.player_key.map(_k2id)
_h = _h.dropna(subset=["pid"]).copy()
_h["pid"] = _h.pid.astype(int)
_h["yr"] = _h.season.str[:4].astype(int)
HM = _h.groupby(["pid", "yr"]).size().to_dict()
sb["hmissed"] = [(min(HM.get((int(a), int(b)), 0) / l, 1.0) if b >= 2022 else np.nan) for a, b, l in zip(sb.PLAYER_ID, sb.yr, sb.len)]
S = sb.set_index(["PLAYER_ID", "yr"])
LAST = int(max(y for y in sb.yr.unique() if sb[sb.yr == y].GP.max() >= 70))      # latest COMPLETED season (2025 = 2025-26); we predict the one after
BASE = ["m1", "m2", "m3", "age", "mpg", "m2_na", "m3_na", "mn3", "mean3", "n40"]      # mn3/mean3/n40 = persistence: a player bad in EVERY recent season (Embiid) vs one bad once


def gm(pid, y, col):
    """his missed rate in season y on this measure; NaN when unknown (no data for that season)"""
    if (pid, y) in S.index:
        return float(S.loc[(pid, y)][col])
    if col == "hmissed" and y >= 2022 and (pid, y) in HM:          # listed hurt all season without playing
        return min(HM[(pid, y)] / 82.0, 1.0)
    return np.nan


def feats(pid, t, col, age_now=None):
    """features for predicting season t from seasons before it. A player with no games in t-1 is 'missed everything' on the availability measure, but on the health measure only if
    he was actually listed hurt (otherwise unknown -> no rating)"""
    if (pid, t - 1) in S.index:
        r = S.loc[(pid, t - 1)]
        m1, age, mpg = float(r[col]), r.AGE + 1, r.mpg
        if np.isnan(m1):
            return None
    else:
        older = [k for k in range(2, 8) if (pid, t - k) in S.index]
        if not older:
            return None
        r = S.loc[(pid, t - older[0])]
        age, mpg = r.AGE + older[0], r.mpg
        if col == "missed":
            m1 = 1.0
        else:
            m1 = gm(pid, t - 1, col)
            if np.isnan(m1):
                return None
    return dict(m1=m1, m2=gm(pid, t - 2, col), m3=gm(pid, t - 3, col), age=age_now or age, mpg=mpg)


def rotation_now(pid, t):
    """a real rotation player recently: 20+ games at 15+ mpg in one of the last two seasons (a one-game 17-minute cameo does not count)"""
    return any((pid, y) in S.index and S.loc[(pid, y)].GP >= 20 and S.loc[(pid, y)].mpg >= 15 for y in (t - 1, t - 2))


def fit(col, thresh, first_t):
    rows = []
    for t in range(first_t, LAST + 1):
        p2 = sb[(sb.yr == t - 2) & (sb.mpg >= 15) & (sb.GP >= 20)].PLAYER_ID
        prev = sb[(sb.yr == t - 1) & (sb.GP >= 5) & ((sb.mpg >= 15) | sb.PLAYER_ID.isin(p2))]
        for pid in prev.PLAYER_ID:
            if (pid, t) not in S.index or S.loc[(pid, t)].GP < 5 or np.isnan(S.loc[(pid, t)][col]):
                continue
            f = feats(pid, t, col)
            if f is None:
                continue
            f.update(y=int(S.loc[(pid, t)][col] >= thresh), miss=float(S.loc[(pid, t)][col]), t=t)
            rows.append(f)
    D = pd.DataFrame(rows)
    for c in ("m2", "m3"):
        D[c + "_na"] = D[c].isna() * 1.0
        D[c] = D[c].fillna(D.m1)
    D["mn3"] = D[["m1", "m2", "m3"]].min(axis=1)
    D["mean3"] = D[["m1", "m2", "m3"]].mean(axis=1)
    D["n40"] = (D[["m1", "m2", "m3"]] >= 0.4).sum(axis=1)
    mu, sd = D[BASE].mean(), D[BASE].std().replace(0, 1)
    clf = LogisticRegression(C=0.5, max_iter=3000).fit((D[BASE] - mu) / sd, D.y)
    reg = Ridge(alpha=5.0).fit((D[BASE] - mu) / sd, D.miss)
    from sklearn.metrics import roc_auc_score
    oos = pd.Series(index=D.index, dtype=float)
    for tt in D.t.unique():
        tr, te = D[D.t != tt], D[D.t == tt]
        m_ = LogisticRegression(C=0.5, max_iter=3000).fit((tr[BASE] - mu) / sd, tr.y)
        oos[te.index] = m_.predict_proba((te[BASE] - mu) / sd)[:, 1]
    top = D.y[oos >= oos.quantile(0.8)].mean()
    print(f"[{col}] leave-one-season-out AUC {roc_auc_score(D.y, oos):.3f} (last-season rate alone: {roc_auc_score(D.y, D.m1):.3f}); top quintile predicted {oos[oos >= oos.quantile(0.8)].mean():.2f}, actual {top:.2f}")
    print(f"[{col}] trained on {len(D)} player-seasons (base rate {D.y.mean():.3f} for >= {thresh:.0%} missed); coefficients:", {c: round(b, 2) for c, b in zip(BASE, clf.coef_[0])})
    return clf, reg, mu, sd, D


def score(pid, t, col, model):
    clf, reg, mu, sd, _ = model
    f = feats(pid, t, col)
    if f is None or f["mpg"] < 15 or not rotation_now(pid, t):     # bench / non-rotation players get no rating: their missed games are mostly coaching decisions
        return None
    X = pd.DataFrame([{**f, "m2_na": float(np.isnan(f["m2"])), "m3_na": float(np.isnan(f["m3"]))}])
    X["m2"] = X.m2.fillna(X.m1)
    X["m3"] = X.m3.fillna(X.m1)
    X["mn3"] = X[["m1", "m2", "m3"]].min(axis=1)
    X["mean3"] = X[["m1", "m2", "m3"]].mean(axis=1)
    X["n40"] = (X[["m1", "m2", "m3"]] >= 0.4).sum(axis=1)
    Z = (X[BASE] - mu) / sd
    return dict(p=float(clf.predict_proba(Z)[:, 1][0]), miss=float(np.clip(reg.predict(Z)[0], 0, 0.9) * 82), f=f, X=X)


MH = fit("hmissed", THRESH_H, 2023)      # health-confirmed model: seasons 2023-24 .. 2025-26 as outcomes (2022-23 is the first season with a full year of reports)
MT = fit("missed", THRESH_T, 2022)       # availability model: total games missed for any reason (feeds the expected-games-played projection only)

# ---- episodes -> tags
ep = pd.read_pickle(C.D / "injury_episodes.pkl")
lg = pd.read_csv(C.D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
lg["player_key"] = lg.PLAYER_NAME.map(C.key_of_log)
k2id = lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
ep["pid"] = ep.player_key.map(k2id)
ep = ep.dropna(subset=["pid"]).copy()
ep["pid"] = ep.pid.astype(int)
ep["yr"] = ep.season.str[:4].astype(int)
rows_ = pd.read_pickle(C.D / "injury_rows_05pm.pkl")
last_day = rows_[rows_.season == f"{LAST}-{str(LAST + 1)[2:]}"].gd.max()

out = []
pids = sb[sb.yr >= LAST - 6].PLAYER_ID.unique()
t = LAST + 1
for pid in pids:
    if not ((pid, LAST) in S.index or (pid, LAST - 1) in S.index):        # only players who are current (played in the last two seasons)
        continue
    h = score(pid, t, "hmissed", MH)
    if h is None:
        continue
    a = score(pid, t, "missed", MT)
    f, X = h["f"], h["X"]
    e = ep[(ep.pid == pid) & (ep.yr >= LAST - 2) & (ep.n_out >= 3)]         # three seasons of reports for recurrence tags
    tags = []
    for (part, side), g in e[e.side != ""].groupby(["part", "side"]):
        if len(g) >= 2:
            tags.append(f"recurring {side} {part} x{len(g)}")
    for part, g in e[e.side == ""].groupby("part"):
        if len(g) >= 3:
            tags.append(f"recurring {part} x{len(g)}")
    if float(X["mn3"].iat[0]) >= 0.40 and not X["m2_na"].iat[0] and not X["m3_na"].iat[0]:
        tags.append("listed out with an injury or illness in 40%+ of games in each of the last 3 seasons")
    seen = [k for k in range(0, 4) if (pid, LAST - k) in S.index and not np.isnan(S.loc[(pid, LAST - k)].hmissed)]
    bad = [k for k in seen if S.loc[(pid, LAST - k)].hmissed >= 0.25]
    if len(bad) >= 3:
        tags.append(f"history: listed out with an injury or illness in 25%+ of games in {len(bad)} of the last {len(seen)} seasons")
    late = ep[(ep.pid == pid) & (ep.yr == LAST) & (ep.end >= last_day - pd.Timedelta(days=10)) & (ep.n_out >= 5)]
    if len(late) and ((late.major == 1).any() or (late.n_out >= 25).any()):
        lp = late.sort_values("n_out").iloc[-1]
        tags.append(f"returning: still out at end of last season ({(lp.side + ' ') if lp.side else ''}{lp.part})")
    out.append(dict(PLAYER_ID=pid, injury_p=round(h["p"], 3), injury_tier=-1, injury_label="", injury_health_missed=round(h["miss"], 0), injury_missed=round((a or h)["miss"], 0),
                    m1=round(f["m1"], 2), age=round(f["age"], 1), tags=json.dumps(tags)))
R = pd.DataFrame(out)
_cuts = R.injury_p.quantile(PCTS).tolist()
R["injury_tier"] = np.digitize(R.injury_p, _cuts)
R["injury_label"] = R.injury_tier.map(dict(enumerate(LABELS)))
print("tier cut points on P(listed out 20%+):", [round(c, 3) for c in _cuts], "| typical player:", round(R.injury_p.median(), 3), "expected health games missed", R.injury_health_missed.median(), "| any-reason games missed", R.injury_missed.median())
_lastseason = sb[(sb.yr == LAST) & (sb.mpg >= 15) & (sb.GP >= 20)]
json.dump({"cuts": _cuts, "labels": LABELS, "season": f"{LAST}-{str(LAST + 1)[2:]}", "n_rotation": int(len(_lastseason)), "basis": "health",
           "typical_missed": float(round(_lastseason.hmissed.median() * 82)), "typical_share_25plus": float(round((_lastseason.hmissed >= THRESH_H).mean(), 3)), "threshold": THRESH_H},
          open(C.D / "injury_risk_meta.json", "w"))
print("last completed season, rotation players (15+ mpg, 20+ GP): median games listed out for injury/illness", round(_lastseason.hmissed.median() * 82), "| share at 20%+", round((_lastseason.hmissed >= THRESH_H).mean(), 3))
R.to_csv(C.D / "injury_risk_v4.csv", index=False)
print("wrote injury_risk_v4.csv for", len(R), "players; tier counts:")
print(R.injury_label.value_counts().reindex(LABELS).to_string())
print("tagged recurring:", (R.tags.str.contains("recurring")).sum(), " returning:", (R.tags.str.contains("returning")).sum())
