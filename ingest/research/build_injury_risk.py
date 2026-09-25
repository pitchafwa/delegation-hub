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
THRESH = 0.30
PCTS = [0.15, 0.35, 0.65, 0.85]        # tiers are RELATIVE to other rotation players: bottom 15%, next 20%, middle 30%, next 20%, top 15%
LABELS = ["Low", "Below typical", "Typical", "Elevated", "High"]
sb = pd.read_csv(C.D / "player_season_base.csv")
sb["yr"] = sb.SEASON.str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["len"] = np.where(sb.yr.isin([2019, 2020]), 72, 82)
sb["missed"] = (1 - sb.GP / sb.len).clip(lower=0)
sb["mpg"] = sb.MIN / sb.GP
S = sb.set_index(["PLAYER_ID", "yr"])
LAST = int(max(y for y in sb.yr.unique() if sb[sb.yr == y].GP.max() >= 70))      # latest COMPLETED season (2025 = 2025-26); we predict the one after
BASE = ["m1", "m2", "m3", "age", "mpg", "m2_na", "m3_na", "mn3", "mean3", "n40"]      # mn3/mean3/n40 = persistence: a player bad in EVERY recent season (Embiid) vs one bad once


def feats(pid, t, age_now=None):
    """features for predicting season t from seasons before it; a player with no row for t-1 but earlier seasons is treated as having missed t-1 entirely"""
    if (pid, t - 1) in S.index:
        r = S.loc[(pid, t - 1)]
        m1, age, mpg = r.missed, r.AGE + 1, r.mpg
    else:
        older = [k for k in range(2, 8) if (pid, t - k) in S.index]
        if not older:
            return None
        r = S.loc[(pid, t - older[0])]
        m1, age, mpg = 1.0, r.AGE + older[0], r.mpg
    m2 = S.loc[(pid, t - 2)].missed if (pid, t - 2) in S.index else np.nan
    m3 = S.loc[(pid, t - 3)].missed if (pid, t - 3) in S.index else np.nan
    return dict(m1=m1, m2=m2, m3=m3, age=age_now or age, mpg=mpg)


# ---- training: outcome seasons 2022-23 .. 2025-26
rows = []
for t in range(2022, LAST + 1):
    p2 = sb[(sb.yr == t - 2) & (sb.mpg >= 15) & (sb.GP >= 20)].PLAYER_ID
    prev = sb[(sb.yr == t - 1) & (sb.GP >= 5) & ((sb.mpg >= 15) | sb.PLAYER_ID.isin(p2))]
    for pid in prev.PLAYER_ID:
        if (pid, t) not in S.index or S.loc[(pid, t)].GP < 5:
            continue
        f = feats(pid, t)
        f.update(y=int(S.loc[(pid, t)].missed >= THRESH), miss=float(S.loc[(pid, t)].missed))
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
print(f"trained on {len(D)} player-seasons; coefficients:", {c: round(b, 2) for c, b in zip(BASE, clf.coef_[0])})

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
for pid in pids:
    t = LAST + 1
    # only players who are current (played in the last two seasons)
    if not ((pid, LAST) in S.index or (pid, LAST - 1) in S.index):
        continue
    f = feats(pid, t)
    if f is None or f["mpg"] < 15:      # bench players' games missed are mostly coaching decisions, so they get no injury tier
        continue
    X = pd.DataFrame([{**f, "m2_na": float(np.isnan(f["m2"])), "m3_na": float(np.isnan(f["m3"]))}])
    X["m2"] = X.m2.fillna(X.m1)
    X["m3"] = X.m3.fillna(X.m1)
    X["mn3"] = X[["m1", "m2", "m3"]].min(axis=1)
    X["mean3"] = X[["m1", "m2", "m3"]].mean(axis=1)
    X["n40"] = (X[["m1", "m2", "m3"]] >= 0.4).sum(axis=1)
    Z = (X[BASE] - mu) / sd
    p = float(clf.predict_proba(Z)[:, 1][0])
    exp_missed = float(np.clip(reg.predict(Z)[0], 0, 0.9) * 82)
    e = ep[(ep.pid == pid) & (ep.yr >= LAST - 2) & (ep.n_out >= 3)]         # three seasons of reports for recurrence tags
    tags = []
    for (part, side), g in e[e.side != ""].groupby(["part", "side"]):
        if len(g) >= 2:
            tags.append(f"recurring {side} {part} x{len(g)}")
    for part, g in e[e.side == ""].groupby("part"):
        if len(g) >= 3:
            tags.append(f"recurring {part} x{len(g)}")
    if float(X["mn3"].iat[0]) >= 0.40:
        tags.append("missed 40%+ of games in each of the last 3 seasons")
    bad = [k for k in range(0, 5) if (pid, LAST - k) in S.index and S.loc[(pid, LAST - k)].missed >= 0.25]
    seen = [k for k in range(0, 5) if (pid, LAST - k) in S.index]
    if len(bad) >= 3:
        tags.append(f"history: missed 25%+ of games in {len(bad)} of the last {len(seen)} seasons")
    late = ep[(ep.pid == pid) & (ep.yr == LAST) & (ep.end >= last_day - pd.Timedelta(days=10)) & (ep.n_out >= 5)]
    if len(late) and ((late.major == 1).any() or (late.n_out >= 25).any()):
        lp = late.sort_values("n_out").iloc[-1]
        tags.append(f"returning: still out at end of last season ({(lp.side + ' ') if lp.side else ''}{lp.part})")
    out.append(dict(PLAYER_ID=pid, injury_p=round(p, 3), injury_tier=-1, injury_label="", injury_missed=round(exp_missed, 0),
                    m1=round(f["m1"], 2), age=round(f["age"], 1), tags=json.dumps(tags)))
R = pd.DataFrame(out)
_cuts = R.injury_p.quantile(PCTS).tolist()
R["injury_tier"] = np.digitize(R.injury_p, _cuts)
R["injury_label"] = R.injury_tier.map(dict(enumerate(LABELS)))
print("tier cut points on P(miss 30%+):", [round(c, 3) for c in _cuts], "| typical player:", round(R.injury_p.median(), 3), "expected games missed", R.injury_missed.median())
_lastseason = sb[(sb.yr == LAST) & (sb.mpg >= 15) & (sb.GP >= 5)]
json.dump({"cuts": _cuts, "labels": LABELS, "season": f"{LAST}-{str(LAST + 1)[2:]}", "n_rotation": int(len(_lastseason)),
           "typical_missed": float(round(_lastseason.missed.median() * 82)), "typical_share_25plus": float(round((_lastseason.missed >= 0.30).mean(), 3))},
          open(C.D / "injury_risk_meta.json", "w"))
print("last completed season, rotation players (15+ mpg): median games missed", round(_lastseason.missed.median() * 82), "| share missing 25+ games", round((_lastseason.missed >= 0.30).mean(), 3))
R.to_csv(C.D / "injury_risk_v4.csv", index=False)
print(f"wrote injury_risk_v4.csv for {len(R)} players; tier counts:\n{R.injury_label.value_counts().reindex(LABELS).to_string()}")
print("tagged recurring:", (R.tags.str.contains("recurring")).sum(), " returning:", (R.tags.str.contains("returning")).sum())
names = {"Tatum": "Jayson Tatum", "Haliburton": "Tyrese Haliburton", "Embiid": "Joel Embiid", "Wemby": "Victor Wembanyama", "LeBron": "LeBron James", "Kawhi": "Kawhi Leonard",
         "Jokic": "Nikola Jokic", "Curry": "Stephen Curry", "Edey": "Zach Edey", "Kessler": "Walker Kessler", "Irving": "Kyrie Irving"}
nm = pd.read_csv(C.D / "player_season_base.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").PLAYER_NAME
R["name"] = R.PLAYER_ID.map(nm)
print(R[R.name.isin(names.values())][["name", "injury_label", "injury_p", "injury_missed", "m1", "age", "tags"]].to_string(index=False))
