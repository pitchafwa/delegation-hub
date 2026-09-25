"""Study 1: how likely is a player to PLAY, given the official status AND what the injury is?

Data: 5 seasons of official NBA injury reports (pull_injury_reports_hourly.py), joined to box-score logs (played = minutes > 0 that date).
Each row = one listed player for one game.  Uses the 5:30 PM ET report (what the afternoon plan refresh sees) unless noted.
Features: status; reason (tier: minor / moderate / major / recovery / management, body-part group); how long he has been listed for this trouble (episode
length); whether he was Out the last time he was listed (returning); minutes per game in his previous 15 games (rotation vs bench).
Run from ingest/:  uv run python research/injury_playrate_study.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import injury_common as C

sys.stdout.reconfigure(encoding="utf-8")
d = C.load_reports()
g = C.load_logs()
g = g[g.MIN > 0].sort_values(["key", "gd"])
played = g.groupby(["key", "gd"]).size().reset_index()[["key", "gd"]]
played["played"] = 1
# rotation: mean minutes over the player's previous 15 games (before the game date), same or previous season
g["mp15"] = g.groupby("key").MIN.transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())
mp = g.groupby(["key", "gd"]).mp15.last().reset_index()


def season_of(x):
    return f"{x.year if x.month >= 8 else x.year - 1}"


d = d.merge(played, left_on=["player_key", "gd"], right_on=["key", "gd"], how="left").drop(columns=["key"])
d["played"] = d.played.fillna(0)
d = d.sort_values(["player_key", "gd", "slot"]).reset_index(drop=True)
# mp15 as of the last game BEFORE gd
mp = mp.rename(columns={"key": "player_key", "gd": "mgd"}).sort_values("mgd")
d = pd.merge_asof(d.sort_values("gd"), mp, left_on="gd", right_on="mgd", by="player_key", direction="backward", allow_exact_matches=False)
d["rot"] = np.where(d.mp15 >= 20, "rotation", np.where(d.mp15.notna(), "bench", "unknown"))

# ---- episodes: consecutive listings of the same player for the same trouble (gap of more than 7 days ends an episode)
e = d[d.slot == "05PM"].sort_values(["player_key", "game_date"]).copy()
e["gap"] = e.groupby("player_key").gd.diff().dt.days
e["new_ep"] = (e.gap.isna() | (e.gap > 7)).astype(int)
e["ep"] = e.groupby("player_key").new_ep.cumsum()
e["ep_n"] = e.groupby(["player_key", "ep"]).cumcount() + 1             # how many listings so far in this episode (1 = first day)
e["prev_status"] = e.groupby("player_key").status.shift(1)
e.loc[e.gap.isna() | (e.gap > 7), "prev_status"] = None
e["returning"] = (e.prev_status == "Out").astype(int)
e["first_day"] = (e.ep_n == 1).astype(int)
e["prev_played"] = e.groupby("player_key").played.shift(1)
e.loc[e.gap.isna() | (e.gap > 7), "prev_played"] = np.nan
e.to_pickle(C.D / "injury_rows_05pm.pkl")
inj = e[e.kind.isin(["injury", "illness"]) & e.status.isin(["Questionable", "Probable", "Doubtful", "Available"])].copy()
print(f"{len(e):,} report rows (5:30 PM); {len(inj):,} injury/illness rows with a play-or-not designation")


def show(title, df, by, minn=150):
    t = df.groupby(by).played.agg(["mean", "size"])
    t = t[t["size"] >= minn]
    print(f"\n{title}")
    print(t.assign(mean=t["mean"].round(3)).to_string())


show("P(play) by status, rotation vs bench (all seasons)", inj, ["status", "rot"])
q = inj[inj.status == "Questionable"]
show("QUESTIONABLE: by injury tier (rotation players only)", q[q.rot == "rotation"], "tier")
show("QUESTIONABLE: by body-part group (rotation players)", q[q.rot == "rotation"], "group")
show("QUESTIONABLE: by tier x first listing vs continuing (rotation)", q[q.rot == "rotation"], ["tier", "first_day"], 100)
show("QUESTIONABLE: by number of listings in this episode (rotation)", q[q.rot == "rotation"].assign(epb=lambda x: pd.cut(x.ep_n, [0, 1, 2, 4, 8, 100], labels=["1", "2", "3-4", "5-8", "9+"])), "epb")
show("QUESTIONABLE: was Out the last time he was listed (returning) (rotation)", q[q.rot == "rotation"], "returning")
show("QUESTIONABLE: missed his last game or not (rotation, continuing episodes)", q[(q.rot == "rotation") & q.prev_played.notna()], "prev_played")
p = inj[inj.status == "Probable"]
show("PROBABLE: by tier (rotation)", p[p.rot == "rotation"], "tier")
dd = inj[inj.status == "Doubtful"]
show("DOUBTFUL: by tier (all)", dd, "tier", 80)
show("Specific: QUESTIONABLE by (group, tier), rotation, n>=120", q[q.rot == "rotation"], ["group", "tier"], 120)

# ---- does knowing the injury add predictive information? out-of-sample logistic model
base_feats = ["Q", "D", "P", "rot"]
X = pd.DataFrame(index=inj.index)
X["Q"] = (inj.status == "Questionable") * 1.0
X["D"] = (inj.status == "Doubtful") * 1.0
X["P"] = (inj.status == "Probable") * 1.0
X["rot"] = (inj.rot == "rotation") * 1.0
X["unk"] = (inj.rot == "unknown") * 1.0
for t in ["minor", "moderate", "major", "recovery", "management"]:
    X["t_" + t] = (inj.tier == t) * 1.0
for gr in ["ankle_foot", "knee", "leg_soft", "upper", "hip_back", "head", "achilles", "illness"]:
    X["g_" + gr] = (inj.group == gr) * 1.0
X["first"] = inj.first_day * 1.0
X["logep"] = np.log(inj.ep_n)
X["ret"] = inj.returning * 1.0
X["prevmiss"] = (inj.prev_played == 0) * 1.0
X["q_x_major"] = X.Q * X.t_major
X["q_x_minor"] = X.Q * X.t_minor
X["q_x_rot"] = X.Q * X.rot
X["q_x_first"] = X.Q * X["first"]
y = inj.played.to_numpy()
train = (inj.season <= "2023-24").to_numpy()
test = ~train


def fit(cols, name):
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X.loc[train, cols], y[train])
    pr = m.predict_proba(X.loc[test, cols])[:, 1]
    ll = -np.mean(y[test] * np.log(pr) + (1 - y[test]) * np.log(1 - pr))
    br = np.mean((pr - y[test]) ** 2)
    print(f"   {name:46s} log loss {ll:.4f}  Brier {br:.4f}")
    return m, pr


print(f"\nOUT-OF-SAMPLE (train 2021-22..2023-24: {train.sum():,} rows; test 2024-25..2025-26: {test.sum():,} rows)")
fit([], "constant") if False else None
m0, p0 = fit(["Q", "D", "P"], "status only")
m1, p1 = fit(["Q", "D", "P", "rot", "unk", "q_x_rot"], "+ rotation vs bench")
m2, p2 = fit(["Q", "D", "P", "rot", "unk", "q_x_rot", "t_minor", "t_moderate", "t_major", "t_recovery", "t_management", "q_x_major", "q_x_minor"], "+ injury tier")
cols3 = ["Q", "D", "P", "rot", "unk", "q_x_rot", "t_minor", "t_moderate", "t_major", "t_recovery", "t_management", "q_x_major", "q_x_minor"] + [c for c in X.columns if c.startswith("g_")]
m3, p3 = fit(cols3, "+ body part")
cols4 = cols3 + ["first", "logep", "ret", "prevmiss", "q_x_first"]
m4, p4 = fit(cols4, "+ how long listed / returning")
print("\nfull-model coefficients (log-odds of playing):")
for c, b in sorted(zip(cols4, m4.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"   {c:14s} {b:+.2f}")
X.assign(played=y, season=inj.season.values).to_pickle(C.D / "injury_model_X.pkl")
