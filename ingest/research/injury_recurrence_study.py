"""Study 2: does a prior injury to the SAME body part (same side) predict a new one, and does injury history predict games missed next season?

Episodes come from the official reports (5:30 PM slot): a run of listings for the same player and the same body part with gaps of 7 days or less, counted by
the number of game days listed OUT.  A player-season panel (games played, minutes per game, age) comes from the box-score logs.
  Test 1  group level: P(new episode in body-part group G this season | had a G episode in the previous two seasons / only other groups / none)
  Test 2  same side vs the other side (hamstring, calf, ankle, knee, ...): among players with a prior episode in part P on side S, how often is the next
          episode on the same side S versus the opposite side?  (Left and right are otherwise equally likely, so a same-side excess is real recurrence.)
  Test 3  are recurrences longer than first episodes?
  Test 4  next-season games missed: prior games missed alone vs adding injury-history features (rotation players, out of sample)
Run from ingest/:  uv run python research/injury_recurrence_study.py
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import injury_common as C

sys.stdout.reconfigure(encoding="utf-8")
e = pd.read_pickle(C.D / "injury_rows_05pm.pkl")
r = e[(e.kind == "injury") & (e.part != "")].copy()
r["side"] = r.side.fillna("")
r = r.sort_values(["player_key", "part", "gd"])
r["gap"] = r.groupby(["player_key", "part"]).gd.diff().dt.days
r["new"] = (r.gap.isna() | (r.gap > 7)).astype(int)
r["epi"] = r.groupby(["player_key", "part"]).new.cumsum()
r["out"] = (r.status == "Out") * 1
ep = r.groupby(["player_key", "part", "epi"]).agg(start=("gd", "min"), end=("gd", "max"), n_out=("out", "sum"), n_rows=("out", "size"), group=("group", "first"),
                                                     side=("side", lambda s: s[s != ""].mode().iat[0] if (s != "").any() else ""), season=("season", "first"), major=("tier", lambda x: int(x.isin(["major", "recovery"]).any())), tier=("tier", lambda x: x.mode().iat[0]), nature=("nature", lambda x: x.mode().iat[0] if len(x.mode()) else "")).reset_index()
# MERGE fragments: two listings of the same player and body part with NO game played in between are the same absence (he never came back), even if the
# report skipped a few days.  Only a real return (at least one game played) makes the next listing a new episode.
_lg = C.load_logs()
_lg = _lg[_lg.MIN > 0]
_played = {k: np.sort(v.gd.values) for k, v in _lg.groupby("key")}
def _n_played(pk, a, b):
    arr = _played.get(pk)
    if arr is None:
        return 0
    return int(np.searchsorted(arr, np.datetime64(b), side="left") - np.searchsorted(arr, np.datetime64(a), side="right"))
ep = ep.sort_values(["player_key", "part", "start"]).reset_index(drop=True)
merged = []
for (pk, part), grp in ep.groupby(["player_key", "part"], sort=False):
    cur = None
    for row in grp.itertuples(index=False):
        if cur is not None and _n_played(pk, cur["end"], row.start) == 0:
            cur["end"] = max(cur["end"], row.end)
            cur["n_out"] += row.n_out
            cur["n_rows"] += row.n_rows
            cur["major"] = max(cur["major"], row.major)
        else:
            if cur is not None:
                merged.append(cur)
            cur = row._asdict()
    merged.append(cur)
n_before = len(ep)
ep = pd.DataFrame(merged)
print(f"merged listing fragments with no game played in between: {n_before:,} -> {len(ep):,} episodes")
ep["days"] = (ep.end - ep.start).dt.days + 1
MIN_OUT = 3
big = ep[ep.n_out >= MIN_OUT].copy()
print(f"{len(ep):,} episodes ({len(big):,} with {MIN_OUT}+ game days out) for {ep.player_key.nunique():,} players, 2021-22 to 2025-26")
print("episode length (game days out), median and 90th pct:", ep.n_out.median(), ep.n_out.quantile(.9))

# ---------------- player-season panel
g = C.load_logs()
g = g[g.MIN > 0]
pan = g.groupby(["key", "SEASON"]).agg(gp=("MIN", "size"), mpg=("MIN", "mean"), age=("AGE", "max") if "AGE" in g.columns else ("MIN", "size")).reset_index().rename(columns={"key": "player_key", "SEASON": "season"})
SEAS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
sidx = {s: i for i, s in enumerate(SEAS)}
pan = pan[pan.season.isin(SEAS)].copy()
pan["si"] = pan.season.map(sidx)
big["si"] = big.season.map(sidx)
GROUPS = ["ankle_foot", "knee", "leg_soft", "upper", "hip_back", "achilles"]

# ---------------- Test 1: group-level recurrence
rows = []
for t in range(1, 5):                                  # seasons 2022-23 .. 2025-26 (need at least one prior season of reports)
    prev = pan[(pan.si == t - 1) & (pan.gp >= 20) & (pan.mpg >= 15)]
    for pk in prev.player_key:
        pr = big[(big.player_key == pk) & (big.si.between(t - 2, t - 1))]
        cur = big[(big.player_key == pk) & (big.si == t)]
        mp0 = float(prev[prev.player_key == pk].mpg.iat[0])
        for gname in GROUPS:
            rows.append(dict(t=t, player_key=pk, group=gname, mpg=mp0, prior_same=int((pr.group == gname).sum() > 0), n_same=int((pr.group == gname).sum()),
                             prior_other=int((pr.group != gname).sum()), y=int((cur.group == gname).sum() > 0)))
p1 = pd.DataFrame(rows)
print(f"\nTEST 1. New {MIN_OUT}+-game episode in a body-part group during a season, for players who were rotation players (15+ mpg, 20+ GP) the year before "
      f"({p1.player_key.nunique()} players x {len(GROUPS)} groups x seasons)")
p1["hist"] = np.where(p1.prior_same == 1, "prior episode, SAME group", np.where(p1.prior_other > 0, "prior episodes, OTHER groups only", "no prior episodes"))
print(p1.groupby("hist").y.agg(["mean", "size"]).round(3).to_string())
print("by group (rate with same-group history vs other-group history vs none):")
print(p1.groupby(["group", "hist"]).y.mean().unstack().round(3).to_string())
p1["tot"] = (p1.n_same + p1.prior_other).clip(upper=4)
print("controlling for the TOTAL number of prior 3+ game episodes (so 'injury prone overall' is held equal): P(new episode in the group)")
st = p1[p1.tot > 0].assign(hist2=lambda x: np.where(x.prior_same == 1, "SAME group", "other groups only")).groupby(["tot", "hist2"]).y.agg(["mean", "size"]).round(3)
print(st.to_string())
X = pd.DataFrame({"same": p1.prior_same, "tot": p1.tot, "mpg": p1.mpg / 10})
X = pd.concat([X, pd.get_dummies(p1.group, prefix="g", dtype=float)], axis=1)
m = LogisticRegression(C=1e4, max_iter=3000).fit(X, p1.y)
print("logistic (controls: total prior episodes, minutes, group): odds ratio for a prior SAME-group episode = %.2f; each extra prior episode of any kind = %.2f" % (np.exp(m.coef_[0][0]), np.exp(m.coef_[0][1])))

# ---------------- Test 2: same side vs other side, within the same part
print("\nTEST 2. Players with a prior episode in part P on side S: next episode of P on the SAME side vs the OPPOSITE side (any later episode, 3+ game days)")
rows = []
for (pk, part), grp in big[big.side != ""].sort_values("start").groupby(["player_key", "part"]):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - 1):
        a, b = grp.loc[i], grp.loc[i + 1]
        if (b.start - a.end).days <= 730:
            rows.append(dict(part=part, same=int(a.side == b.side), gap=(b.start - a.end).days, n_out=b.n_out))
t2 = pd.DataFrame(rows)
tt = t2.groupby("part").same.agg(["mean", "size"])
tt = tt[tt["size"] >= 25].sort_values("size", ascending=False)
print("   share of next same-part episodes that are on the SAME side (50% = no side effect):")
print(tt.round(3).to_string())
allsame = t2[t2.part.isin(tt.index)]
print(f"   all listed parts pooled: {allsame.same.mean():.3f} same-side over {len(allsame)} pairs; within 6 months of the previous episode: "
      f"{allsame[allsame.gap <= 180].same.mean():.3f} (n={len(allsame[allsame.gap <= 180])})")

# ---------------- Test 3: are recurrences longer?
big = big.sort_values(["player_key", "part", "start"])
big["prev_end"] = big.groupby(["player_key", "part"]).end.shift(1)
big["prev_side"] = big.groupby(["player_key", "part"]).side.shift(1)
big["recur"] = ((big.start - big.prev_end).dt.days <= 365)
big["recur_same"] = big.recur & (big.side == big.prev_side)
print("\nTEST 3. Length (game days out) of episodes: first-in-a-year vs a recurrence of the same part within 12 months")
print(big.groupby(["recur", "recur_same"]).n_out.agg(["mean", "median", "size"]).round(1).to_string())

# ---------------- Test 4: next-season games missed
panel = pan.set_index(["player_key", "si"])
rows = []
for t in range(2, 5):                                  # predict season t from t-1 and t-2 (2023-24 .. 2025-26)
    for (pk, si), row in panel.xs(t - 1, level="si", drop_level=False).iterrows() if False else []:
        pass
d = pan.copy()
d["missed"] = (82 - d.gp).clip(lower=0)
lag = d.set_index(["player_key", "si"])
out = []
for t in range(2, 5):
    for pk in d[(d.si == t - 1) & (d.gp >= 20) & (d.mpg >= 18)].player_key:
        cur = lag.loc[(pk, t)] if (pk, t) in lag.index else None
        if cur is None:
            continue
        prev1 = lag.loc[(pk, t - 1)]
        prev2 = lag.loc[(pk, t - 2)] if (pk, t - 2) in lag.index else None
        b = big[(big.player_key == pk) & (big.si.between(t - 2, t - 1))]
        b1 = big[(big.player_key == pk) & (big.si == t - 1)]
        rep = b.groupby("part").size()
        out.append(dict(t=t, y=float(cur.missed), m1=float(prev1.missed), m2=float(prev2.missed) if prev2 is not None else float(prev1.missed), age=float(prev1.age) if "age" in prev1 else 27.0,
                        mpg=float(prev1.mpg), n_ep=len(b), n_ep1=len(b1), n_big=int((b.n_out >= 10).sum()), out_days=float(b.n_out.sum()), repeats=int((rep >= 2).sum()),
                        repeat_parts=int(rep[rep >= 2].sum()) if len(rep) else 0, major=int(b.part.str.contains("acl|achilles|surgery", regex=True).sum())))
p4 = pd.DataFrame(out)
print(f"\nTEST 4. Games missed next season, rotation players (18+ mpg, 20+ GP the year before): {len(p4)} player-seasons, train 2023-24/2024-25, test 2025-26")
tr, te = p4[p4.t <= 3], p4[p4.t == 4]
def ev(cols, name):
    mdl = Ridge(alpha=5.0).fit(tr[cols], tr.y)
    pr = mdl.predict(te[cols]).clip(0, 82)
    print(f"   {name:52s} MAE {np.abs(te.y - pr).mean():5.2f} games  R2 {1 - ((te.y - pr) ** 2).sum() / ((te.y - te.y.mean()) ** 2).sum():.3f}")
    return mdl
print(f"   (naive: everyone misses {tr.y.mean():.1f} games -> MAE {np.abs(te.y - tr.y.mean()).mean():.2f})")
ev(["m1"], "last season's games missed")
ev(["m1", "m2"], "+ the season before")
ev(["m1", "m2", "age", "mpg"], "+ age, minutes")
ev(["m1", "m2", "age", "mpg", "n_ep", "out_days"], "+ number of injury episodes and game days out (reports)")
ev(["m1", "m2", "age", "mpg", "n_ep", "out_days", "repeats", "repeat_parts", "major"], "+ repeat body parts, major injuries")
mm = ev(["m1", "m2", "age", "mpg", "n_ep", "out_days", "repeats", "repeat_parts", "major"], "(final)")
for c, b in zip(["m1", "m2", "age", "mpg", "n_ep", "out_days", "repeats", "repeat_parts", "major"], mm.coef_):
    print(f"      {c:14s} {b:+.3f} games")
p4.to_pickle(C.D / "injury_next_season.pkl")
ep.to_pickle(C.D / "injury_episodes.pkl")
