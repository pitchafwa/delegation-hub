"""Follow-up to injury_discount_retest.py (2026-09-27): that test asked whether injury history predicts next season's GAMES MISSED. Tommy's real
question was narrower: does an injury predict a drop in a player's RATE of production (points per minute, i.e. efficiency/role/explosiveness) the
season after, beyond ordinary aging -- separate from whether he plays fewer games. Tests that directly.

Outcome: next season's fantasy points per minute (his rate when he IS on the floor), for players who stayed rotation-caliber (20+ mpg, 40+ GP)
in both seasons (so this isn't just "hurt players play worse minutes/role" showing up as a rate drop by another name -- these are all real,
sustained rotation seasons on both sides). Predictors: this season's own rate (the obvious baseline -- rates persist), age, plus injury features:
whether he had a real (health-confirmed) absence this season, its severity (major-tier share), whether it was recent (still hurt near season's end
vs resolved with time to spare), and a recurring-injury flag.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F
import injury_common as C

sys.stdout.reconfigure(encoding="utf-8")
D = C.D

g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
g = g[g.season != "2025-26"].copy() if False else g  # keep 2025-26 as a usable OUTCOME season; just not a target for pulling injury FEATURES beyond it
g["fp"] = F.fp(g)

season_order = sorted(g.season.unique())
nxt = {s: season_order[i + 1] for i, s in enumerate(season_order[:-1])}

per = g.groupby(["pid", "season"]).agg(min_sum=("min", "sum"), fp_sum=("fp", "sum"), gp=("min", "size"), mpg=("min", "mean")).reset_index()
per = per[(per.gp >= 40) & (per.mpg >= 20)].copy()
per["rate"] = per.fp_sum / per.min_sum
RATE = per.set_index(["pid", "season"]).rate.to_dict()
MPG = per.set_index(["pid", "season"]).mpg.to_dict()
GOOD = set(zip(per.pid, per.season))

# ---------------- health-confirmed injury features per player-season (reason text -> severity tier; only 2022-23+ has report coverage)
r5 = pd.read_pickle(D / "injury_rows_05pm.pkl")
h = r5[r5.status.isin(["Out", "Doubtful"]) & r5.kind.isin(["injury", "illness", "mgmt"])].drop_duplicates(["player_key", "game_date"]).copy()
lg = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
lg["player_key"] = lg.PLAYER_NAME.map(C.key_of_log)
k2id = lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
h["pid"] = h.player_key.map(k2id)
h = h.dropna(subset=["pid"]).copy()
h["pid"] = h.pid.astype(int)
h["tier"] = h.reason.map(lambda x: C.parse_reason(x)["tier"])
h["gd"] = pd.to_datetime(h.game_date)
season_end = g.groupby("season").date.max().to_dict()

feat = {}
for (pid, season), sub in h.groupby(["pid", "season"]):
    n = len(sub)
    major = (sub.tier == "major").mean()
    end = season_end.get(season)
    late = (end is not None) and (sub.gd.max() >= end - pd.Timedelta(days=14))     # still missing games in the last 2 weeks of the season
    feat[(pid, season)] = dict(n_out=n, major=major, late=int(late))

# recurring flag from episodes (2+ same-part-and-side episodes, trailing 2 seasons of report coverage)
ep = pd.read_pickle(D / "injury_episodes.pkl")
ep["pid"] = ep.player_key.map(k2id)
ep = ep.dropna(subset=["pid"]).copy()
ep["pid"] = ep.pid.astype(int)
season_ix = {s: i for i, s in enumerate(season_order)}


def recurring(pid, season, lookback=2):
    ix = season_ix.get(season)
    if ix is None:
        return 0
    win = {s for s in season_order if 0 <= ix - season_ix.get(s, 999) < lookback}
    e = ep[(ep.pid == pid) & (ep.season.isin(win)) & (ep.n_out >= 3)]
    for (part, side), gg in e[e.side != ""].groupby(["part", "side"]):
        if len(gg) >= 2:
            return 1
    for part, gg in e[e.side == ""].groupby("part"):
        if len(gg) >= 3:
            return 1
    return 0


# ---------------- build the panel: season t (has report coverage, 2022-23+) -> outcome t+1 (both rotation-caliber)
rows = []
for pid, season in GOOD:
    if season not in nxt or season < "2022-23":
        continue
    t1 = nxt[season]
    if (pid, t1) not in GOOD:
        continue
    r0, r1 = RATE[(pid, season)], RATE[(pid, t1)]
    f = feat.get((pid, season), dict(n_out=0, major=0.0, late=0))
    rows.append(dict(pid=pid, season=season, rate0=r0, rate1=r1, mpg0=MPG[(pid, season)],
                      hurt=int(f["n_out"] >= 5), n_out=f["n_out"], major=f["major"], late=f["late"], rec=recurring(pid, season)))
P = pd.DataFrame(rows)
print(f"{len(P)} rotation-to-rotation player-season pairs, {P.hurt.sum()} with a real (5+ game) health-confirmed absence in the FIRST season")


def fit_report(name, cols):
    tr = P.dropna(subset=cols + ["rate1"])
    mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
    # leave-one-season-out
    errs = []
    for s in tr.season.unique():
        trn, te = tr[tr.season != s], tr[tr.season == s]
        if len(te) < 15:
            continue
        m = Ridge(alpha=3.0).fit((trn[cols] - mu) / sd, trn.rate1)
        pred = m.predict((te[cols] - mu) / sd)
        errs.append((te.rate1.values - pred))
    e = np.concatenate(errs)
    print(f"{name:60s} RMSE {np.sqrt(np.mean(e**2)):.4f}  (target sd {tr.rate1.std():.4f})")


print("\n=== leave-one-season-out: predicting next season's rate (fp per minute) ===")
fit_report("A: rate0 only (persistence)", ["rate0"])
fit_report("B: rate0 + mpg0", ["rate0", "mpg0"])
fit_report("C: B + hurt flag (5+ game health-confirmed absence)", ["rate0", "mpg0", "hurt"])
fit_report("D: B + n_out + major-tier share + late-season flag + recurring", ["rate0", "mpg0", "n_out", "major", "late", "rec"])

print("\n=== direct comparison: hurt vs not-hurt players' rate CHANGE (rate1 - rate0), unadjusted ===")
for grp, sub in P.groupby("hurt"):
    print(f"  hurt={grp}: n={len(sub)}  mean change {(sub.rate1-sub.rate0).mean():+.4f}  (rate0 mean {sub.rate0.mean():.3f})")

print("\n=== does it matter WHEN the injury resolved? late-season (still out in the last 2 weeks) vs resolved with time to spare ===")
hurt = P[P.hurt == 1]
for grp, sub in hurt.groupby("late"):
    print(f"  still out late in the season={grp}: n={len(sub)}  mean change {(sub.rate1-sub.rate0).mean():+.4f}")

print("\n=== does severity (major-tier share) matter, among the hurt group? ===")
hurt2 = hurt.assign(sev=pd.cut(hurt.major, [-0.01, 0.01, 0.5, 1.01], labels=["none/minor", "mixed", "mostly major"]))
for grp, sub in hurt2.groupby("sev", observed=True):
    print(f"  {grp}: n={len(sub)}  mean change {(sub.rate1-sub.rate0).mean():+.4f}")

print("\n=== Tommy's specific scenario: a SHORT (5-9 game), non-major, EARLY-resolved injury -- does it predict any drop at all next season? ===")
minor_early = hurt[(hurt.n_out < 10) & (hurt.major < 0.3) & (hurt.late == 0)]
other = P[~P.index.isin(minor_early.index)]
print(f"  minor & resolved with time to spare: n={len(minor_early)}  mean change {(minor_early.rate1-minor_early.rate0).mean():+.4f}")
print(f"  everyone else:                        n={len(other)}  mean change {(other.rate1-other.rate0).mean():+.4f}")
