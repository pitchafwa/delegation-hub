"""Age fall-off study (2026-11-09, Tommy, after DeMar DeRozan: DELCO 40 -> 33, actual 23.8 through 7 games, age 37).

Question: when a player starts a season well below his DELCO number, does the rest of his season stay lower than DELCO for OLDER players more than for younger
ones? If so, DELCO should drop faster for older fallers.

Part 1. At 5, 8, 12 and 20 games into each season 2011-12..2025-26: cur = DELCO after that game, sofar = his average so far, future = his average over the next 20
games played (at least 10). For players DELCO opened the season at 20+: future - cur = a + k x (sofar - cur). k > 0 means DELCO has not moved far enough toward what he
is doing. Fit by age band, separately for fallers and risers, with bootstrap ranges by player. Theory holds if fallers' k rises with age and risers' doesn't explain it.
Part 2. Out-of-sample: fit on seasons through 2020-21, score 2021-22..2025-26 fallers: DELCO as is / one k for all ages / k by age.
Part 3. DeRozan-like cases: age 33+, 5-10 games in, opened 28+, 6+ under DELCO.
Part 4. A frozen preseason number (the role ESPN's plays in long_level, using DELCO's own opening number as the stand-in): is (cur + open) / 2 better or worse than cur?
Part 5. This season: ESPN vs DELCO vs the average, predicting games after each checkpoint date (projection_log.csv). Small sample; blend_study.py is the real test in December.
Run from ingest/:  .venv/Scripts/python research/age_falloff_study.py   (writes data/age_falloff_study.log)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import form_common as FC

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
OUT = open(D / "age_falloff_study.log", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


g = FC.load_all()[["pid", "date", "fp"]].copy()
g["date"] = pd.to_datetime(g["date"]).dt.normalize()
g = g.groupby(["pid", "date"], as_index=False).fp.max()
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "SEASON", "MIN", "pre", "after", "preMIN"], low_memory=False)
k["date"] = pd.to_datetime(k.GAME_DATE).dt.normalize()
k = k[k.MIN > 0]
a = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "GAME_DATE", "AGE_AT_GAME"], low_memory=False)
a["date"] = pd.to_datetime(a.GAME_DATE).dt.normalize()
a = a.groupby(["PLAYER_ID", "date"], as_index=False).AGE_AT_GAME.max()
m = k.merge(g, left_on=["PLAYER_ID", "date"], right_on=["pid", "date"]).merge(a, on=["PLAYER_ID", "date"], how="left").sort_values(["PLAYER_ID", "date"])
say(f"games joined {len(m):,} of {len(k):,}; seasons {m.SEASON.min()}..{m.SEASON.max()}; age known {m.AGE_AT_GAME.notna().mean():.1%}")

rows, reached = [], []
for (pid, season), x in m.groupby(["PLAYER_ID", "SEASON"], sort=False):
    if season < "2011-12" or season > "2025-26" or len(x) < 5:
        continue
    fp_, after, pre, age, mins = x.fp.to_numpy(float), x["after"].to_numpy(float), x["pre"].to_numpy(float), x.AGE_AT_GAME.to_numpy(float), x.MIN.to_numpy(float)
    if np.isnan(pre[0]) or pre[0] < 20:
        continue
    for n in (5, 8, 12, 20):
        fut = fp_[n:n + 20]
        if len(fp_) >= n and not np.isnan(after[n - 1]) and not np.isnan(age[n - 1]):      # everyone who reached this point, for the survivorship check
            reached.append({"age": age[n - 1], "n": n, "gap": fp_[:n].mean() - after[n - 1], "open": pre[0], "later": len(fut)})
        if len(x) < 15 or len(fut) < 10 or np.isnan(after[n - 1]) or np.isnan(age[n - 1]):
            continue
        rows.append({"openmin": x.preMIN.iloc[0], "pid": pid, "name": x.PLAYER_NAME.iloc[0], "season": season, "n": n, "age": age[n - 1], "open": pre[0], "cur": after[n - 1], "sofar": fp_[:n].mean(),
                     "fut": fut.mean(), "nf": len(fut), "min_sofar": mins[:n].mean(), "min_fut": mins[n:n + 20].mean()})
P = pd.DataFrame(rows)
P["gap"] = P.sofar - P.cur
P["res"] = P.fut - P.cur
BANDS = [(0, 25, "under 25"), (25, 30, "25-29"), (30, 33, "30-32"), (33, 35, "33-34"), (35, 99, "35+")]
P["band"] = pd.cut(P.age, [b[0] for b in BANDS] + [99], right=False, labels=[b[2] for b in BANDS])
say(f"checkpoints {len(P):,}, player-seasons {P.groupby(['pid', 'season']).ngroups:,}\n")


def fit(x, w=True):
    """future - cur = a + k x gap, weighted by games in the target"""
    if len(x) < 8:
        return np.nan, np.nan
    A = np.c_[np.ones(len(x)), x.gap]
    W = np.sqrt(x.nf.to_numpy(float))
    co = np.linalg.lstsq(A * W[:, None], x.res.to_numpy() * W, rcond=None)[0]
    return co[0], co[1]


def boot(x, B=300, seed=7):
    """range for k, resampling whole players (their checkpoints are not independent)"""
    r = np.random.default_rng(seed)
    grp = {p: v for p, v in x.groupby("pid")}
    ids = list(grp)
    vals = []
    for _ in range(B):
        s = pd.concat([grp[i] for i in r.choice(ids, len(ids))])
        vals.append(fit(s)[1])
    return np.nanpercentile(vals, [5, 95])


for label, sel in (("FALLERS (5+ under DELCO so far)", P.gap <= -5), ("all under DELCO", P.gap < 0), ("RISERS (5+ over DELCO so far)", P.gap >= 5)):
    say(f"PART 1. {label}: k by age. k = share of the gap DELCO should still move; avg gap; what actually happened vs DELCO")
    say("   age        n  players     k   (90% range)   intercept  avg gap  future-DELCO   minutes so far -> next 20")
    for _, _, b in BANDS:
        x = P[sel & (P.band == b)]
        if len(x) < 8:
            say(f"   {b:8s} {len(x):5d}  too few")
            continue
        a0, kk = fit(x)
        lo, hi = boot(x)
        say(f"   {b:8s} {len(x):5d} {x.pid.nunique():6d}  {kk:5.2f}  ({lo:5.2f},{hi:5.2f})   {a0:6.2f}   {x.gap.mean():6.1f}   {np.average(x.res, weights=x.nf):7.2f}        {x.min_sofar.mean():.1f} -> {x.min_fut.mean():.1f}")
    say("")

say("PART 1b. Fallers by age, at each games checkpoint: future minus DELCO (points per game), n in brackets")
say("   games " + "".join(f"{b[2]:>16s}" for b in BANDS))
for n in (5, 8, 12, 20):
    cells = []
    for _, _, b in BANDS:
        x = P[(P.gap <= -5) & (P.band == b) & (P.n == n)]
        cells.append(f"{np.average(x.res, weights=x.nf):7.2f} ({len(x):4d})" if len(x) else "      -")
    say(f"   {n:5d} " + "".join(f"{c:>16s}" for c in cells))
say("")

say("PART 1c. Survivorship: of fallers who reached 5-12 games, the share who played fewer than 10 more games that season (they drop out of everything above)")
R_ = pd.DataFrame(reached)
R_ = R_[(R_.n <= 12) & (R_.gap <= -5)]
R_["band"] = pd.cut(R_.age, [b[0] for b in BANDS] + [99], right=False, labels=[b[2] for b in BANDS])
for _, _, b in BANDS:
    x = R_[R_.band == b]
    say(f"   {b:8s} {len(x):5d}  fewer than 10 more games: {(x.later < 10).mean():.0%}")
say("")

say("PART 1d. Is it minutes? Fallers 5-12 games in, opened 28+: minutes so far vs DELCO's opening minutes. 'stayed' = share of the gap that stayed in the next 20")
say("   age        minutes vs expected         n   DELCO  so far  next 20  stayed   minutes so far -> next")
Q = P[(P.gap <= -5) & (P.n <= 12) & (P.open >= 28)].sort_values(["pid", "season", "n"]).groupby(["pid", "season"], as_index=False).first()
Q["mdiff"] = Q.min_sofar - Q.openmin
for lab, sel in (("under 33", Q.age < 33), ("33+", Q.age >= 33)):
    for ml, ms in (("down 6+", Q.mdiff <= -6), ("down 3-6", (Q.mdiff > -6) & (Q.mdiff <= -3)), ("within 3", Q.mdiff > -3)):
        x = Q[sel & ms]
        if len(x):
            say(f"   {lab:9s}  {ml:24s} {len(x):5d}   {x.cur.mean():5.1f}  {x.sofar.mean():5.1f}   {x.fut.mean():5.1f}   {(x.cur - x.fut).sum() / (x.cur - x.sofar).sum():5.0%}    {x.min_sofar.mean():.1f} -> {x.min_fut.mean():.1f}")
x = Q[(Q.age >= 33) & (Q.mdiff <= -6)].sort_values("season")
for _, r in x.iterrows():
    say(f"     {r['season']} {r['name'][:22]:22s} age {r.age:4.1f}: expected {r.openmin:4.1f} min, played {r.min_sofar:4.1f} -> {r.min_fut:4.1f}; DELCO {r.cur:4.1f} so far {r.sofar:4.1f} next {r.fut:4.1f}")
say("")

tr, te = P[P.season <= "2020-21"], P[P.season >= "2021-22"]
fall = lambda x: x[x.gap <= -5]
say("PART 2. Out of sample: fit on 2011-21, score fallers 2021-26 (error of next-20 average, points per game)")
a_all, k_all = fit(fall(tr))
kb = {b: fit(fall(tr[tr.band == b])) for _, _, b in BANDS}
# older players only, everyone else stays on the one-k rule: the shape an age rule would really take
say(f"   fit: one k {k_all:.2f}; by age " + ", ".join(f"{b} {v[1]:.2f}" for b, v in kb.items()))
t = fall(te)
pred0 = t.cur
pred1 = t.cur + k_all * t.gap
pred2 = t.cur + t.band.map(lambda b: kb[b][1]).astype(float) * t.gap
pred3 = t.cur + t.band.map(lambda b: kb[b][1] if b in ("33-34", "35+") else 0.0).astype(float) * t.gap


def err(p, x):
    e = x.fut - p
    return np.sqrt(np.average(e ** 2, weights=x.nf)), np.average(e, weights=x.nf)


say("   variant                         everyone: error  bias |  33+: error  bias, n")
old = t.age >= 33
for nm, pr in (("DELCO as is", pred0), ("one k for all ages", pred1), ("k by age", pred2), ("k only for 33+", pred3)):
    e1, b1 = err(pr, t)
    e2, b2 = err(pr[old], t[old])
    say(f"   {nm:30s}   {e1:6.2f} {b1:6.2f}  |  {e2:6.2f} {b2:6.2f}  {int(old.sum())}")
say("")

say("PART 3. DeRozan-like: age 33+, 5-12 games in, opened 28+, 6+ under DELCO (one row per player-season, the first checkpoint that qualifies)")
c = P[(P.age >= 33) & (P.n <= 12) & (P.open >= 28) & (P.gap <= -6)].sort_values(["pid", "season", "n"]).groupby(["pid", "season"], as_index=False).first()
say(f"   cases {len(c)}; DELCO then {c.cur.mean():.1f}, so far {c.sofar.mean():.1f}, next 20 games {c.fut.mean():.1f}; minutes {c.min_sofar.mean():.1f} -> {c.min_fut.mean():.1f}; share of the gap that stayed: {(c.cur - c.fut).sum() / (c.cur - c.sofar).sum():.0%}")
y = P[(P.age < 30) & (P.n <= 12) & (P.open >= 28) & (P.gap <= -6)].sort_values(["pid", "season", "n"]).groupby(["pid", "season"], as_index=False).first()
say(f"   same, under 30: cases {len(y)}; DELCO then {y.cur.mean():.1f}, so far {y.sofar.mean():.1f}, next 20 games {y.fut.mean():.1f}; minutes {y.min_sofar.mean():.1f} -> {y.min_fut.mean():.1f}; share of the gap that stayed: {(y.cur - y.fut).sum() / (y.cur - y.sofar).sum():.0%}")
for _, r in c.sort_values("season").iterrows():
    say(f"     {r['season']} {r['name'][:22]:22s} age {r.age:4.1f} at {int(r.n):2d}g: open {r.open:4.1f} DELCO {r.cur:4.1f} so far {r.sofar:4.1f} -> next {int(r.nf):2d}: {r.fut:4.1f}  min {r.min_sofar:4.1f}->{r.min_fut:4.1f}")
say("")

say("PART 4. A frozen preseason number in the blend (stand-in for ESPN: DELCO's own opening number), error predicting the next 20 games, all players opening 20+")
say("   games   DELCO now   (DELCO now + opening) / 2   opening alone")
for n in (5, 8, 12, 20):
    x = P[P.n == n]
    say(f"   {n:5d}     {err(x.cur, x)[0]:6.2f}      {err((x.cur + x.open) / 2, x)[0]:6.2f}                   {err(x.open, x)[0]:6.2f}")
x = P[(P.n <= 12) & (P.gap <= -5)]
say(f"   fallers, 5-12 games: DELCO now {err(x.cur, x)[0]:.2f} (bias {err(x.cur, x)[1]:+.2f}), with opening averaged in {err((x.cur + x.open) / 2, x)[0]:.2f} (bias {err((x.cur + x.open) / 2, x)[1]:+.2f})")
say("")

say("PART 5. This season, ESPN's number vs DELCO, predicting the games after each date (projection_log.csv)")
L = pd.read_csv(FC.Path(__file__).resolve().parent.parent.parent / "dashboard" / "projection_log.csv", low_memory=False)
L = L.dropna(subset=["espn", "model"])
L["date"] = pd.to_datetime(L.date)
L["key"] = L.name.map(lambda s: "".join(ch for ch in str(s).lower() if ch.isalpha()))
say(f"   ESPN's number changed for {(L.groupby('espn_id').espn.nunique() > 1).mean():.1%} of {L.espn_id.nunique()} players between {L.date.min().date()} and {L.date.max().date()}")
import json
RD = json.load(open(FC.Path(__file__).resolve().parent.parent.parent / "dashboard" / "redraft_data.json", encoding="utf-8"))
gl = []
for p in RD["players"]:
    for v, d_ in zip(p.get("g") or [], p.get("gd") or []):
        if d_ >= "10-01":                      # this season's games (the list is his last 10)
            gl.append({"key": "".join(ch for ch in p["name"].lower() if ch.isalpha()), "date": pd.Timestamp("2026-" + d_), "fp": v})
gl = pd.DataFrame(gl)
for day in ("2026-10-28", "2026-11-02"):
    d0 = pd.Timestamp(day)
    last = L[L.date <= d0].sort_values("date").groupby("key").tail(1)
    fut = gl[gl.date > d0].groupby("key").fp.agg(["mean", "count"])
    sf = gl[gl.date <= d0].groupby("key").fp.agg(["mean", "count"]).rename(columns={"mean": "sf", "count": "nsf"})
    z = last.merge(fut, left_on="key", right_index=True).merge(sf, left_on="key", right_index=True, how="left")
    z = z[(z["count"] >= 2) & (z.espn >= 15)]
    row = lambda p, q=z: f"error {np.sqrt(np.average((q['mean'] - p) ** 2, weights=q['count'])):5.2f} bias {np.average(q['mean'] - p, weights=q['count']):+5.2f}"
    say(f"   from {day}: {len(z)} players, {int(z['count'].sum())} games:  ESPN {row(z.espn)} | DELCO {row(z.model)} | average {row((z.espn + z.model) / 2)}")
    q = z[(z.nsf >= 3) & (z.sf <= z.espn - 5)]
    if len(q):
        say(f"     running 5+ under ESPN so far ({len(q)} players):  ESPN {row(q.espn, q)} | DELCO {row(q.model, q)} | average {row((q.espn + q.model) / 2, q)}")
OUT.close()
