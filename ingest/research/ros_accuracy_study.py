"""Rest-of-season accuracy study, stage 1 (2026-11-16, Tommy: tune the future-weeks projection behind playoff odds and projected record as accurately as
possible; plan in RESEARCH_playoff_accuracy.md). Replays this league's 2025-26 season (backtest_phase2: real rosters, real box scores).

From each anchor week A (start of week, only what was known then), the solver forecasts every later regular-season week T from the roster held at A
(what the live schedule_plan does today). Questions:
  A  How does the error grow with the horizon (T - A)? How far below/above real points do held-roster forecasts drift as rosters churn?
  B  Manager factor = real points / forecast. Does a team's factor persist? (factor in weeks before A vs factor in weeks after A, horizon by horizon)
     Which shrink of the team's own factor predicts best at each horizon? (today: league factor x (team/league)^(n/(n+5)), the same at every horizon)
  C  Rest-of-season TOTAL points by team from each anchor: error and rank correlation of each method (what drives playoff odds).
Caveat (Tommy): DRNK only started being actively managed in 2026-27, so DRNK's 2025-26 factor is not a guide to DRNK now (reported, not used for DRNK).
Run from ingest/:  python research/ros_accuracy_study.py   -> data/ros_accuracy_stage1.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import backtest_phase2 as P

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "data" / "ros_accuracy_stage1.txt"
REG = list(P.REG)
TEAMS = list(P.TEAMS)
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


# forecasts f[(t, A, T)] from roster at A; same-week forecasts give the realized manager factor per week
rows = []
CACHE = Path(__file__).resolve().parent / "data" / "ros_accuracy_forecasts.csv"
for ai, A in enumerate(REG if not CACHE.exists() else []):
    for t in TEAMS:
        for T in REG[ai:]:
            m = P.project(t, A, T)[0]
            if m:
                rows.append({"team": t, "A": A, "T": T, "h": REG.index(T) - ai, "fc": m, "act": P.actual_pts(T, t)})
    say(f"anchor {A}: {len(rows)} forecasts so far")
df = pd.DataFrame(rows) if rows else pd.read_csv(CACHE)
df = df[df["act"].notna() & (df["act"] > 0)]
df.to_csv(Path(__file__).resolve().parent / "data" / "ros_accuracy_forecasts.csv", index=False)
same = df[df.h == 0].set_index(["team", "T"])["act"] / df[df.h == 0].set_index(["team", "T"])["fc"]      # weekly manager factor

say("\n=== A. Held-roster forecast vs real points, by horizon (weeks ahead) ===")
say("h   n    real/forecast (pooled)   mean abs miss   miss % of real")
for h, g in df.groupby("h"):
    if h > 16:
        break
    say(f"{h:<3} {len(g):<4} {g.act.sum() / g.fc.sum():8.3f}   {np.mean(np.abs(g.act - g.fc)):12.0f}   {np.mean(np.abs(g.act - g.fc) / g.act) * 100:8.1f}%")


def factor_before(t, A):
    ws = [w for w in REG if w < A and (t, w) in same.index]
    if not ws:
        return None, 0
    r = df[(df.team == t) & (df.h == 0) & (df["T"].isin(ws))]
    return r.act.sum() / r.fc.sum(), len(ws)


def league_before(A):
    r = df[(df.h == 0) & (df["T"] < A)]
    return r.act.sum() / r.fc.sum() if len(r) else None


say("\n=== B. Does a team's manager factor persist? ===")
say("Correlation across teams: factor before anchor vs real/forecast in later weeks (by horizon)")
for A in [w for w in REG if REG.index(w) >= 2][::3]:
    for hs in ((0, 2), (3, 6), (7, 12)):
        sub = df[(df.A == A) & df.h.between(*hs)]
        if sub.empty:
            continue
        g = sub.groupby("team").apply(lambda x: x.act.sum() / x.fc.sum())
        fb = pd.Series({t: factor_before(t, A)[0] for t in g.index}).dropna()
        common = g.index.intersection(fb.index)
        if len(common) >= 6:
            say(f"anchor {A:>2} horizon {hs[0]}-{hs[1]:<2}: corr {np.corrcoef(fb[common], g[common])[0, 1]:+.2f}  (n teams {len(common)})")

say("\nShrink choice: forecast x league x (team/league)^(n/(n+K)); mean abs miss by horizon band (anchors with 2+ past weeks)")
Ks = [0, 2, 5, 10, 20, 1e9]
res = {}
for (t, A), g in df[df.A.isin(REG[2:])].groupby(["team", "A"]):
    tf, n = factor_before(t, A)
    lf = league_before(A)
    if tf is None or lf is None:
        continue
    for _, r in g.iterrows():
        band = "0-2" if r.h <= 2 else ("3-6" if r.h <= 6 else "7+")
        for K in Ks:
            mult = lf * (tf / lf) ** (n / (n + K)) if K < 1e9 else lf
            res.setdefault((band, K), []).append(abs(r.act - r.fc * mult))
        res.setdefault((band, "raw"), []).append(abs(r.act - r.fc))
for band in ("0-2", "3-6", "7+"):
    say(f"  {band:4} raw {np.mean(res[(band, 'raw')]):5.0f} | " + " ".join(f"K={'inf' if K >= 1e9 else int(K)}:{np.mean(res[(band, K)]):5.0f}" for K in Ks))

say("\n=== C. Rest-of-season total points from each anchor (what playoff odds rest on) ===")
for A in REG[2::3]:
    sub = df[df.A == A]
    tot = sub.groupby("team").agg(fc=("fc", "sum"), act=("act", "sum"))
    lf = league_before(A) or 1.0
    k5 = []
    for t in tot.index:
        tf, n = factor_before(t, A)
        k5.append(tot.loc[t, "fc"] * (lf * (tf / lf) ** (n / (n + 5)) if tf else lf))
    tot["k5"] = k5
    rk = lambda c: pd.Series(tot[c]).rank().corr(tot["act"].rank())
    say(f"anchor {A:>2} ({sub['T'].nunique()} weeks left): miss % raw {np.mean(np.abs(tot.act - tot.fc) / tot.act) * 100:4.1f}  live(K=5) {np.mean(np.abs(tot.act - tot.k5) / tot.act) * 100:4.1f} | rank corr raw {rk('fc'):.2f} live {rk('k5'):.2f}")

say("\nPer-team 2025-26 manager factor (all weeks; DRNK's is not used for DRNK now -- Tommy began active management in 2026-27):")
r0 = df[df.h == 0].groupby("team").apply(lambda x: x.act.sum() / x.fc.sum()).sort_values()
for t, v in r0.items():
    say(f"  {t}: {v:.3f}")
OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT)
