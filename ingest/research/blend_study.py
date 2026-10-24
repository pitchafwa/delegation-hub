"""Strategy study #4 (set up 2026-10-24, run in mid-December once there is enough data): what is the best blend of our projection sources?

Sources logged every game day by build_week_plan.py in dashboard/projection_log.csv (from 2026-10-24 on all columns are filled):
  espn          ESPN's season projection (points per game)          model         DELCO season mode            model_career  DELCO career mode
  l15           ESPN's last-15-games average (3+ games)             props         sportsbook-props level        level         what the plan actually used
plus DailyFantasyFuel's projection at tip-off (local log data/dfs/log.csv, written by dfs_monitor.py on the PC).
Actual results: this league's fantasy points from the box scores (form_common.load_all).

  Q1 TONIGHT    which mix best predicts a player's points in tonight's game (players who played)?  Today: DFF when posted, else the plan's level.
  Q2 SEASON     which mix best predicts his average over his NEXT 20 games?                          Today: (ESPN + DELCO) / 2  (build_week_plan.long_level)
  Q3            do the best weights differ early vs later (fewer than 10 vs 10+ games in ESPN's last-15 window)?
Fit on the earlier 60% of dates, judged on the later 40% (never seen during fitting). Weights are non-negative and sum to 1 (a blend, no free intercept).
Writes data/blend_study_report.txt.  Run from ingest/:  .venv/Scripts/python research/blend_study.py
"""
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
HUB = R.parent.parent / "dashboard"
out = []
P = lambda s="": (out.append(s), print(s))


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def best_blend(df, cols, target, step=0.05):
    """grid search over weights summing to 1 (coarse but robust with few sources); returns (weights, MAE)"""
    grid = np.arange(0, 1 + 1e-9, step)
    best = (None, 1e9)
    X = df[cols].to_numpy(float)
    y = df[target].to_numpy(float)
    for w in product(grid, repeat=len(cols) - 1):
        last = 1 - sum(w)
        if last < -1e-9:
            continue
        ww = np.array(list(w) + [last])
        mae = np.mean(np.abs(X @ ww - y))
        if mae < best[1]:
            best = (ww, mae)
    return best


def mae(df, col, target):
    return float(np.mean(np.abs(df[col] - df[target])))


import form_common as F
games = F.load_all(cache=False)
games["key"] = games["name"].map(norm)
games["day"] = games["date"].dt.strftime("%Y-%m-%d")
games = games.sort_values("date")
act = games.drop_duplicates(["day", "key"]).set_index(["day", "key"])["fp"]

log = pd.read_csv(HUB / "projection_log.csv")
log["key"] = log["name"].map(norm)
for c in ("espn", "model", "model_career", "l15", "props", "level"):
    if c in log:
        log[c] = pd.to_numeric(log[c], errors="coerce")
log = log.drop_duplicates(["date", "key"])
P(f"projection log: {len(log)} player-days, {log.date.nunique()} dates ({log.date.min()} .. {log.date.max()})")

# DFF at tip-off
dff = {}
dlog_p = R / "data" / "dfs" / "log.csv"
if dlog_p.exists():
    from grade_dfs import tips_for
    dl = pd.read_csv(dlog_p)
    dl["t"] = pd.to_datetime(dl["fetched_utc"], utc=True)
    dl["key"] = dl["name"].map(norm)
    for day, g in dl.groupby("game_date"):
        tips = tips_for(day)
        for (key, team), h in g.groupby(["key", "team"]):
            tip = tips.get(team)
            if tip is None:
                continue
            b = h[h.t < tip]
            if len(b):
                dff[(day, key)] = float(b.sort_values("t").iloc[-1]["league_proj"])
log["dff"] = [dff.get((d, k), np.nan) for d, k in zip(log.date, log.key)]
log["actual"] = [float(act.loc[(d, k)]) if (d, k) in act.index else np.nan for d, k in zip(log.date, log.key)]

# ---------------- Q1 tonight
P("\nQ1 TONIGHT: predicting tonight's points (players who played)")
q1 = log.dropna(subset=["actual"])
dates = sorted(q1.date.unique())
cut = dates[int(len(dates) * 0.6)] if len(dates) >= 5 else None
if cut is None or len(q1) < 300:
    P(f"  not enough data yet ({len(q1)} player-games, {len(dates)} dates; want 300+ over 5+ dates)")
else:
    for cols in (["dff", "espn", "model", "l15"], ["dff", "espn", "model"], ["espn", "model", "l15"]):
        d = q1.dropna(subset=cols)
        fit, test = d[d.date < cut], d[d.date >= cut]
        if len(fit) < 100 or len(test) < 50:
            continue
        w, _ = best_blend(fit, cols, "actual")
        test = test.assign(blend=test[cols].to_numpy(float) @ w)
        cur = "dff" if "dff" in cols else "level"
        P(f"  sources {cols}: best weights {dict(zip(cols, np.round(w, 2)))}  test MAE blend {mae(test, 'blend', 'actual'):.2f}"
          f" vs today's rule ({cur}) {mae(test.dropna(subset=[cur]), cur, 'actual'):.2f}   (fit n={len(fit)}, test n={len(test)})")

# ---------------- Q2 rest of season (next 20 games)
P("\nQ2 SEASON: predicting a player's average over his next 20 games")
nxt = {}
for key, g in games.groupby("key"):
    days_ = g["day"].to_numpy()
    fp = g["fp"].to_numpy(float)
    for i in range(len(g)):
        if len(fp) - i >= 20:
            nxt[(days_[i], key)] = float(fp[i:i + 20].mean())
log["next20"] = [nxt.get((d, k), np.nan) for d, k in zip(log.date, log.key)]
q2 = log.dropna(subset=["next20", "espn", "model"]).copy()
q2["current_rule"] = (q2.espn + q2.model) / 2
if len(q2) < 300 or q2.date.nunique() < 5:
    P(f"  not enough data yet ({len(q2)} player-days with 20 later games; want 300+ over 5+ dates)")
else:
    dates = sorted(q2.date.unique())
    cut = dates[int(len(dates) * 0.6)]
    for cols in (["espn", "model", "l15"], ["espn", "model", "model_career", "l15"], ["espn", "model"]):
        d = q2.dropna(subset=cols)
        fit, test = d[d.date < cut], d[d.date >= cut]
        if len(fit) < 100 or len(test) < 50:
            continue
        w, _ = best_blend(fit, cols, "next20")
        test = test.assign(blend=test[cols].to_numpy(float) @ w)
        P(f"  sources {cols}: best weights {dict(zip(cols, np.round(w, 2)))}  test MAE blend {mae(test, 'blend', 'next20'):.2f} vs (ESPN+DELCO)/2 {mae(test, 'current_rule', 'next20'):.2f}"
          f"   ESPN alone {mae(test, 'espn', 'next20'):.2f}   DELCO alone {mae(test, 'model', 'next20'):.2f}")
    # Q3: early vs later
    P("\nQ3: same, split by games in ESPN's last-15 window")
    q2["gp15"] = pd.to_numeric(q2.get("gp15"), errors="coerce")
    for lab, m in (("under 10 games", q2.gp15 < 10), ("10+ games", q2.gp15 >= 10)):
        d = q2[m].dropna(subset=["espn", "model", "l15"])
        if len(d) < 200:
            P(f"  {lab}: not enough data ({len(d)})")
            continue
        fit, test = d[d.date < cut], d[d.date >= cut]
        if len(fit) < 100 or len(test) < 50:
            P(f"  {lab}: not enough data after the date split")
            continue
        w, _ = best_blend(fit, ["espn", "model", "l15"], "next20")
        test = test.assign(blend=test[["espn", "model", "l15"]].to_numpy(float) @ w)
        P(f"  {lab}: weights ESPN/DELCO/last-15 {np.round(w, 2)}  test MAE {mae(test, 'blend', 'next20'):.2f} vs current {mae(test, 'current_rule', 'next20'):.2f}")
(R / "data" / "blend_study_report.txt").write_text("\n".join(out), encoding="utf-8")
