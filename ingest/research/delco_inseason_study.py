"""DELCO early-season responsiveness study (Tommy, 2026-10-24).

Question: DELCO's Kalman parameters (data/kalman_fit_<STAT>.json) were fit to predict seasons 3-4 YEARS ahead. Used in season, they make it very slow to react
(minutes: Q=0.0021/day, R=50 -> each game moves the minutes estimate by about 1%). Early in a season roles change (new teammates, trades, returning injured players,
a coach's rotation) and human projections fold that in. Does letting DELCO update faster -- especially right after the offseason -- predict the REST OF THE SEASON better?

Two knobs per stat, on top of the fitted parameters:
  QM  multiplies the day-to-day drift Q during the season (how fast the estimate follows new games all year)
  J   extra variance added at a player's first game of a new season (a role can change over the summer), in units of the stat's per-game observation noise share
Scoring: at checkpoints of 0,1,2,3,5,8,12,20 games into a season, compare each stat's estimate with what the player actually did over the REST of that season
(minutes per game; per-minute rate for the others), weighted by remaining games; then combine into fantasy points per game (league scoring).
Fit on seasons 2016-17 .. 2020-21, judge on 2021-22 .. 2025-26 (never seen during fitting).

Run from ingest/:  .venv/Scripts/python research/delco_inseason_study.py   (writes data/delco_inseason_study.json)
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R_ = Path(__file__).resolve().parent
STATS = ["MIN", "PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA"]
W = {"PTS": 1, "REB": 1.5, "AST": 2, "BLK": 3, "FG3M": 1, "FTM": 2, "FTA": -1, "TOV": -1}     # STL (3) is a flat career rate in DELCO: identical in every variant, left out
CHECKS = [0, 1, 2, 3, 5, 8, 12, 20]
FIT_SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]
TEST_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

t0 = time.time()
df = pd.read_csv(R_ / "data" / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_DATE", "TEAM", "AGE_AT_GAME", "DAYS_SINCE_LAST"] + STATS + ["STL"])
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
pid = df["PLAYER_ID"].to_numpy()
days = df["DAYS_SINCE_LAST"].to_numpy(float)
age = df["AGE_AT_GAME"].to_numpy(float)
mins = df["MIN"].to_numpy(float)
new_player = np.r_[True, pid[1:] != pid[:-1]]
season = df["SEASON"].to_numpy()
season_start = np.r_[False, (season[1:] != season[:-1]) & ~new_player[1:]]          # first game of a new season for a returning player
gi = df.groupby(["PLAYER_ID", "SEASON"]).cumcount().to_numpy()                        # games into this season before this row (0 = first game)
gn = df.groupby(["PLAYER_ID", "SEASON"])["MIN"].transform("size").to_numpy()
# team change since the previous season (a crude role-change flag for the subgroup check)
prev_team = df.groupby("PLAYER_ID")["TEAM"].shift(1)
df["moved"] = season_start & (df["TEAM"].to_numpy() != prev_team.to_numpy())
moved_ps = set(map(tuple, df.loc[df["moved"], ["PLAYER_ID", "SEASON"]].to_numpy()))
print(f"{len(df)} games, {df.PLAYER_ID.nunique()} players, loaded in {time.time() - t0:.0f}s")


moved_row = df["moved"].to_numpy()


def run(stat, QM, J, JM=1.0):
    Q, R, peak, su, sd = json.load(open(R_ / "data" / f"kalman_fit_{stat}.json"))["params"]
    is_min = stat == "MIN"
    obs = mins if is_min else df[stat].to_numpy(float)
    gain = np.ones_like(mins) if is_min else mins
    x0 = float(mins.mean()) if is_min else float(obs.sum() / max(gain.sum(), 1.0))
    jump = J * R / (1.0 if is_min else 30.0)              # J in units of one game's observation noise (rates: a 30-minute game)
    pre = np.empty(len(df))
    x, P = x0, 50.0
    for i in range(len(df)):
        if new_player[i]:
            x, P = x0, 50.0
        dd = days[i]
        slope = (su if age[i] <= peak else sd) / 365.0
        x = x + slope * dd
        P = P + Q * dd * (QM if dd < 60 else 1.0) + ((jump * (JM if moved_row[i] else 1.0)) if season_start[i] else 0.0)
        pre[i] = x                                         # estimate BEFORE this game: what we knew after gi[i] games of this season
        H = gain[i]
        if H > 1e-6:
            Re = R * max(H, 1.0)
            S = H * H * P + Re
            K = H * P / S
            x = x + K * (obs[i] - H * x)
            P = (1 - K * H) * P
    return pre


# actual rest-of-season targets at each row (from this game to season end, inclusive): minutes per game, per-minute rates, fantasy points per game
grp = df.groupby(["PLAYER_ID", "SEASON"])
rev = lambda c: grp[c].transform(lambda s: s[::-1].cumsum()[::-1]).to_numpy(float)
ros_n = (gn - gi).astype(float)
ros_min_tot = rev("MIN")
target = {"MIN": ros_min_tot / ros_n}
for s in STATS[1:]:
    target[s] = rev(s) / np.maximum(ros_min_tot, 1.0)
fp_game = sum(W[s] * df[s].to_numpy(float) for s in W) + 3 * df["STL"].to_numpy(float)
ros_fp = rev_fp = pd.Series(fp_game).groupby([df["PLAYER_ID"], df["SEASON"]]).transform(lambda s: s[::-1].cumsum()[::-1]).to_numpy() / ros_n
stl_rate = (df.groupby("PLAYER_ID")["STL"].transform("sum") / df.groupby("PLAYER_ID")["MIN"].transform("sum").clip(lower=1)).to_numpy()   # flat career rate, as DELCO
print(f"targets built in {time.time() - t0:.0f}s")


def mask(seasons, min_games=20):
    return np.isin(season, seasons) & np.isin(gi, CHECKS) & (gn >= min_games) & (ros_n >= 10)


def stat_err(pred, stat, m):
    w = ros_n[m]
    return float(np.sqrt(np.sum(w * (pred[m] - target[stat][m]) ** 2) / np.sum(w)))


import os
GRID_QM = [1, 3, 10, 30, 100]
GRID_J = [0, 0.5, 2, 8]
GRID_JM = [1.0]
REFINE = {"MIN": ([2, 3, 4, 6], [0.25, 0.5, 1.0, 1.5], [1.0, 2.0, 4.0]), "FG3M": ([100, 300, 1000], [0], [1.0])} if os.environ.get("REFINE") else {}
best, table = {}, {}
mf, mt = mask(FIT_SEASONS), mask(TEST_SEASONS)
base_pred = {}
for stat in STATS:
    res = []
    gq, gj, gm = REFINE.get(stat, (GRID_QM, GRID_J, GRID_JM))
    base_pred[stat] = run(stat, 1, 0)
    e0_fit = stat_err(base_pred[stat], stat, mf)
    for QM, J, JM in itertools.product(gq, gj, gm):
        p = run(stat, QM, J, JM)
        res.append((stat_err(p, stat, mf), QM, J, p, JM))
    res.sort(key=lambda r: r[0])
    e_fit, QM, J, p, JM = res[0]
    best[stat] = {"QM": QM, "J": J, "JM": JM, "pred": p}
    table[stat] = {"QM": QM, "J": J, "JM": JM, "fit_err_base": round(e0_fit, 4), "fit_err_best": round(e_fit, 4),
                   "test_err_base": round(stat_err(base_pred[stat], stat, mt), 4), "test_err_best": round(stat_err(p, stat, mt), 4)}
    print(f"{stat:5s} best QM={QM:<4} J={J:<4} JM={JM:<3}  fit {e0_fit:.4f} -> {e_fit:.4f}   TEST {table[stat]['test_err_base']:.4f} -> {table[stat]['test_err_best']:.4f}   ({time.time() - t0:.0f}s)", flush=True)


def fpg(preds):
    m_ = np.maximum(preds["MIN"], 0)
    return sum(W[s] * np.maximum(preds[s], 0) * m_ for s in W) + 3 * stl_rate * m_


fp_base = fpg(base_pred)
fp_best = fpg({s: best[s]["pred"] for s in STATS})
season_avg = None
# simple reference: season-to-date average (prior-season average before game 1)
summary = {"by_check": [], "stats": table}
for k in CHECKS:
    m = mt & (gi == k)
    w = ros_n[m]
    e_b = float(np.sum(w * np.abs(fp_base[m] - ros_fp[m])) / np.sum(w))
    e_n = float(np.sum(w * np.abs(fp_best[m] - ros_fp[m])) / np.sum(w))
    mv = m & np.array([(a, b) in moved_ps for a, b in zip(pid, season)])
    wv = ros_n[mv]
    e_bv = float(np.sum(wv * np.abs(fp_base[mv] - ros_fp[mv])) / np.sum(wv)) if mv.any() else None
    e_nv = float(np.sum(wv * np.abs(fp_best[mv] - ros_fp[mv])) / np.sum(wv)) if mv.any() else None
    summary["by_check"].append({"games_in": k, "n": int(m.sum()), "mae_current": round(e_b, 2), "mae_new": round(e_n, 2), "moved_n": int(mv.sum()), "moved_mae_current": e_bv and round(e_bv, 2), "moved_mae_new": e_nv and round(e_nv, 2)})
    print(f"after {k:2d} games: fantasy pts/game error (rest of season, test seasons) current {e_b:5.2f} -> new {e_n:5.2f}" + (f"   | changed teams (n={int(mv.sum())}): {e_bv:5.2f} -> {e_nv:5.2f}" if mv.any() else ""))
(R_ / "data" / "delco_inseason_study.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(f"done in {time.time() - t0:.0f}s")
