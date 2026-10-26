"""Short-term projection study (Tommy, 2026-10-26): tests #1 (role changes) and #2 (injury fill-ins).

The plan values a player's next few days with season-level projections (DELCO season mode, ESPN), which are built to estimate full-season averages and
react slowly. DailyFantasyFuel reacts fast but only covers tonight. Question: what predicts a player's NEXT FEW GAMES best, right after his latest game?

Data: every regular-season game 2015-16 .. 2025-26 (data/game_logs_unified.csv) and DELCO season mode's belief going into / after each game
(data/delco_pregame_all.csv.gz, from kalman_vor.py). Info point = just after a player's game t. Target = his average fantasy points over his next 3
games (games he played, within 12 days).
Signals at the info point:
  delco      DELCO season belief after game t (fantasy pts/g)                         -- what the plan uses now (with ESPN)
  min_gap    his minutes over the last 3 games minus DELCO's minutes belief            -- #1 role change
  fp_gap     his fantasy points over the last 3 games minus delco                      -- hot / cold
  out_fp     DELCO level of rotation teammates (22+ min belief) who sat out game t     -- #2 injured teammates (production up for grabs)
  out_new    out_fp minus the team's average out_fp over its previous 10 games         -- a NEW absence vs one already baked in
Models fit on 2015-16 .. 2021-22, judged on 2022-23 .. 2025-26 (never seen in fitting). Linear corrections to delco; then the subsets that matter.
Run from ingest/:  .venv/Scripts/python research/short_term_study.py   (writes data/short_term_study.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2026) if y != 2019]
TRAIN = [s for s in SEASONS if s <= "2021-22"]

g = pd.read_csv(D / "game_logs_unified.csv", usecols=["PLAYER_ID", "SEASON", "GAME_ID", "GAME_DATE", "TEAM", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
g = g[g["SEASON"].isin(SEASONS) & (g["MIN"] > 0)].copy()
g["fp"] = g["PTS"] + 1.5 * g["REB"] + 2 * g["AST"] + 3 * g["STL"] + 3 * g["BLK"] + g["FG3M"] + 2 * g["FTM"] - g["FTA"] - g["TOV"] + 3 * g["TD3"]
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "after", "afterMIN"])
g = g.merge(k.drop_duplicates(["PLAYER_ID", "GAME_DATE"]), on=["PLAYER_ID", "GAME_DATE"], how="left")
g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
g = g.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
print(f"{len(g)} player-games, {g['PLAYER_ID'].nunique()} players")

# ---- teammates out: rotation players (DELCO minutes belief 22+ after their latest game, last game for this team within 10 days) missing from a team game
tg = g.groupby(["TEAM", "GAME_ID"])["GAME_DATE"].first().reset_index().sort_values(["TEAM", "GAME_DATE"])
out_vals = {}
for team, tgs in tg.groupby("TEAM"):
    sub = g[g["TEAM"] == team]
    by_game = {gid: d for gid, d in sub.groupby("GAME_ID")}
    state = {}
    for r in tgs.itertuples():
        d = by_game[r.GAME_ID]
        here = set(d["PLAYER_ID"])
        out_vals[(team, r.GAME_ID)] = sum(v[1] for pid, v in state.items()
                                          if pid not in here and v[2] >= 22 and (r.GAME_DATE - v[0]).days <= 10 and np.isfinite(v[1]))
        for x in d.itertuples():
            state[x.PLAYER_ID] = (x.GAME_DATE, x.after, x.afterMIN if np.isfinite(x.afterMIN) else 0.0)
g["out_fp"] = [out_vals.get((t, gid), 0.0) for t, gid in zip(g["TEAM"], g["GAME_ID"])]
team_avg = {}
for team, tgs in tg.groupby("TEAM"):
    vals = [out_vals[(team, gid)] for gid in tgs["GAME_ID"]]
    prev = pd.Series(vals).shift(1).rolling(10, min_periods=3).mean()
    for gid, v in zip(tgs["GAME_ID"], prev):
        team_avg[(team, gid)] = v if np.isfinite(v) else 0.0
g["out_new"] = g["out_fp"] - [team_avg.get((t, gid), 0.0) for t, gid in zip(g["TEAM"], g["GAME_ID"])]

# ---- per player: last-3 signals and the next-3 target
pg = g.groupby("PLAYER_ID")
g["l3_fp"] = pg["fp"].transform(lambda s: s.rolling(3, min_periods=3).mean())
g["l3_min"] = pg["MIN"].transform(lambda s: s.rolling(3, min_periods=3).mean())
# CLEAN GAMES (2026-10-26, Tommy: Flagg's 17 minutes on 10/23 were an injury exit, not a smaller role). A short game says nothing about the role when
#   blowout   the final margin was 20+ (starters sit the fourth quarter), or
#   left hurt minutes under 60% of what DELCO expected AND he missed his team's next game.
# l3c_* = the last 3 CLEAN games (a short game of either kind is skipped).
tm_pts = g.groupby(["GAME_ID", "TEAM"])["PTS"].sum()
opp = tm_pts.groupby(level=0).transform("sum") - tm_pts
g["margin"] = [abs(tm_pts[(a, b)] - opp[(a, b)]) for a, b in zip(g["GAME_ID"], g["TEAM"])]
g["exp_min"] = pg["afterMIN"].shift(1)
team_seq = {t: list(d.sort_values("GAME_DATE")["GAME_ID"]) for t, d in g.groupby("TEAM")[["GAME_ID", "GAME_DATE"]].apply(lambda d: d.drop_duplicates()).reset_index(level=0).groupby("TEAM")}
nxt_game = {}
for t, seq in team_seq.items():
    for a, b in zip(seq, seq[1:]):
        nxt_game[(t, a)] = b
played_set = set(zip(g["PLAYER_ID"], g["GAME_ID"]))
short = g["MIN"] < 0.6 * g["exp_min"].fillna(0)
missed_next = [(pid, nxt_game.get((t, gid))) not in played_set if nxt_game.get((t, gid)) else False for pid, t, gid in zip(g["PLAYER_ID"], g["TEAM"], g["GAME_ID"])]
g["censored"] = (short & (g["margin"] >= 20)) | (short & pd.Series(missed_next, index=g.index))
print(f"games marked not-a-role-signal: {g['censored'].mean():.1%} (blowout short {((short) & (g['margin'] >= 20)).mean():.1%}, left hurt {(short & pd.Series(missed_next, index=g.index)).mean():.1%})")
for col, src in (("l3c_fp", "fp"), ("l3c_min", "MIN")):
    clean = g[src].where(~g["censored"])
    g[col] = clean.groupby(g["PLAYER_ID"]).transform(lambda s: s.dropna().rolling(3, min_periods=3).mean().reindex(s.index).ffill())
nxt = []
for pid, d in pg:
    fp, dt = d["fp"].to_numpy(), d["GAME_DATE"].to_numpy()
    t = np.full(len(d), np.nan)
    for i in range(len(d) - 3):
        if (dt[i + 3] - dt[i]) / np.timedelta64(1, "D") <= 12:
            t[i] = fp[i + 1:i + 4].mean()
    nxt.append(pd.Series(t, index=d.index))
g["target"] = pd.concat(nxt)
X = g.dropna(subset=["target", "after", "afterMIN", "l3_fp", "l3_min", "l3c_fp", "l3c_min"]).copy()
X = X[X["after"] >= 10]
X["min_gap"] = X["l3_min"] - X["afterMIN"]
X["fp_gap"] = X["l3_fp"] - X["after"]
X["rate"] = X["after"] / X["afterMIN"].clip(lower=8)
X["min_gap_pts"] = X["min_gap"] * X["rate"]                 # minutes change converted to points at his usual rate
X["min_gap_c"] = (X["l3c_min"] - X["afterMIN"]) * X["rate"]  # same over the last 3 CLEAN games
X["fp_gap_c"] = X["l3c_fp"] - X["after"]
X["out_share"] = X["out_fp"] * X["afterMIN"] / 240.0        # production up for grabs, weighted by his share of the floor time
X["out_new_share"] = X["out_new"] * X["afterMIN"] / 240.0
tr, te = X[X["SEASON"].isin(TRAIN)], X[~X["SEASON"].isin(TRAIN)]
print(f"info points: train {len(tr)}, test {len(te)}\n")

MODELS = {"DELCO only": [], "+ recent points (fp_gap)": ["fp_gap"], "+ minutes change": ["min_gap_pts"], "+ both": ["fp_gap", "min_gap_pts"],
          "+ both + teammates out": ["fp_gap", "min_gap_pts", "out_share", "out_new_share"], "+ both, clean games": ["fp_gap_c", "min_gap_c"]}
res, coefs = {}, {}
def fit(cols, d):
    if not cols:
        return np.array([0.0])
    A = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
    return np.linalg.lstsq(A, d["target"] - d["after"], rcond=None)[0]
def pred(cols, b, d):
    if not cols:
        return d["after"].to_numpy()
    return d["after"].to_numpy() + b[0] + sum(b[i + 1] * d[c].to_numpy() for i, c in enumerate(cols))
rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
SUBSETS = {"all": te.index, "role up (minutes +5 over 3 games)": te.index[te["min_gap"] >= 5], "role down (minutes -5)": te.index[te["min_gap"] <= -5],
           "big teammate out (new, 30+ pts of production)": te.index[te["out_new"] >= 30], "hot (last 3 +10 over DELCO)": te.index[te["fp_gap"] >= 10],
           "a short game in the last 3": te.index[te["min_gap"] - (te["l3c_min"] - te["afterMIN"]) <= -3]}
print("Error predicting the next 3 games (fantasy pts/g, test seasons 2022-26):")
hdr = "  model".ljust(34) + "".join(f"{k[:26]:>28}" for k in SUBSETS)
print(hdr)
for name, cols in MODELS.items():
    b = fit(cols, tr)
    coefs[name] = dict(zip(["const"] + cols, [round(float(x), 3) for x in b]))
    p = pred(cols, b, te)
    yv = te["target"].to_numpy()
    row = {}
    for sname, idx in SUBSETS.items():
        m = te.index.isin(idx)
        row[sname] = (rmse(yv[m], p[m]), float(np.mean(yv[m] - p[m])), int(m.sum()))
    res[name] = row
    print(f"  {name:32s}" + "".join(f"{v[0]:>15.2f} (bias {v[1]:+.1f})" for v in row.values()))
print("  n:" + " " * 30 + "".join(f"{v[2]:>28d}" for v in res["DELCO only"].values()))
print("\nFitted corrections (added to DELCO):")
for k_, v in coefs.items():
    print(f"  {k_:32s} {v}")
# persistence of a role jump: how much of a +5 minutes jump is still there over the next 3 games
up = te[te["min_gap"] >= 5]
nxt_min = []
print(f"\nRole up: last-3 minutes {up['l3_min'].mean():.1f} vs DELCO belief {up['afterMIN'].mean():.1f}; points next 3 {up['target'].mean():.1f} vs DELCO {up['after'].mean():.1f} and last 3 {up['l3_fp'].mean():.1f}")
bo = te[te["out_new"] >= 30]
print(f"Big new absence: DELCO {bo['after'].mean():.1f}, next 3 {bo['target'].mean():.1f} (+{(bo['target'] - bo['after']).mean():.1f})")
json.dump({"results": {k_: {s: list(v) for s, v in r.items()} for k_, r in res.items()}, "coefs": coefs}, open(D / "short_term_study.json", "w"), indent=1)
