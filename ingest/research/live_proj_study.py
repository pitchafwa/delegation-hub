"""Live in-game final projection: backtest (2026-10-31, Tommy). Which formula best predicts a player's FINAL fantasy points from the middle of his game?
Same idea as the Winner's Circle model (league-hub/ingest/LIVE_PROJECTION_RESEARCH.md): shrink tonight's pace toward the pregame number by real sample size
(here: minutes played), cap early outliers, and project his remaining MINUTES separately (foul trouble, blowouts, injury exits).

Data: data/pbp/pbp_<season>.csv (pull_pbp.py) replayed action by action -> each player's fantasy points, minutes, fouls and the score margin every 3 game
minutes; pregame number = DELCO's real pregame projection for that game and its projected minutes (data/delco_pregame_all.csv.gz: pre, preMIN); actual
final = the box score (data/game_logs/nba_api_<season>.csv). The replay is checked against the box score first.
Methods compared (mean absolute error of the projected final, on games held out of the fitting):
  freeze  = points so far + pregame x share of the game left            (what "still to come" does now)
  pace    = points so far / share of the game played                     (naive pace)
  model   = points so far + expected remaining minutes x blended rate    (fitted here)
Run from ingest/:  .venv/Scripts/python research/live_proj_study.py [season]
"""
import json
import re
import unicodedata
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
import os
DEBUG = bool(os.environ.get("LP_DEBUG"))
FP = dict(PTS=1, REB=1.5, AST=2, STL=3, BLK=3, FG3M=1, FTM=2, FTA=-1, TOV=-1)


def fold(n):
    return unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().strip().lower()


def secs(clock):
    m = re.match(r"PT(\d+)M([\d.]+)S", str(clock))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


def plen(p):
    return 720.0 if p <= 4 else 300.0


def elapsed(p, clock):
    return sum(plen(q) for q in range(1, p)) + plen(p) - secs(clock)


pbp = pd.read_csv(D / "pbp" / f"pbp_{season}.csv", dtype={"gameId": str}, keep_default_na=False)
pbp["gameId"] = pbp["gameId"].str.zfill(10)
for c in ("period", "personId", "teamId", "actionNumber"):
    pbp[c] = pd.to_numeric(pbp[c], errors="coerce").fillna(0).astype(int)
box = pd.read_csv(D / "game_logs" / f"nba_api_{season}.csv", dtype={"GAME_ID": str})
box["GAME_ID"] = box["GAME_ID"].str.zfill(10)
box["fp"] = sum(box[k] * v for k, v in FP.items())
box_fp = {(g, p): (f, m) for g, p, f, m in zip(box.GAME_ID, box.PLAYER_ID, box.fp, box.MIN)}
gdate = dict(zip(box.GAME_ID, box.GAME_DATE))
# what the live tracker can know before the game without DELCO's minutes: his average minutes over his previous 10 games (redraft_data m10)
box = box.sort_values("GAME_DATE")
box["m10"] = box.groupby("PLAYER_ID")["MIN"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
m10 = {(g, p): v for g, p, v in zip(box.GAME_ID, box.PLAYER_ID, box.m10)}
home_team = {}
for g, sub in box.groupby("GAME_ID"):
    for t, m in zip(sub.TEAM_ID, sub.MATCHUP):
        if " vs. " in str(m):
            home_team[g] = t
dl = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "pre", "preMIN"])
pre = {(p, d): (a, b) for p, d, a, b in zip(dl.PLAYER_ID, dl.GAME_DATE, dl.pre, dl.preMIN)}

CHECK = [x * 60.0 for x in range(3, 48, 3)]          # every 3 game minutes
rows, check_fp, check_min = [], [], []
for gid, g in pbp.groupby("gameId", sort=False):            # the feed's own row order (actionNumber is not in time order)
    if gid not in gdate:
        continue
    g = g.copy()
    g["t"] = [elapsed(p, c) for p, c in zip(g.period, g.clock)]
    team_of, last_names = {}, defaultdict(dict)
    _cnt = g[g.personId > 0].personId.value_counts().to_dict()
    for pid, tid, nm in zip(g.personId, g.teamId, g.playerNameI):
        if pid > 0 and tid > 0 and nm:
            team_of[pid] = tid
            k_ = fold(nm.split(". ", 1)[-1])
            if k_ not in last_names[tid] or _cnt.get(pid, 0) > _cnt.get(last_names[tid][k_], 0):      # two players with one last name: the busier one
                last_names[tid][k_] = pid
            last_names[tid][fold(nm)] = pid                 # "G. Antetokounmpo": descriptions use the initial when two teammates share a last name
    stat = defaultdict(lambda: defaultdict(float))
    on = defaultdict(set)               # team -> players on court
    since = {}                          # player -> time he came on
    mins = defaultdict(float)
    fouls = defaultdict(int)
    snaps = []
    ci = 0
    hs = as_ = 0
    # who starts each period on the court: appears in an action, or is subbed OUT, before being subbed IN in that period
    starters = defaultdict(set)
    for p, sub in g.groupby("period"):
        seen_in = set()
        for _, r in sub.iterrows():
            if r.actionType == "Substitution":
                m = re.match(r"SUB: (.+?) FOR (.+)$", r.description)
                if m and r.personId > 0 and r.personId not in seen_in:
                    starters[p].add(r.personId)
                inn = last_names[r.teamId].get(fold(m.group(1))) if m else None
                if inn:
                    seen_in.add(inn)
            elif r.personId > 0 and r.personId in team_of and r.personId not in seen_in:
                starters[p].add(r.personId)
    cur_p = 0

    def snap(t):
        for pid in set(team_of):
            mn = mins[pid] + ((t - since[pid]) if pid in since else 0.0)
            f = sum(stat[pid][k] * v for k, v in FP.items())
            if mn <= 0 and f == 0:
                continue
            tm = team_of[pid]
            marg = (hs - as_) if tm == home_team.get(gid) else (as_ - hs)
            snaps.append((t, pid, f, mn / 60.0, fouls[pid], marg))

    for _, r in g.iterrows():
        t = r.t
        while ci < len(CHECK) and CHECK[ci] <= t:
            snap(CHECK[ci])
            ci += 1
        if r.period != cur_p:                                 # period change: close everyone, open this period's starters
            for pid in list(since):
                mins[pid] += sum(plen(q) for q in range(1, cur_p + 1)) - since[pid] if cur_p else 0.0
            since.clear()
            cur_p = r.period
            t0 = sum(plen(q) for q in range(1, cur_p))
            for pid in starters[cur_p]:
                since[pid] = t0
        pid, at, d = r.personId, r.actionType, r.description
        if at == "Substitution":
            m = re.match(r"SUB: (.+?) FOR (.+)$", d)
            if pid in since:
                mins[pid] += t - since.pop(pid)
            inn = last_names[r.teamId].get(fold(m.group(1))) if m else None
            if inn and inn not in since:
                since[inn] = t
        elif at == "Made Shot" and pid > 0:
            v = int(r.shotValue or 0) or (3 if "3PT" in d else 2)
            stat[pid]["PTS"] += v
            stat[pid]["FG3M"] += v == 3
            m = re.search(r"\(([^()]+?) \d+ AST\)", d)
            if m:
                a = last_names[r.teamId].get(fold(m.group(1)))
                if a:
                    stat[a]["AST"] += 1
        elif at == "Free Throw" and pid > 0:
            stat[pid]["FTA"] += 1
            if not d.startswith("MISS"):
                stat[pid]["FTM"] += 1
                stat[pid]["PTS"] += 1
        elif at == "Rebound" and pid > 0 and pid in team_of:
            stat[pid]["REB"] += 1
        elif at == "Turnover" and pid > 0 and pid in team_of:
            stat[pid]["TOV"] += 1
        elif at == "Foul" and pid > 0 and "Technical" not in str(r.subType):
            fouls[pid] += 1
        elif at == "" and pid > 0:
            if "STEAL" in d:
                stat[pid]["STL"] += 1
            elif "BLOCK" in d:
                stat[pid]["BLK"] += 1
        if str(r.scoreHome).strip() not in ("", "0") or str(r.scoreAway).strip() not in ("", "0"):
            try:
                hs, as_ = int(r.scoreHome), int(r.scoreAway)
            except ValueError:
                pass
    end = sum(plen(q) for q in range(1, cur_p + 1))
    for pid in list(since):
        mins[pid] += end - since[pid]
    while ci < len(CHECK):
        snap(min(CHECK[ci], end))
        ci += 1
    for pid in team_of:
        b = box_fp.get((gid, pid))
        if b:
            check_fp.append((sum(stat[pid][k] * v for k, v in FP.items()), b[0]))
            if DEBUG and abs(check_fp[-1][0] - b[0]) > 2 and len(check_fp) < 400:
                bx = box[(box.GAME_ID == gid) & (box.PLAYER_ID == pid)].iloc[0]
                print("DBG", bx.PLAYER_NAME, {k: (stat[pid][k], bx[k]) for k in FP if stat[pid][k] != bx[k]})
            check_min.append((mins[pid] / 60.0, b[1]))
            if DEBUG and abs(mins[pid] / 60.0 - b[1]) > 3 and len(check_min) < 300:
                print("DBM", gid, pid, round(mins[pid] / 60.0, 1), b[1], {q: pid in starters[q] for q in starters})
    for (t, pid, f, mn, pf, marg) in snaps:
        b = box_fp.get((gid, pid))
        pr = pre.get((pid, gdate[gid]))
        if b and pr and not np.isnan(pr[0]):
            rows.append(dict(game=gid, pid=pid, t=t / 60.0, fp=f, mn=mn, pf=pf, marg=marg, y=b[0], ymin=b[1], pre=pr[0], preMIN=pr[1], m10=m10.get((gid, pid))))

cf, cm = np.array(check_fp), np.array(check_min)
print(f"replay check vs box score ({len(cf)} player-games): fantasy points exact {np.mean(np.abs(cf[:, 0] - cf[:, 1]) < 0.01):.1%}, within 2: {np.mean(np.abs(cf[:, 0] - cf[:, 1]) <= 2):.1%};"
      f" minutes within 1: {np.mean(np.abs(cm[:, 0] - cm[:, 1]) <= 1):.1%}, within 3: {np.mean(np.abs(cm[:, 0] - cm[:, 1]) <= 3):.1%}")
R = pd.DataFrame(rows)
R.to_pickle(D / "_live_proj_rows.pkl")
print(f"{len(R)} snapshots, {R.game.nunique()} games, {R.pid.nunique()} players")
