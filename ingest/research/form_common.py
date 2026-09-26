"""Shared pieces for the "why is he hot" work: league fantasy-point formula, unified game logs (2010-11 .. 2025-26), the form split itself, and the fitted persistence weights.

Form split: recent window (last 10 games played) vs baseline (his previous 365 days of games) -> what changed
   minutes   he plays more/fewer minutes                                   (m_w - m_b) x baseline points per minute
   luck      makes more/fewer shots than his own baseline percentages       4(3PM - p3 3PA) + 2(2PM - p2 2PA) + 3(FTM - pFT FTA)   per game
   volume    takes more/fewer shots per minute (at baseline percentages)   m_w x change in expected shooting points per minute
   other     rebounds, assists, steals, blocks, turnovers per minute        the rest
   the four add up exactly to (recent points/game - baseline points/game)
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent / "data" / "game_logs"
LG = dict(p3=0.355, p2=0.535, pft=0.775)
import os
_KM = float(os.environ.get("FORM_K_MULT", 1))
K = dict(p3=250.0 * _KM, p2=300.0 * _KM, pft=120.0 * _KM)      # pseudo-attempts pulling a player's own percentage toward the league mean
COLS = ["pid", "name", "team", "gid", "date", "season", "min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts"]


def fp(df):
    tds = ((df.pts >= 10).astype(int) + (df.reb >= 10) + (df.ast >= 10) + (df.stl >= 10) + (df.blk >= 10)) >= 3
    return df.pts + 1.5 * df.reb + 2 * df.ast + 3 * df.stl + 3 * df.blk + df.fg3m + 2 * df.ftm - df.fta - df.tov + 3 * tds.astype(float)


def _mins(s):
    if isinstance(s, str) and ":" in s:
        a, b = s.split(":")[:2]
        return int(a) + int(b) / 60.0
    try:
        return float(s)
    except Exception:
        return 0.0


def load_all(cache=True):
    p = Path(__file__).resolve().parent / "data" / "form" / "games.pkl"
    if cache and p.exists():
        return pd.read_pickle(p)
    fs = []
    for i in (1, 2, 3):
        r = pd.read_csv(D / f"regular_season_box_scores_2010_2024_part_{i}.csv", usecols=["season_year", "game_date", "gameId", "teamTricode", "personId", "personName", "minutes", "fieldGoalsMade",
                        "fieldGoalsAttempted", "threePointersMade", "threePointersAttempted", "freeThrowsMade", "freeThrowsAttempted", "reboundsTotal", "assists", "steals", "blocks", "turnovers", "points"])
        r.columns = ["season", "date", "gid", "team", "pid", "name", "min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts"][:len(r.columns)] if False else \
            ["season", "date", "gid", "team", "pid", "name", "min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts"]
        fs.append(r)
    old = pd.concat(fs, ignore_index=True)
    old["min"] = old["min"].map(_mins)
    new = []
    for fpath in sorted(D.glob("nba_api_*.csv")):
        s = fpath.stem.replace("nba_api_", "")
        a = pd.read_csv(fpath)
        new.append(pd.DataFrame({"season": s, "date": a.GAME_DATE, "gid": a.GAME_ID, "team": a.TEAM_ABBREVIATION, "pid": a.PLAYER_ID, "name": a.PLAYER_NAME, "min": a.MIN.astype(float), "fgm": a.FGM, "fga": a.FGA,
                                 "fg3m": a.FG3M, "fg3a": a.FG3A, "ftm": a.FTM, "fta": a.FTA, "reb": a.REB, "ast": a.AST, "stl": a.STL, "blk": a.BLK, "tov": a.TOV, "pts": a.PTS}))
    g = pd.concat([old] + new, ignore_index=True)
    g["date"] = pd.to_datetime(g["date"])
    g = g.dropna(subset=["min"]).copy()
    g = g[g["min"] > 0].copy()
    for c in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "ast", "stl", "blk", "tov", "pts"]:
        g[c] = g[c].fillna(0).astype(float)
    g["fp"] = fp(g)
    g = g.sort_values(["pid", "date", "gid"]).reset_index(drop=True)
    if cache:
        g.to_pickle(p)
    return g


def add_absence(g):
    """absent_min: minutes-per-game of the team's regulars (>=18 mpg over his previous 15 team games, appeared in the previous 3 team games) who did not play this game"""
    out = np.zeros(len(g))
    g = g.reset_index(drop=True)
    for (season, team), t in g.groupby(["season", "team"], sort=False):
        gids = list(dict.fromkeys(t.sort_values("date")["gid"]))
        gpos = {x: i for i, x in enumerate(gids)}
        played = {}
        mins = {}
        for pid, gi, m in zip(t.pid.values, t.gid.values, t["min"].values):
            played.setdefault(pid, {})[gpos[gi]] = m
        for pid, dct in played.items():
            pass
        # for every team game index i, regulars from window (i-15, i)
        reg_cache = {}
        for i in range(len(gids)):
            regs = []
            for pid, dct in played.items():
                w = [dct[j] for j in range(max(0, i - 15), i) if j in dct]
                if len(w) >= 8 and np.mean(w) >= 18 and any((i - k) in dct for k in (1, 2, 3)):
                    regs.append((pid, np.mean(w)))
            reg_cache[i] = regs
        idx = t.index.values
        for ix, pid, gi in zip(idx, t.pid.values, t.gid.values):
            i = gpos[gi]
            out[ix] = sum(m for q, m in reg_cache[i] if i not in played[q])
    return out


def split(w, b, shrink=None):
    """w, b: DataFrames of games (window / baseline). returns dict of components (all in fantasy points per game)"""
    def tot(x):
        return {c: x[c].sum() for c in ["min", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "fp", "reb", "ast", "stl", "blk", "tov"]}, len(x)
    tw, nw = tot(w)
    tb, nb = tot(b)
    fg2m_b, fg2a_b = tb["fgm"] - tb["fg3m"], tb["fga"] - tb["fg3a"]
    p3 = (tb["fg3m"] + K["p3"] * LG["p3"]) / (tb["fg3a"] + K["p3"])
    p2 = (fg2m_b + K["p2"] * LG["p2"]) / (fg2a_b + K["p2"])
    pf = (tb["ftm"] + K["pft"] * LG["pft"]) / (tb["fta"] + K["pft"])
    fg2m_w, fg2a_w = tw["fgm"] - tw["fg3m"], tw["fga"] - tw["fg3a"]
    luck3 = 4 * (tw["fg3m"] - p3 * tw["fg3a"]) / nw
    luck2 = 2 * (fg2m_w - p2 * fg2a_w) / nw
    luckf = 3 * (tw["ftm"] - pf * tw["fta"]) / nw
    luck = luck3 + luck2 + luckf
    ex = lambda t_, n_: (4 * p3 * t_["fg3a"] + 2 * p2 * (t_["fga"] - t_["fg3a"]) + (3 * pf - 1) * t_["fta"]) / n_
    m_w, m_b = tw["min"] / nw, tb["min"] / nb
    fp_w, fp_b = tw["fp"] / nw, tb["fp"] / nb
    r_b = fp_b / m_b
    mins_eff = (m_w - m_b) * r_b
    rate_eff = m_w * (fp_w / m_w - r_b)                 # = fp_w - m_w * r_b
    vol = m_w * (ex(tw, nw) / m_w - ex(tb, nb) / m_b)
    other = rate_eff - luck - vol
    pm = lambda c, wgt: wgt * m_w * (tw[c] / tw["min"] - tb[c] / tb["min"])
    o_reb, o_ast, o_stk, o_tov = pm("reb", 1.5), pm("ast", 2.0), pm("stl", 3.0) + pm("blk", 3.0), pm("tov", -1.0)
    o_rest = other - o_reb - o_ast - o_stk - o_tov
    return dict(luck3=luck3, luck2=luck2, luckf=luckf, o_reb=o_reb, o_ast=o_ast, o_stk=o_stk, o_tov=o_tov, o_rest=o_rest, fp_w=fp_w, fp_b=fp_b, m_w=m_w, m_b=m_b, minutes=mins_eff, luck=luck, volume=vol, other=other, p3=p3, p2=p2, pft=pf)
