"""Study 5: how long is a player out, and how does he look when he comes back?  (feeds the injured-player advisor)

A. TIME OUT.  From the official reports (injury_episodes.pkl, 2021-22 to 2025-26): for an absence that has already lasted s game days Out, what share of comparable absences
   (same body-part group and injury tier) were over within k more games?  Cells with fewer than 40 episodes fall back to the tier, then to all injuries.  Absences still open when a
   season ended are censored (counted as not yet returned), which slightly overstates the long-injury tail.  Scored out of sample: fit on 2021-22..2023-24, tested on 2024-26.
B. THE RAMP.  From 14 seasons of box scores (DNP rows included): for every return after 5+ straight missed team games by a rotation player (previous-15-game minutes 15+),
   minutes and fantasy points in his first games back as a fraction of his pre-injury level, by length of absence.
Writes injury_return_model.json.
Run from ingest/:  uv run python research/injury_return_study.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
D = R / "data"
ep = pd.read_pickle(D / "injury_episodes.pkl")
rows = pd.read_pickle(D / "injury_rows_05pm.pkl")
season_last = rows.groupby("season").gd.max().to_dict()
ep = ep[ep.n_out >= 1].copy()
ep["censored"] = ep.apply(lambda r: (season_last[r.season] - r.end).days <= 10, axis=1)
ep["tier2"] = ep.tier.replace({"recovery": "major", "management": "moderate", "other": "moderate"})
SB = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 10000)]
KS = [1, 2, 3, 5, 8, 12, 20, 30, 45]


def sbin(s):
    for i, (a, b) in enumerate(SB):
        if a <= s <= b:
            return i
    return len(SB) - 1


def cell_curve(df, si):
    """P(returned within k more games | out at least s games) for s in the bin, pooled over the s values present; also n"""
    a, b = SB[si]
    s_rep = a if a == b else int(round((a + min(b, 60)) / 2))
    at = df[df.n_out >= s_rep]
    n = len(at)
    if n == 0:
        return None, 0
    rem = at.n_out - s_rep
    cens = at.censored.to_numpy()
    p = [float(((rem <= k) & ~cens).mean()) for k in KS]
    return p, n


def build(df):
    out = {}
    for si in range(len(SB)):
        for key, sub in (("all", df),):
            c, n = cell_curve(sub, si)
            out[f"all|all|{si}"] = (c, n)
        for tier, sub in df.groupby("tier2"):
            c, n = cell_curve(sub, si)
            out[f"all|{tier}|{si}"] = (c, n)
        for (grp, tier), sub in df.groupby(["group", "tier2"]):
            c, n = cell_curve(sub, si)
            out[f"{grp}|{tier}|{si}"] = (c, n)
    return out


def lookup(cells, grp, tier, si, min_n=40):
    for key in (f"{grp}|{tier}|{si}", f"all|{tier}|{si}", f"all|all|{si}"):
        c, n = cells.get(key, (None, 0))
        if c is not None and n >= min_n:
            return c, n, key
    return cells[f"all|all|{si}"][0], cells[f"all|all|{si}"][1], "all"


# ---- out-of-sample check of A: fit <= 2023-24, test 2024-26. For each test episode and each s in {1,3,6,10}: predicted P(back within 5 games) vs actual
tr = ep[ep.season <= "2023-24"]
te = ep[ep.season > "2023-24"]
cells_tr = build(tr)
chk = []
for r in te.itertuples():
    for s in (1, 3, 6, 10):
        if r.n_out < s:
            continue
        c, n, _ = lookup(cells_tr, r.group, r.tier2, sbin(s))
        rem = r.n_out - s
        for ki, k in enumerate(KS):
            if k in (3, 8, 20):
                chk.append(dict(s=s, k=k, pred=c[ki], act=float(rem <= k and not r.censored)))
chk = pd.DataFrame(chk)
print("A. OUT OF SAMPLE (fit 2021-24, test 2024-26): predicted vs actual share back within k more games")
print(chk.groupby(["s", "k"]).agg(pred=("pred", "mean"), actual=("act", "mean"), n=("act", "size")).round(3).to_string())
cells = build(ep)
print("\nexample: knee, major, out 8+ games: P(back within k more) for k in", KS)
c, n, key = lookup(cells, "knee", "major", sbin(10))
print(np.round(c, 2), n, key)
c, n, key = lookup(cells, "ankle_foot", "moderate", sbin(2))
print("ankle/foot moderate, out 2-3 games:", np.round(c, 2), n, key)


# ---- B. the ramp (box scores)
sys.path.insert(0, str(R))
fs = [D / "game_logs" / f"regular_season_box_scores_2010_2024_part_{i}.csv" for i in (1, 2, 3)]
cols = ["season_year", "game_date", "gameId", "teamTricode", "personId", "comment", "minutes", "points", "reboundsTotal", "assists", "steals", "blocks", "turnovers", "threePointersMade", "freeThrowsMade", "freeThrowsAttempted"]
d = pd.concat([pd.read_csv(f, usecols=cols) for f in fs], ignore_index=True).drop_duplicates(["gameId", "personId"])


def mins(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, str) and ":" in x:
        a, b = x.split(":")
        return float(a) + float(b) / 60
    try:
        return float(x)
    except Exception:
        return 0.0


d["min"] = d.minutes.map(mins)
d["played"] = d["min"] > 0
for c in cols[7:]:
    d[c] = d[c].fillna(0)
cats = (d[["points", "reboundsTotal", "assists", "steals", "blocks"]] >= 10).sum(axis=1)
d["fp"] = (d.points + 1.5 * d.reboundsTotal + 2 * d.assists + 3 * d.steals + 3 * d.blocks + d.threePointersMade + 2 * d.freeThrowsMade - d.freeThrowsAttempted - d.turnovers + 3 * (cats >= 3)).where(d.played, 0.0)
d["gd"] = pd.to_datetime(d.game_date)
d = d.sort_values(["personId", "gd", "gameId"]).reset_index(drop=True)
out = []
for pid, g in d.groupby("personId", sort=False):
    pl = g.played.to_numpy()
    mn = g["min"].to_numpy()
    fp = g.fp.to_numpy()
    n = len(g)
    i = 0
    while i < n:
        if pl[i]:
            i += 1
            continue
        j = i
        while j < n and not pl[j]:
            j += 1
        miss = j - i
        if miss >= 5 and j < n and i >= 10:
            pre_idx = np.where(pl[max(0, i - 25):i])[0]
            if len(pre_idx) >= 8:
                base_m = mn[max(0, i - 25):i][pre_idx][-15:].mean()
                base_f = fp[max(0, i - 25):i][pre_idx][-15:].mean()
                if base_m >= 15:
                    # first games back: the next 10 rows that were played (a missed row in between counts as a miss)
                    nxt = [(mn[k], fp[k], pl[k]) for k in range(j, min(n, j + 10))]
                    out.append(dict(miss=miss, base_m=base_m, base_f=base_f, mins=[x[0] for x in nxt], fps=[x[1] for x in nxt], played=[x[2] for x in nxt]))
        i = j
print(f"\nB. {len(out):,} returns after 5+ missed team games by rotation players")
buckets = [(5, 9), (10, 24), (25, 49), (50, 1000)]
ramp = {}
for a, b in buckets:
    sel = [o for o in out if a <= o["miss"] <= b]
    if len(sel) < 30:
        continue
    row = {}
    for k in (1, 2, 3, 5, 8, 10):
        mr, fr, pr = [], [], []
        for o in sel:
            if len(o["mins"]) >= k:
                pr.append(o["played"][k - 1])
                if o["played"][k - 1]:                      # minutes and points ONLY in games he actually plays; the play rate is separate
                    mr.append(o["mins"][k - 1] / o["base_m"])
                    fr.append(o["fps"][k - 1] / max(o["base_f"], 1))
        row[k] = dict(minutes=round(float(np.mean(mr)), 3), fp=round(float(np.mean(fr)), 3), played=round(float(np.mean(pr)), 3), n=len(pr))
    ramp[f"{a}-{b}"] = row
    print(f"absence {a}-{b} games (n={len(sel)}): " + "  ".join(f"g{k}: min {row[k]['minutes']:.2f} fp {row[k]['fp']:.2f} play {row[k]['played']:.2f}" for k in (1, 2, 3, 5, 8, 10)))
json.dump({"ks": KS, "sbins": SB, "cells": {k: {"p": v[0], "n": v[1]} for k, v in cells.items() if v[0] is not None}, "ramp": ramp,
           "note": "P(returned within k more games | out at least s games) by body-part group and tier (reports 2021-26); ramp = first games back / pre-injury level (box scores 2010-24)"},
          open(R / "injury_return_model.json", "w"))
print("wrote injury_return_model.json")
