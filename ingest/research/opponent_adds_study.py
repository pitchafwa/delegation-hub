"""Strategy study (2026-10-25): can we anticipate an opponent's STREAMING (players he adds during the week), the way study #3 anticipates his lineup habits?

Data: data/league_days_<year>.json (every team's roster and lineup every day, 2023-24 .. 2025-26) and ESPN owners by season (habits follow the owner).
For every team-week:
  adds        players who appear on the roster after day 1 (new arrivals; IR moves are not arrivals)
  stream pts  points scored in starting slots by players who were NOT on the roster at the start of the week
  last-day    points on the final day by players added that day or the day before (the late streaming that swung DRNK-ROB)
Tests
  1. Is it an owner trait? split-half within a season, and season to season for the same owner.
  2. Forecast: predict each team-week's stream points (and last-day points) from that owner's EARLIER weeks only (shrunk to the league mean), vs the current
     assumption (0: the opponent's roster is fixed) and a league-wide average. Mean error, and how much of the streaming it explains.
  3. Does it depend on adds left going into the last day?
Excluded: DRNK (Tommy did not manage in past seasons) and computer-run teams.
Run from ingest/:  .venv/Scripts/python research/opponent_adds_study.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
STARTING = set(range(0, 12))
owners = {}
for y in (2024, 2025, 2026, 2027):
    for t in League(league_id=config.LEAGUE_ID, year=y, espn_s2=config.ESPN_S2, swid=config.SWID).espn_request.league_get(params={"view": ["mTeam"]})["teams"]:
        owners[(y, t["id"])] = t.get("primaryOwner") or ""
TOMMY = owners[(2027, 12)]
human = lambda y, t: bool(owners.get((y, t))) and owners[(y, t)] != TOMMY

rows = []
for y in (2024, 2025, 2026):
    days = json.load(open(R / "data" / f"league_days_{y}.json", encoding="utf-8"))
    wk = defaultdict(dict)                                   # (mp, team) -> {sp: entries}
    for sp, recs in days.items():
        for r in recs:
            wk[(r["mp"], r["team"])][int(sp)] = r["e"]
    for (mp, team), dd in wk.items():
        sps = sorted(dd)
        if len(sps) < 5 or len(sps) > 8 or not human(y, team):
            continue
        ros = [{e[0] for e in dd[sp]} for sp in sps]
        start = ros[0]
        arrivals = set().union(*ros[1:]) - start
        stream = sum((e[4] or 0) for sp in sps for e in dd[sp] if e[0] not in start and e[2] in STARTING and e[3])
        last = sps[-1]
        late_new = ros[-1] - (ros[-3] if len(ros) >= 3 else start)          # added on the last day or the day before
        last_pts = sum((e[4] or 0) for e in dd[last] if e[0] in late_new and e[2] in STARTING and e[3])
        arr_before_last = len((set().union(*ros[1:-1]) if len(ros) > 2 else set()) - start)
        rows.append(dict(y=y, mp=mp, team=team, owner=owners[(y, team)], adds=len(arrivals), stream=stream, last=last_pts, adds_before_last=arr_before_last, n=len(sps)))
print(f"{len(rows)} team-weeks (3 seasons, human owners other than DRNK)")
A = lambda k: np.array([r[k] for r in rows], float)
print(f"per week: adds {A('adds').mean():.1f} (of 7), streamed-player starter points {A('stream').mean():.0f}, last-day points from late adds {A('last').mean():.0f}")

# 1. owner trait
def season_means(key):
    m = defaultdict(list)
    for r in rows:
        m[(r["y"], r["owner"])].append(r)
    return m
m = season_means("stream")
odd = [(np.mean([r["stream"] for r in v if r["mp"] % 2]), np.mean([r["stream"] for r in v if not r["mp"] % 2])) for v in m.values() if len(v) > 8]
a = np.array(odd)
print(f"\n1. owner trait: split-half (odd vs even weeks) r = {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.2f} (stream points), n={len(a)} owner-seasons")
ss = []
for (y, o), v in m.items():
    w = m.get((y + 1, o))
    if w:
        ss.append((np.mean([r["stream"] for r in v]), np.mean([r["stream"] for r in w])))
if len(ss) > 3:
    b = np.array(ss)
    print(f"   season to next season, same owner: r = {np.corrcoef(b[:, 0], b[:, 1])[0, 1]:.2f} (n={len(b)})")
by_owner = sorted(((np.mean([r["stream"] for r in v]), y, o[:6]) for (y, o), v in m.items()), reverse=True)
print("   range across owner-seasons: " + ", ".join(f"{v:.0f}" for v, _, _ in by_owner[::max(1, len(by_owner) // 8)]))

# 2. forecast from earlier weeks only
def forecast(key, k_shrink):
    errs = {"zero (current)": [], "league average": [], "owner habit": []}
    expl = []
    for r in rows:
        prior = [q for q in rows if q["owner"] == r["owner"] and (q["y"], q["mp"]) < (r["y"], r["mp"])][-15:]
        lg = [q[key] for q in rows if (q["y"], q["mp"]) < (r["y"], r["mp"])]
        if len(lg) < 30:
            continue
        lgm = float(np.mean(lg))
        own = (sum(q[key] for q in prior) + k_shrink * lgm) / (len(prior) + k_shrink)
        errs["zero (current)"].append(abs(r[key]))
        errs["league average"].append(abs(r[key] - lgm))
        errs["owner habit"].append(abs(r[key] - own))
        expl.append((own, r[key]))
    e = np.array(expl)
    return {k: float(np.mean(v)) for k, v in errs.items()}, float(np.corrcoef(e[:, 0], e[:, 1])[0, 1]), len(e)
print("\n2. forecast error (points) using only earlier weeks:")
for key, lbl in (("stream", "whole-week streaming points"), ("last", "last-day late-add points")):
    for k in (4, 8):
        res, rr, n = forecast(key, k)
        print(f"   {lbl}, shrink {k}: " + "   ".join(f"{a_} {b_:.0f}" for a_, b_ in res.items()) + f"   (corr {rr:.2f}, n={n})")

# 3. adds left going into the last day
print("\n3. last-day late-add points by adds already used before the last day:")
for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 99)):
    v = [r["last"] for r in rows if lo <= r["adds_before_last"] < hi]
    if v:
        print(f"   {lo}-{min(hi, 7) - 1 if hi < 99 else '6+'} used: {np.mean(v):.0f} pts (n={len(v)})")
json.dump(rows, open(R / "data" / "opponent_adds_study.json", "w"))
