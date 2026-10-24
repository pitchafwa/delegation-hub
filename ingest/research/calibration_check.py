"""Calibration sweep (strategy study #9, set up 2026-10-24): are the numbers the plan relies on still right, now that this season's results are coming in?

Each check reports its sample so far and whether that is enough to act on. Nothing is changed automatically.
  1. WIN CHANCE       the plan's win chance for every team at the first run of each matchup (ledger "league" records) vs the real result  (want 60+ team-matchups)
  2. OPPONENT HABITS  realistic projection (perfect-lineup projection x owner efficiency, study #3) vs perfect-lineup projection, against real totals
  3. INJURY TAGS      how often Questionable / Doubtful / Probable players actually played (DailyFantasyFuel tags at tip-off; grade_dfs.py) vs the plan's 57% / 4% / 92%
  4. DROPPED PLAYERS  share of players dropped in this league who were added by another team within 2 days, by points per game, vs the plan's 25% / 16% / 12% (RECLAIM_LOSE 15%)
  5. DFF ACCURACY     grade_dfs.py's report (accuracy and how fast it reacts to injury news)
  6. ADDS             adds used per matchup by Tommy vs the 7 allowed (unused adds expire)
Writes ingest/research/data/calibration_report.txt.  Run from ingest/:  .venv/Scripts/python research/calibration_check.py
"""
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
ROOT = R.parent.parent
out = []
P = lambda s="": (out.append(s), print(s))
import config
from espn_api.basketball import League

lg = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)
raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mTeam"]})
final = {}
for m in raw["schedule"]:
    if "away" in m and m.get("winner") in ("HOME", "AWAY", "TIE"):
        for me, op in (("home", "away"), ("away", "home")):
            final[(m["matchupPeriodId"], m[me]["teamId"])] = (float(m[me].get("totalPoints") or 0), float(m[op].get("totalPoints") or 0))

recs = []
for f in sorted((ROOT / "ledger").glob("plan-*.jsonl")):
    for line in open(f, encoding="utf-8"):
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
league = [r for r in recs if r.get("kind") == "league"]
first = {}
for r in sorted(league, key=lambda r: r["ts"]):
    first.setdefault(r["matchup"], r)

# 1 + 2
P("1. WIN CHANCE (first plan run of each matchup vs the real result)")
pw, ys, err_real, err_raw = [], [], [], []
for mp, r in first.items():
    for t in r["teams"]:
        res = final.get((mp, t["id"]))
        if not res or t.get("win_prob") is None:
            continue
        pw.append(t["win_prob"])
        ys.append(1.0 if res[0] > res[1] else 0.0)
        raw_total = t["pts_so_far"] + t["expected"]
        real_total = t["pts_so_far"] + t["expected"] * (t.get("eff") or 1.0)
        err_raw.append(abs(raw_total - res[0]))
        err_real.append(abs(real_total - res[0]))
n = len(pw)
if n < 60:
    P(f"   {n} team-matchups so far (want 60+, about 5 matchups)")
else:
    pw, ys = np.array(pw), np.array(ys)
    P(f"   {n} team-matchups: Brier {np.mean((pw - ys) ** 2):.3f} (coin flip 0.250)")
    for lo, hi in ((0, .3), (.3, .45), (.45, .55), (.55, .7), (.7, 1.01)):
        m = (pw >= lo) & (pw < hi)
        if m.sum():
            P(f"   predicted {lo:.0%}-{min(hi, 1):.0%}: won {ys[m].mean():.0%} of {int(m.sum())} (average prediction {pw[m].mean():.0%})")
P("\n2. OPPONENT HABITS (week totals)")
if len(err_real) < 60:
    P(f"   {len(err_real)} team-matchups so far (want 60+)")
else:
    P(f"   error with owner habits {np.mean(err_real):.0f} pts vs assuming perfect lineups {np.mean(err_raw):.0f} pts  (n={len(err_real)})")

# 3 + 5: DailyFantasyFuel grading (local log)
P("\n3 + 5. INJURY TAGS and DAILYFANTASYFUEL (grade_dfs.py)")
try:
    rep = subprocess.run([sys.executable, str(R / "grade_dfs.py")], capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=str(R.parent)).stdout
    keep = [l for l in rep.splitlines() if l.strip()]
    P("   " + "\n   ".join(keep[:60]))
    P("   (plan assumes: Questionable 57%, Doubtful 4%, Probable 92%, available 96% for rotation players)")
except Exception as ex:
    P(f"   grade_dfs.py failed: {ex!r}"[:200])

# 4: dropped players taken within 2 days
P("\n4. DROPPED PLAYERS: taken by another team within 2 days")
try:
    acts = lg.recent_activity(size=1000)
    drops, adds = [], defaultdict(list)
    for a in acts:
        t = datetime.fromtimestamp(a.date / 1000)
        for team, action, player, *_ in a.actions:
            act = str(action).upper()
            nm = getattr(player, "name", None) or str(player)
            if "DROP" in act:
                drops.append((t, getattr(team, "team_id", None), nm, nm))
            elif "ADD" in act or "WAIVER" in act:
                adds[nm].append((t, getattr(team, "team_id", None)))
    rd = {}
    rp = ROOT / "dashboard" / "redraft_data.json"
    if rp.exists():
        rd = {p["name"]: p.get("ppg") or 0 for p in json.load(open(rp, encoding="utf-8"))["players"]}
    bands = defaultdict(lambda: [0, 0])
    for t, tid, name, pid in drops:
        if t > datetime.now() - timedelta(days=2):
            continue                                       # too recent to know
        ppg = rd.get(pid, 0)
        b = "34+" if ppg >= 33 else ("27-34" if ppg >= 27 else "under 27")
        taken = any(t < ta <= t + timedelta(days=2) and ttid != tid for ta, ttid in adds.get(pid, []))
        bands[b][0] += 1
        bands[b][1] += taken
    tot = sum(v[0] for v in bands.values())
    if tot < 40:
        P(f"   {tot} drops old enough to judge so far (want 40+)")
    for b in ("34+", "27-34", "under 27"):
        nb, tk = bands[b]
        if nb:
            P(f"   {b:9s}: {tk}/{nb} taken ({tk / nb:.0%})   plan assumes {dict([('34+', '25%'), ('27-34', '16%'), ('under 27', '12%')])[b]}")
except Exception as ex:
    P(f"   transaction history unavailable: {ex!r}"[:200])

# 6: adds used
P("\n6. ADDS USED PER MATCHUP (yours)")
last = {}
for r in sorted([r for r in recs if r.get("kind") == "plan"], key=lambda r: r["ts"]):
    last[r["matchup"]] = (r.get("adds_used"), r.get("adds_limit"))
done = {mp: v for mp, v in last.items() if any(k[0] == mp for k in final)}
if not done:
    P("   no finished matchups yet")
for mp, (u, lim) in sorted(done.items()):
    P(f"   matchup {mp}: {u} of {lim}")
(R / "data" / "calibration_report.txt").write_text("\n".join(out), encoding="utf-8")
