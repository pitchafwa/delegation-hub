"""Rebuild the Standings tab's odds history (dashboard/odds_history.json) as if today's model had been in use every day (2026-11-07, Tommy: the title-odds
sparkline jumped with each model change).

For every day in the history: the inputs the site really had that day (the last commit of that day that touched playoff_odds.json: week_plan.json,
nba_schedule.json, week_forecasts.json), the season forecast rebuilt with today's build_schedule_plan.py (later-week decay, return curves where that day's
week_plan.json has injury details, otherwise the old injured-player rule), the standings from matchups finished before that day, and the season
simulation with today's settings (week-dependent drift). Nothing from after that day is used, except the final scores of matchups already over by then.
The old history is kept as odds_history_before_rebuild.json.
Run from ingest/:  python research/rebuild_odds_history.py [--dry]
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import build_playoff_odds as O
import season_calendar as SCAL
from playoff_sim import season_tau, simulate

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
ROOT = R.parent.parent
DASH = ROOT / "dashboard"
ET = ZoneInfo("America/New_York")
TMP = Path(os.environ.get("TEMP", "/tmp")) / "odds_rebuild"
FILES = ("week_plan.json", "nba_schedule.json", "week_forecasts.json")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, check=True).stdout


hist = json.load(open(DASH / "odds_history.json", encoding="utf-8"))
days = sorted(hist["days"])
log = git("log", "--format=%H %cI", "--", "dashboard/playoff_odds.json").decode().split("\n")
last_of_day = {}
for ln in log:
    if not ln.strip():
        continue
    h, iso = ln.split()
    d = datetime.fromisoformat(iso).astimezone(ET).date().isoformat()
    last_of_day.setdefault(d, h)            # log is newest first: the first seen is the day's last
teams, games = O.pull(2027)
cal = SCAL.load()
end_of = {p["id"]: date.fromisoformat(p["end"]) for p in cal["periods"]}
lens = {p["id"]: p["days"] for p in cal["periods"]}
PAR = O.PARAMS
ids = sorted(teams)
new_days = {}
for d in days:
    c = last_of_day.get(d)
    if not c:
        print(d, "no snapshot that day: kept as it was")
        new_days[d] = hist["days"][d]
        continue
    TMP.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        try:
            (TMP / f).write_bytes(git("show", f"{c}:dashboard/{f}"))
        except subprocess.CalledProcessError:
            if (TMP / f).exists():
                (TMP / f).unlink()
    # injury details for the return curve: snapshots before 2026-11-07 have no roster "inj" field, but every team's injury advisor block that day has the
    # same inputs (group, tier, games out so far, ESPN's return date). Players without an advisor entry keep the old injured-player rule.
    try:
        wp_ = json.load(open(TMP / "week_plan.json", encoding="utf-8"))
        n_fill = 0
        for t_ in wp_["teams"]:
            adv = {a["id"]: a for a in (t_.get("injury_advice") or [])}
            for p_ in t_["roster"]:
                a = adv.get(p_["id"])
                if p_.get("inj") is None and a and (a.get("injury") or {}).get("group"):
                    inj_ = a["injury"]
                    p_["inj"] = {"g": inj_["group"], "t": inj_.get("tier") or "moderate", "s": int(a.get("streak") or 1), "rd": (inj_.get("espn_return") or "")[:10] or None}
                    n_fill += 1
        (TMP / "week_plan.json").write_text(json.dumps(wp_), encoding="utf-8")
    except Exception as ex:
        n_fill = f"failed ({ex})"
    env = dict(os.environ, SP_HUB=str(TMP), SP_TODAY=d)
    r = subprocess.run([sys.executable, str(R / "build_schedule_plan.py")], cwd=R.parent, env=env, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(d, "forecast failed, kept as it was:", (r.stderr or "")[-300:])
        new_days[d] = hist["days"][d]
        continue
    sp = json.load(open(TMP / "schedule_plan.json", encoding="utf-8"))
    D = date.fromisoformat(d)
    done = [g for g in games if g["done"] and end_of[g["week"]] < D]
    wins, pf = O.standings(done)
    for t in ids:
        wins.setdefault(t, 0)
        pf.setdefault(t, 0.0)
    done_set = {(g["week"], g["a"], g["b"]) for g in done}
    pairs = [(g["week"], g["a"], g["b"]) for g in games if (g["week"], g["a"], g["b"]) not in done_set]
    cur = min((w for w, _, _ in pairs), default=O.REG_LAST + 1)
    mu = {(t["id"], int(w)): float(v) for t in sp["teams"] for w, v in t["proj"].items()}
    basis = sp["teams"][0].get("proj_basis")
    res = simulate(ids, wins, pf, pairs, mu, lens, O.REG_LAST, tau=season_tau(PAR, cur), sigma_e=PAR["sigma_e"],
                   bias=PAR["bias_blend"] if basis == "blend" else PAR["bias_preseason"])
    new_days[d] = {teams[t]: [round(float(res["made"][i]), 3), round(float(res["title"][i]), 3)] for i, t in enumerate(ids)}
    me = sp["my_abbrev"]
    print(f"{d}  week {cur:2d}  {basis:6s}  {me} title {hist['days'][d].get(me, [0, 0])[1]:.0%} -> {new_days[d][me][1]:.0%}   playoffs {hist['days'][d].get(me, [0, 0])[0]:.0%} -> {new_days[d][me][0]:.0%}   injury details filled {n_fill}", flush=True)
shutil.rmtree(TMP, ignore_errors=True)
if "--dry" not in sys.argv:
    shutil.copy(DASH / "odds_history.json", DASH / "odds_history_before_rebuild.json")
    h2 = json.load(open(DASH / "odds_history.json", encoding="utf-8"))           # re-read: keep any day written while this ran
    for d, v in new_days.items():
        h2["days"][d] = v
    (DASH / "odds_history.json").write_text(json.dumps(h2, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    print("wrote odds_history.json (old copy: odds_history_before_rebuild.json)")
