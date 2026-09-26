"""Live matchup tracker data: writes live_matchup.json (score so far, expected remaining points, win probability, and its history through the week).

Runs every 20 minutes from the alerts workflow (needs ESPN_S2 / SWID) and publishes the file to the `live-data` branch, which the page reads from raw.githubusercontent.com.
  score          ESPN's live matchup totals for the user's team and this week's opponent
  expected rest  players in each team's ACTUAL lineup today who have not played yet (level from the weekly plan x chance of playing), plus the plan's expected points for the days after today
  win chance     Phi( (margin now + 0.72 x expected remaining margin) / sigma(days left) ), fit and checked on 2025-26 (win_prob_study.py: calibrated within ~2 points)
  the cap        starts so far vs the games-played cap (a locked lineup ends scoring for the week)
Test on last season:  LIVE_YEAR=2026 LIVE_DATE=2026-01-15 LIVE_MP=13 uv run python research/live_matchup.py --dry
"""
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent.parent
DASH = ROOT / "dashboard"
RAW_URL = "https://raw.githubusercontent.com/pitchafwa/delegation-hub/live-data/live_matchup.json"
YEAR = int(os.environ.get("LIVE_YEAR", 2027))
SEASON_START = {2027: date(2026, 10, 20), 2026: date(2025, 10, 21)}.get(YEAR, date(2026, 10, 20))
TODAY = date.fromisoformat(os.environ["LIVE_DATE"]) if os.environ.get("LIVE_DATE") else datetime.now(ET).date()
NOW = datetime.now(ET)
DRY = "--dry" in sys.argv
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "ingest" / "live_out" / "live_matchup.json"
MODEL = json.load(open(Path(__file__).resolve().parent / "win_prob_model.json"))
BETA = MODEL["beta"]
SIG = {int(k): v for k, v in MODEL["sigma"].items()}
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
CAP7 = 40.0


def sigma(rem):
    if rem <= 0:
        return 1e-6
    if rem < 1:
        return SIG[1] * math.sqrt(rem)
    if rem <= 7:
        lo = int(rem)
        hi = min(7, lo + 1)
        return SIG[lo] + (SIG[hi] - SIG[lo]) * (rem - lo)
    return SIG[7] * math.sqrt(rem / 7.0)


def load(name):
    p = DASH / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main():
    if TODAY < SEASON_START and not os.environ.get("LIVE_YEAR"):
        print("before the season: nothing to track")
        return
    W = load("week_plan.json") or {}
    plan_ok = W.get("season") == YEAR
    my_ab = W.get("my_abbrev", "DRNK")
    lg = League(league_id=config.LEAGUE_ID, year=YEAR, espn_s2=config.ESPN_S2, swid=config.SWID)
    raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mTeam"]})
    teams = {t["id"]: (t.get("abbrev") or t.get("name") or str(t["id"])) for t in raw["teams"]}
    names = {t["id"]: ((t.get("location", "") + " " + t.get("nickname", "")).strip() or t.get("name", "")) for t in raw["teams"]}
    my_id = next((i for i, a in teams.items() if a == my_ab), None)
    if my_id is None:
        print("cannot find my team")
        return
    mp = int(os.environ.get("LIVE_MP") or raw.get("status", {}).get("currentMatchupPeriod") or 0)
    m = next((x for x in raw["schedule"] if x.get("matchupPeriodId") == mp and "away" in x and my_id in (x["home"]["teamId"], x["away"]["teamId"])), None)
    if m is None:
        print("no matchup found for period", mp)
        return
    mine_side, opp_side = ("home", "away") if m["home"]["teamId"] == my_id else ("away", "home")
    opp_id = m[opp_side]["teamId"]
    pts = {my_id: float(m[mine_side].get("totalPoints") or 0), opp_id: float(m[opp_side].get("totalPoints") or 0)}
    by_sp = {my_id: m[mine_side].get("pointsByScoringPeriod") or {}, opp_id: m[opp_side].get("pointsByScoringPeriod") or {}}
    if os.environ.get("LIVE_DATE"):                              # testing on a finished season: only count periods up to the test date
        lim = (TODAY - SEASON_START).days + 1
        for tid in pts:
            pts[tid] = float(sum(v for k, v in by_sp[tid].items() if int(k) <= lim))
    plan_match = plan_ok and W["matchup"]["id"] == mp
    days = [date.fromisoformat(d) for d in W["matchup"]["days"]] if plan_match else None
    if days is None:                                        # no matching plan: infer the week from the scoring periods ESPN has scored so far
        sp_keys = sorted(int(k) for k in by_sp[my_id])
        first = SEASON_START + timedelta(days=(sp_keys[0] - 1)) if sp_keys else TODAY
        days = [first + timedelta(days=i) for i in range(7)]
    cap = W["matchup"]["cap"] if plan_match else round(CAP7 * len(days) / 7.0, 1)
    sp_of = lambda d: (d - SEASON_START).days + 1
    today_sp = sp_of(TODAY)

    def box(sp):
        b = lg.espn_request.league_get(params={"view": "mBoxscore", "scoringPeriodId": sp})
        res = {my_id: [], opp_id: []}
        for mm in b.get("schedule", []):
            if mm.get("matchupPeriodId") != mp:
                continue
            for side in ("home", "away"):
                t = mm.get(side, {})
                if t.get("teamId") not in res:
                    continue
                for e in (t.get("rosterForCurrentScoringPeriod") or {}).get("entries", []):
                    pl = e.get("playerPoolEntry", {}).get("player", {})
                    st = [s for s in pl.get("stats", []) if s.get("scoringPeriodId") == sp and s.get("statSourceId") == 0]
                    played = bool(st and st[0].get("stats", {}).get("42", 0))
                    res[t["teamId"]].append(dict(id=pl.get("id"), name=pl.get("fullName"), slot=e.get("lineupSlotId"), played=played, pts=(st[0].get("appliedTotal") if st else None) or 0.0,
                                                 status=pl.get("injuryStatus"), team=pl.get("proTeamId")))
        return res

    # ---- starts so far (past days + players who already played today)
    starts = {my_id: 0, opp_id: 0}
    today_box = None
    for d in days:
        if d > TODAY:
            break
        b = box(sp_of(d))
        if d == TODAY:
            today_box = b
        for tid in (my_id, opp_id):
            starts[tid] += sum(1 for e in b[tid] if e["slot"] is not None and e["slot"] <= 11 and e["played"])
    # ---- expected points still to come
    plan_team = {t["id"]: t for t in W.get("teams", [])} if plan_match else {}
    plan_roster = {tid: {p["id"]: p for p in plan_team[tid]["roster"]} for tid in plan_team}
    nba_today = set((W.get("matchup", {}).get("nba_games", {}) or {}).get(TODAY.isoformat(), [])) if plan_match else set()
    rem_today = {my_id: 0.0, opp_id: 0.0}
    left_today = {my_id: [], opp_id: []}
    full_day = {my_id: 0.0, opp_id: 0.0}
    for tid in (my_id, opp_id):
        for e in (today_box or {}).get(tid, []):
            if e["slot"] is None or e["slot"] > 11:
                continue
            pr = plan_roster.get(tid, {}).get(e["id"], {})
            lvl = pr.get("level", 22.0)
            boost = (pr.get("boost") or {}).get(TODAY.isoformat(), 0.0)
            plays_today = (not nba_today) or (pr.get("team") in nba_today) or True
            p_play = 0.0 if e["status"] in ("OUT", "INJURY_RESERVE", "SUSPENSION") else (0.94 if plays_today else 0.0)
            full_day[tid] += (lvl + boost) * p_play
            if not e["played"] and plays_today and p_play > 0:
                ef = (lvl + boost) * p_play
                rem_today[tid] += ef
                left_today[tid].append({"name": e["name"], "ef": round(ef, 1)})
    fut = {my_id: 0.0, opp_id: 0.0}
    fut_days = 0
    for d in days:
        if d <= TODAY:
            continue
        got = False
        for tid in (my_id, opp_id):
            t = plan_team.get(tid)
            dd = next((x for x in (t or {}).get("days", []) if x["date"] == d.isoformat()), None)
            if dd:
                fut[tid] += dd["pts"]
                got = got or dd["pts"] > 0
            elif not plan_match:                             # crude fallback: this team's average points per day so far
                played_days = max(1, len(by_sp[tid]))
                fut[tid] += pts[tid] / played_days
                got = True
        fut_days += 1 if got else 0
    frac_today = 0.0
    if full_day[my_id] + full_day[opp_id] > 0:
        frac_today = min(1.0, (rem_today[my_id] + rem_today[opp_id]) / (full_day[my_id] + full_day[opp_id]))
    elif TODAY in days:
        frac_today = 0.5
    rem_days = frac_today + fut_days
    exp_rem = {tid: rem_today[tid] + fut[tid] for tid in (my_id, opp_id)}
    margin = pts[my_id] - pts[opp_id]
    mu = margin + BETA * (exp_rem[my_id] - exp_rem[opp_id])
    wp = Phi(mu / sigma(rem_days)) if rem_days > 0.02 else (1.0 if margin > 0 else 0.0 if margin < 0 else 0.5)
    # ---- history (kept on the live-data branch between runs)
    hist = []
    try:
        prev = requests.get(RAW_URL + f"?t={int(NOW.timestamp())}", timeout=20).json()
        if prev.get("matchup") == mp and prev.get("season") == YEAR:
            hist = prev.get("history", [])
    except Exception:
        pass
    hist.append({"t": datetime.now(timezone.utc).isoformat(timespec="minutes"), "wp": round(wp, 3), "margin": round(margin, 1), "me": round(pts[my_id], 1), "opp": round(pts[opp_id], 1)})
    hist = hist[-160:]
    my_plan = plan_team.get(my_id, {})
    adds_used = my_plan.get("adds_used")
    best_add = next((a for a in (my_plan.get("sequence") or [])[:1]), None)
    advice = []
    locked = starts[my_id] >= cap
    if locked:
        advice.append("You have reached the games cap: nothing else counts this week.")
    elif starts[my_id] >= cap - 4:
        advice.append(f"You are {cap - starts[my_id]:.0f} starts from the cap; only your best remaining games are worth starting.")
    if wp >= 0.9:
        advice.append("The week is nearly won: avoid needless risk, and do not spend adds you do not need.")
    elif wp <= 0.1 and rem_days > 0.5:
        need = -mu
        advice.append(f"A long shot: you need about {need:.0f} points more than expected. Only moves that add expected points matter; variance choices barely change the odds.")
    elif rem_days > 0.5:
        advice.append(f"Expected final margin {mu:+.0f}. Every extra started game is worth about 36 points, so fill every open slot and use your remaining adds well.")
    if best_add and rem_days > 1 and (adds_used is None or adds_used < W["matchup"]["adds_limit"]):
        advice.append(f"Best remaining add: {best_add['add']['name']} (+{best_add['gain']:.0f}).")
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": YEAR, "matchup": mp, "date": TODAY.isoformat(), "plan_used": bool(plan_match),
           "me": {"abbrev": teams[my_id], "name": names[my_id], "pts": round(pts[my_id], 1), "exp_rem": round(exp_rem[my_id], 1), "exp_final": round(pts[my_id] + exp_rem[my_id], 1), "starts": starts[my_id], "adds_used": adds_used, "left_today": sorted(left_today[my_id], key=lambda x: -x["ef"])[:8]},
           "opp": {"abbrev": teams[opp_id], "name": names[opp_id], "pts": round(pts[opp_id], 1), "exp_rem": round(exp_rem[opp_id], 1), "exp_final": round(pts[opp_id] + exp_rem[opp_id], 1), "starts": starts[opp_id], "left_today": sorted(left_today[opp_id], key=lambda x: -x["ef"])[:8]},
           "cap": cap, "days_left": round(rem_days, 2), "win_prob": round(wp, 3), "expected_margin": round(mu, 1), "advice": advice, "history": hist}
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=1)[:2500])
    if DRY:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
