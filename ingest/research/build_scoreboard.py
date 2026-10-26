"""League scoreboard data for the League > Matchups page: every matchup of every week played so far, with each team's daily totals and each player's
points that counted (started, before the team hit the games cap), points left on the bench, and starts. One ESPN call per day (all matchups at once);
days before yesterday are frozen in ingest/scoreboard_days.json and not pulled again. Writes dashboard/scoreboard.json (index: teams, every matchup's
totals) and dashboard/scoreboard/w<N>.json (one week's player detail, loaded by the page only when that week is opened).
Future weeks' pairings and projections come from playoff_odds.json (schedule) and schedule_plan.json; the current week's plan from week_plan.json.
Run from ingest/:  .venv/Scripts/python research/build_scoreboard.py        (refresh-league and refresh-gameday workflows)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League
from season import current_season_id

sys.stdout.reconfigure(encoding="utf-8")
DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"
OUT = DASH / "scoreboard.json"
WK_DIR = DASH / "scoreboard"
CACHE = Path(__file__).resolve().parent.parent / "scoreboard_days.json"
YEAR = current_season_id()
BENCH, IR = 12, 13
PRO = {}


def day_pull(lg, mp, sp):
    data = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mScoreboard"], "scoringPeriodId": sp},
                                      headers={"x-fantasy-filter": json.dumps({"schedule": {"filterMatchupPeriodIds": {"value": [mp]}}})})
    out = {}
    for m in data["schedule"]:
        if m.get("matchupPeriodId") != mp or "away" not in m:
            continue
        for side in ("home", "away"):
            t = m[side]
            ents = (t.get("rosterForCurrentScoringPeriod") or {}).get("entries") or []
            pl = []
            for e in ents:
                p = e["playerPoolEntry"]["player"]
                pts = float(e["playerPoolEntry"].get("appliedStatTotal") or 0.0)
                played = any(s.get("scoringPeriodId") == sp and s.get("statSourceId") == 0 and (s.get("stats") or {}).get("42") for s in p.get("stats", []))
                PRO[p["id"]] = (p["fullName"], p.get("proTeamId"))
                pl.append([p["id"], e["lineupSlotId"], round(pts, 1), int(played)])
            out[t["teamId"]] = pl
    return out


def main():
    lg = League(league_id=config.LEAGUE_ID, year=YEAR, espn_s2=config.ESPN_S2, swid=config.SWID)
    cur_sp, cur_mp = lg.current_week, lg.currentMatchupPeriod
    old = json.load(open(CACHE, encoding="utf-8")) if CACHE.exists() else {}
    if old.get("season") != YEAR:
        old = {}
    frozen = {int(k): {int(t): v for t, v in d.items()} for k, d in (old.get("days") or {}).items()}          # scoring period -> {team: [[pid, slot, pts, played]]}
    names = {int(k): v for k, v in (old.get("players") or {}).items()}
    raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mTeam"]})
    teams = {t["id"]: {"abbrev": t.get("abbrev"), "name": (t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}").strip()} for t in raw["teams"]}
    days = dict(frozen)
    n_pull = 0
    for mp, sps in sorted(lg.matchup_ids.items()):
        if mp > cur_mp:
            break
        for sp in map(int, sps):
            if sp > cur_sp or (sp in frozen and sp < cur_sp - 1):           # re-pull yesterday and today (stat corrections, games in progress)
                continue
            days[sp] = day_pull(lg, mp, sp)
            n_pull += 1
    for pid, (nm, pro) in PRO.items():
        names[pid] = [nm, pro]
    weeks = []
    for m in raw["schedule"]:
        if "away" not in m or m["matchupPeriodId"] > cur_mp:
            continue
        mp = m["matchupPeriodId"]
        sps = [int(x) for x in lg.matchup_ids.get(mp, [])]
        g = {"week": mp, "days": sps, "winner": m.get("winner"), "teams": []}
        for side in ("away", "home"):
            t = m[side]
            tid = t["teamId"]
            cap = ((t.get("cumulativeScore") or {}).get("statBySlot") or {}).get("15") or {}
            lock = cap.get("exceededOnScoringPeriod") if cap.get("limitExceeded") else None   # the day he went over: later days count nothing
            per = {}
            for i, sp in enumerate(sps):
                for pid, slot, pts, played in (days.get(sp) or {}).get(tid, []):
                    r = per.setdefault(pid, {"id": pid, "pts": 0.0, "bench": 0.0, "starts": 0, "d": [None] * len(sps)})
                    active = slot not in (BENCH, IR)
                    counts = active and played and (lock is None or sp <= lock)
                    if counts:
                        r["pts"] += pts
                        r["starts"] += 1
                    elif played and pts:
                        r["bench"] += pts
                    r["d"][i] = [round(pts, 1), 1 if counts else (0 if played else -1), slot]
            pl = sorted(per.values(), key=lambda r: (-r["pts"], -r["bench"]))
            for r in pl:
                r["pts"], r["bench"] = round(r["pts"], 1), round(r["bench"], 1)
            g["teams"].append({"id": tid, "side": side, "total": float(t.get("totalPoints") or 0.0), "live": float(t.get("totalPointsLive") or t.get("totalPoints") or 0.0),
                               "by_day": [float((t.get("pointsByScoringPeriod") or {}).get(str(sp), 0.0)) for sp in sps],
                               "starts": int(cap.get("value") or 0), "lock_day": lock, "players": pl})
        weeks.append(g)
    gen = datetime.now(timezone.utc).isoformat(timespec="seconds")
    CACHE.write_text(json.dumps({"season": YEAR, "players": {str(k): v for k, v in names.items()},
                                 "days": {str(k): {str(t): v for t, v in d.items()} for k, d in days.items()}}, separators=(",", ":")), encoding="utf-8")
    WK_DIR.mkdir(exist_ok=True)
    for w in sorted({g["week"] for g in weeks}):
        gs = [g for g in weeks if g["week"] == w]
        pids = {p["id"] for g in gs for t in g["teams"] for p in t["players"]}
        (WK_DIR / f"w{w}.json").write_text(json.dumps({"generated": gen, "week": w, "players": {str(k): names[k] for k in pids if k in names}, "matchups": gs},
                                                      separators=(",", ":")), encoding="utf-8")
    index = [{"week": g["week"], "winner": g["winner"], "teams": [{k: t[k] for k in ("id", "side", "total", "live", "starts", "lock_day")} |
                                                                 {"top": [[p["id"], p["pts"]] for p in t["players"][:1]]} for t in g["teams"]]} for g in weeks]
    OUT.write_text(json.dumps({"generated": gen, "season": YEAR, "current_week": cur_mp, "current_day": cur_sp, "teams": {str(k): v for k, v in teams.items()},
                               "players": {str(k): names[k] for k in {p for g in index for t in g["teams"] for p, _ in t["top"]} if k in names},
                               "matchups": index}, separators=(",", ":")), encoding="utf-8")
    print(f"scoreboard: {len(weeks)} matchups through week {cur_mp}, {n_pull} days pulled, {len(days)} days cached")


if __name__ == "__main__":
    main()
