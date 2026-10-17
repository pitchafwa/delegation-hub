"""How much of the games cap each team actually uses, matchup by matchup (dashboard/team_usage.json), for the League > Teams page.

A "start" = a player in an active lineup slot (ids 0-11) who played that day. Per team and matchup: started games, the cap for that matchup (40 for 7 days, scaled for other lengths),
whether the team arrived at the last day already at the cap (the last day was locked), and starts on the final day.

History (2023-24, 2024-25, 2025-26, 7-day matchups) comes from the local league_days files (pull_league_days.py YEAR). The CURRENT season is added from ESPN for every matchup that has finished and is not
in the file yet (7 requests per matchup), so the daily refresh keeps it current. Rows already in the JSON are kept, so the cloud run needs no local files.

  uv run python research/build_team_usage.py        (from ingest/)
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
D = Path(__file__).resolve().parent / "data"
OUT = HUB / "team_usage.json"
STARTING = set(range(0, 12))
HIST_YEARS = [2024, 2025, 2026]


def row(year, mp, team, per_day, cap):
    total = sum(per_day)
    pre_last = sum(per_day[:-1])
    return {"y": year, "mp": mp, "team": team, "total": total, "n": len(per_day), "cap": cap, "locked_last": pre_last >= cap, "last_day": per_day[-1]}


def from_days(year, days):
    """rows from a league_days dict: {scoring period: [{mp, team, e: [[pid, name, slot, gp, ...]]}]}"""
    by = {}
    for d, recs in days.items():
        for r in recs:
            st = sum(1 for e in r["e"] if e[2] in STARTING and e[3])
            by.setdefault((r["mp"], r["team"]), []).append((int(d), st))
    out = []
    for (mp, team), v in by.items():
        v.sort()
        per = [s for _, s in v]
        if len(per) == 7:
            out.append(row(year, mp, team, per, 40))
    return out


def main():
    data = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {"rows": []}
    rows = {(r["y"], r["mp"], r["team"]): r for r in data["rows"]}
    for y in HIST_YEARS:
        f = D / f"league_days_{y}.json"
        if f.exists():
            for r in from_days(y, json.load(open(f, encoding="utf-8"))):
                rows.setdefault((r["y"], r["mp"], r["team"]), r)
    # current season, finished matchups only
    try:
        import config
        import season_calendar as SCAL
        from espn_api.basketball import League
        cal = SCAL.load()
        opener = SCAL.opener(cal)
        done = [p for p in cal["periods"] if date.fromisoformat(p["end"]) < date.today() and not any((config.SEASON, p["id"], t) in rows for t in range(1, 30))]
        if done:
            lg = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)
        for p in done:
            d0 = (date.fromisoformat(p["start"]) - opener).days + 1
            days = {}
            for k in range(p["days"]):
                sp = d0 + k
                raw = lg.espn_request.league_get(params={"view": "mBoxscore", "scoringPeriodId": sp})
                recs = []
                for m in raw.get("schedule", []):
                    if m.get("matchupPeriodId") != p["id"]:
                        continue
                    for side in ("home", "away"):
                        t = m.get(side, {})
                        ents = []
                        for e in (t.get("rosterForCurrentScoringPeriod") or {}).get("entries", []):
                            pl = e.get("playerPoolEntry", {}).get("player", {})
                            st = [s for s in pl.get("stats", []) if s.get("scoringPeriodId") == sp and s.get("statSourceId") == 0]
                            gp = bool(st and st[0].get("stats", {}).get("42", 0))
                            ents.append([pl.get("id"), pl.get("fullName"), e.get("lineupSlotId"), int(gp)])
                        recs.append({"mp": p["id"], "team": t.get("teamId"), "e": ents})
                days[str(sp)] = recs
                time.sleep(0.3)
            by = {}
            for sp, recs in days.items():
                for r in recs:
                    by.setdefault(r["team"], []).append((int(sp), sum(1 for e in r["e"] if e[2] in STARTING and e[3])))
            for team, v in by.items():
                v.sort()
                rows[(config.SEASON, p["id"], team)] = row(config.SEASON, p["id"], team, [s for _, s in v], p["cap"])
            print("added matchup", p["id"], "for", len(by), "teams")
    except Exception as ex:
        print("live additions skipped:", repr(ex)[:120])
    out = sorted(rows.values(), key=lambda r: (r["y"], r["mp"], r["team"]))
    OUT.write_text(json.dumps({"generated": date.today().isoformat(), "cap_note": "cap = 40 for a 7-day matchup, scaled for other lengths", "rows": out}, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT, len(out), "team-matchups")


if __name__ == "__main__":
    main()
