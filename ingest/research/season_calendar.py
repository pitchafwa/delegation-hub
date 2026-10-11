"""The season's dates, derived instead of typed in once a year.

What used to be hard-coded in about eight scripts (opener date, the 6/7/14-day matchup lengths, last fantasy week, playoff weeks, season-end year) now comes from here.

Rules (they reproduce the calendar confirmed against ESPN's real scoreboard on 2026-09-27: matchup 1 = Oct 20-25, matchups 2-17 = Mon-Sun weeks, matchup 18 = the 14-day
All-Star matchup Feb 15-28, matchups 19-22 = Mon-Sun weeks with the playoffs = 20-22):
  * the OPENER is the first game date in dashboard/nba_schedule.json (the current season's NBA schedule);
  * matchup 1 runs from the opener through the first Sunday;
  * after that, Monday-Sunday weeks. ESPN's fantasy season has M matchup periods in total (regular-season count + playoff rounds); M-1 normal weeks after the opener plus ONE extra
    week that is merged into the lightest regular-season week (the All-Star break: fewest NBA games), giving that matchup 14 days. Total scoring periods = opener days + 7 x M.
  * playoff weeks = the last `rounds` periods; rounds come from ESPN's playoffTeamCount (6 teams = 3 rounds);
  * cap = 40 starts per 7 days, scaled by length (6 days = 34.3, 14 days = 80).
ESPN supplies the counts (matchupPeriodCount, playoffTeamCount, finalScoringPeriod) when a league object is available (refresh(lg), run early in the workflows); the result is
written to dashboard/season_calendar.json so scripts without ESPN access (schedule plan, draft data, alerts) read the same answer. If that file and ESPN are both unavailable the
defaults below are used (19 regular matchups + 3 playoff rounds, as this league is set up).
"""
import json
import math
from datetime import date, timedelta
from pathlib import Path

HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
CAL_PATH = HUB / "season_calendar.json"
CAP_PER_7 = 40.0
DEFAULT_REGULAR, DEFAULT_ROUNDS = 19, 3


def _schedule_days():
    sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))["games"]
    return {date.fromisoformat(d): len(g) for d, g in sched.items()}


def rounds_for(teams):
    return 1 if teams <= 2 else (2 if teams <= 4 else 3)


def compute(regular=DEFAULT_REGULAR, rounds=DEFAULT_ROUNDS, games_by_day=None):
    games_by_day = games_by_day or _schedule_days()
    opener = min(games_by_day)
    m_total = regular + rounds
    first_end = opener + timedelta(days=(6 - opener.weekday()) % 7)        # through the first Sunday
    weeks, d = [], first_end + timedelta(days=1)
    for _ in range(m_total):                                                # M Monday-Sunday weeks after the opener; one gets merged, leaving M-1 + the opener = M periods
        weeks.append((d, d + timedelta(days=6)))
        d += timedelta(days=7)
    eligible = range(0, m_total - rounds - 1)                               # a regular-season week that has a following week to merge into

    def n_games(i):
        a, b = weeks[i]
        return sum(games_by_day.get(a + timedelta(days=k), 0) for k in range(7))
    # the All-Star break = the longest run of days with no NBA games; the Monday-Sunday week it starts in is merged with the next week (ESPN's 14-day matchup).
    # (Not simply the lightest week: the NBA Cup knockout week looks sparse too while its games are still TBD.) Fallback with no clear break: the lightest eligible week.
    best_run, run_start, run_len, cur_len, cur_start = None, None, 0, 0, None
    d0, d1 = weeks[0][0], weeks[max(eligible)][1]
    day = d0
    while day <= d1:
        if games_by_day.get(day, 0) == 0:
            if cur_len == 0:
                cur_start = day
            cur_len += 1
            if cur_len > run_len:
                run_len, run_start = cur_len, cur_start
        else:
            cur_len = 0
        day += timedelta(days=1)
    merge = None
    if run_len >= 3 and run_start is not None:
        merge = next((i for i in eligible if weeks[i][0] <= run_start <= weeks[i][1]), None)
    if merge is None:
        merge = min(eligible, key=lambda i: (n_games(i), i))
    spans = [(opener, first_end)]
    i = 0
    while i < len(weeks):
        a, b = weeks[i]
        if i == merge:
            b = weeks[i + 1][1]
            i += 1
        spans.append((a, b))
        i += 1
    periods = []
    for k, (a, b) in enumerate(spans, start=1):
        n = (b - a).days + 1
        periods.append({"id": k, "start": a.isoformat(), "end": b.isoformat(), "days": n, "cap": round(CAP_PER_7 * n / 7.0, 1), "playoff": k > m_total - rounds})
    return {"opener": opener.isoformat(), "periods": periods, "last_week": len(periods), "regular_weeks": regular, "playoff_rounds": rounds,
            "playoff_weeks": [p["id"] for p in periods if p["playoff"]], "final_scoring_period": sum(p["days"] for p in periods)}


def load():
    """the saved calendar (written by refresh) or, failing that, one computed from the NBA schedule alone"""
    if CAL_PATH.exists():
        try:
            return json.load(open(CAL_PATH, encoding="utf-8"))
        except Exception:
            pass
    return compute()


def verify_against_espn(lg, cal):
    """compare the first and last scoring period of every REGULAR-SEASON matchup with the matchup number ESPN's own scoreboard reports for that day (ESPN publishes it for future
    days too). Playoff periods are skipped: ESPN has no playoff matchups until the bracket exists. Returns (all matched, list of mismatches)."""
    sp, bad = 1, []
    for p in cal["periods"]:
        if p["playoff"]:
            break
        for probe in (sp, sp + p["days"] - 1):
            raw = lg.espn_request.league_get(params={"view": "mBoxscore", "scoringPeriodId": probe})
            # the schedule lists every matchup of the season; the one that is ON that day is the one carrying rosters for that scoring period
            ids = {m.get("matchupPeriodId") for m in raw.get("schedule", []) if m.get("matchupPeriodId") is not None
                   and ((m.get("home") or {}).get("rosterForCurrentScoringPeriod") or (m.get("away") or {}).get("rosterForCurrentScoringPeriod"))}
            if ids != {p["id"]}:
                bad.append(f"matchup {p['id']}: scoring period {probe} is {sorted(ids) or 'unscheduled'} in ESPN")
        sp += p["days"]
    return (not bad), bad


def refresh(lg):
    """read ESPN's counts, compute, sanity-check against ESPN's own final scoring period, verify against ESPN's scoreboard (only when the calendar changed or was never verified:
    it is ~40 requests), and save. Returns the calendar dict (cal['verified'] True when every regular-season matchup boundary matches ESPN)."""
    regular, rounds, final_sp = DEFAULT_REGULAR, DEFAULT_ROUNDS, None
    try:
        ss = lg.espn_request.league_get(params={"view": ["mSettings"]})["settings"]["scheduleSettings"]
        regular = int(ss.get("matchupPeriodCount") or regular)
        rounds = rounds_for(int(ss.get("playoffTeamCount") or 6))
        final_sp = (lg.espn_request.league_get(params={"view": ["mStatus"]}).get("status") or {}).get("finalScoringPeriod")
    except Exception as ex:
        print("season_calendar: could not read ESPN's schedule settings, using defaults:", repr(ex)[:80])
    cal = compute(regular, rounds)
    if final_sp and int(final_sp) != cal["final_scoring_period"]:
        print(f"season_calendar WARNING: computed {cal['final_scoring_period']} scoring periods but ESPN says {final_sp}; check the matchup lengths")
        cal["warning"] = f"ESPN final scoring period {final_sp} != computed {cal['final_scoring_period']}"
    old = None
    try:
        old = json.load(open(CAL_PATH, encoding="utf-8")) if CAL_PATH.exists() else None
    except Exception:
        old = None
    if old and old.get("periods") == cal["periods"] and old.get("verified") and not cal.get("warning"):
        return old
    try:
        ok, bad = verify_against_espn(lg, cal)
        cal["verified"], cal["mismatches"] = bool(ok and not cal.get("warning")), bad[:6]
        print("season_calendar: verified against ESPN's scoreboard" if cal["verified"] else f"season_calendar: NOT verified: {bad[:3]}")
    except Exception as ex:
        cal["verified"], cal["mismatches"] = False, [f"verification failed: {repr(ex)[:80]}"]
    CAL_PATH.write_text(json.dumps(cal, separators=(",", ":")), encoding="utf-8")
    return cal


# ---- helpers the scripts use
def opener(cal=None):
    return date.fromisoformat((cal or load())["opener"])


def bounds(cal=None):
    """[(id, start date, end date)] like build_week_plan's old BOUNDS"""
    return [(p["id"], date.fromisoformat(p["start"]), date.fromisoformat(p["end"])) for p in (cal or load())["periods"]]


def verified(cal=None):
    return bool((cal or load()).get("verified"))


def first_tip(cal=None):
    """(date, 'HH:MM' UTC) of the earliest tip-off on the opener, from the NBA schedule file"""
    from datetime import datetime
    o = opener(cal)
    sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))["games"].get(o.isoformat(), [])
    tips = sorted(g[2] for g in sched if len(g) > 2 and g[2])
    return o, (tips[0] if tips else None)


def lengths(cal=None):
    return [p["days"] for p in (cal or load())["periods"]]


if __name__ == "__main__":
    c = compute()
    print("opener", c["opener"], "| periods", len(c["periods"]), "| final scoring period", c["final_scoring_period"], "| playoffs", c["playoff_weeks"])
    for p in c["periods"]:
        print(f"  {p['id']:2d}  {p['start']} to {p['end']}  {p['days']:2d} days  cap {p['cap']}" + ("  PLAYOFF" if p["playoff"] else ""))
