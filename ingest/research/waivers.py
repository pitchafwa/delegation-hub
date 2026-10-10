"""Who is a true free agent and who is on waivers (confirmed 2026-10-10, see OPENING_NIGHT_CHECKLIST 5b and RESEARCH_start_sit_waivers.md).

League rules (standard waivers, 24-hour period, no FAAB):
  * a player nobody has dropped recently is a FREE AGENT: add him instantly, any hour, up to the add limit;
  * a player a team DROPS goes on WAIVERS: he cannot be added directly; a claim is processed by ESPN's nightly run at about 3am ET, never sooner than 24 hours after the drop,
    in waiver-priority order (a team ahead of you can win him). Players nobody claims become free agents after that run.
ESPN's free_agents() call mixes the two groups, so everything that suggests an add (plan, suggested moves, scratch swaps, alerts) uses this module to tell them apart.

  waiver_map(lg)      {espn player id: {"drop_ms", "clear" (ET datetime of the run that releases/awards him), "approx"}} for everyone on waivers now
  available_pool(lg)  the free-agent AND waiver pool with each player's status, most-owned first
  my_priority(lg, id) (waiver priority number, teams) -- 1 = first in line; None if ESPN does not say
  tag(w) / when(w)    short text for the UI and alerts
"""
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
RUN_HOUR = 3            # ESPN's nightly waiver run (40 of 43 claims last season processed in the 3am hour ET, the rest in the 4am hour)
WAIVER_HOURS = 24       # the league's waiver period


def _players(lg, statuses, limit):
    flt = {"players": {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}, "filterStatus": {"value": statuses}}}
    return lg.espn_request.league_get(params={"view": "kona_player_info"}, headers={"x-fantasy-filter": json.dumps(flt)})["players"]


def last_drop_times(lg, periods_back=2):
    """player id -> time (ms) of the most recent executed DROP in the last few scoring periods"""
    cur = lg.scoringPeriodId
    out = {}
    for sp in range(max(1, cur - periods_back), cur + 1):
        try:
            raw = lg.espn_request.league_get(params={"view": "mTransactions2", "scoringPeriodId": sp})
        except Exception:
            continue
        for t in raw.get("transactions", []):
            if t.get("status") != "EXECUTED":
                continue
            ts = t.get("processDate") or t.get("proposedDate")
            for it in t.get("items", []):
                if it.get("type") == "DROP" and ts and ts > out.get(it.get("playerId"), 0):
                    out[it["playerId"]] = ts
    return out


def clear_time(drop_ms, now=None):
    """the first nightly run (3am ET) at or after drop + 24h. With no drop time on record the best guess is the next run (flagged approximate by the caller)."""
    now = now or datetime.now(ET)
    t = datetime.fromtimestamp(drop_ms / 1000, ET) + timedelta(hours=WAIVER_HOURS) if drop_ms else now
    run = t.replace(hour=RUN_HOUR, minute=0, second=0, microsecond=0)
    if run < t:
        run += timedelta(days=1)
    return run


def waiver_map(lg, limit=400):
    try:
        on = _players(lg, ["WAIVERS"], limit)
    except Exception as ex:
        print("could not read the waiver list:", repr(ex)[:90], "- treating everyone as a free agent")
        return {}
    import os
    fake = [int(x) for x in os.environ.get("FAKE_WAIVER_IDS", "").split(",") if x.strip()]      # TEST HOOK: pretend these player ids were just dropped
    if fake:
        now_ms = int(datetime.now(ET).timestamp() * 1000)
        return {i: {"drop_ms": now_ms, "clear": clear_time(now_ms), "approx": False} for i in fake}
    if not on:
        return {}
    drops = last_drop_times(lg)
    out = {}
    for e in on:
        ms = drops.get(e["id"])
        out[e["id"]] = {"drop_ms": ms, "clear": clear_time(ms), "approx": ms is None}
    return out


def available_pool(lg, limit=300):
    """[{id, name, team_id (pro), injury, status 'FREEAGENT'|'WAIVERS', pct_owned}] most-owned first"""
    import os
    fake = {int(x) for x in os.environ.get("FAKE_WAIVER_IDS", "").split(",") if x.strip()}      # TEST HOOK (see waiver_map)
    res = []
    for e in _players(lg, ["FREEAGENT", "WAIVERS"], limit):
        if e["id"] in fake:
            e = dict(e, status="WAIVERS")
        p = e["player"]
        res.append({"id": e["id"], "name": p["fullName"], "pro_team_id": p.get("proTeamId"), "injury": p.get("injuryStatus"), "status": e["status"],
                    "pct_owned": (p.get("ownership") or {}).get("percentOwned")})
    return res


def my_priority(lg, team_id):
    """(waiver priority, number of teams): 1 = first claim. From ESPN's raw team data (the espn_api library does not parse it)."""
    try:
        teams = lg.espn_request.league_get(params={"view": ["mTeam"]})["teams"]
        mine = next(t for t in teams if t["id"] == team_id)
        r = mine.get("waiverRank")
        return (int(r) if r is not None else None), len(teams)
    except Exception:
        return None, 12


def when(w):
    """'Sat 3am ET' style text for the run that releases him"""
    c = w["clear"] if isinstance(w["clear"], datetime) else datetime.fromisoformat(w["clear"])
    return c.strftime("%a ") + c.strftime("%I").lstrip("0") + c.strftime("%p").lower() + " ET"


def export(w):
    """JSON-safe version stored in week_plan.json"""
    if not w:
        return None
    return {"clear": w["clear"].isoformat(timespec="minutes"), "date": w["clear"].date().isoformat(), "when": when(w), "approx": bool(w.get("approx"))}
