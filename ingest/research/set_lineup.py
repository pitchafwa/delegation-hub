"""Set-the-lineup tool for the rest of the current matchup (SKELETON, 2026-10-08): compares the weekly plan's starters for EVERY remaining day against what ESPN
currently has for that day, and writes the list of moves to dashboard/lineup_preview.json (shown on the This week tab).

  python research/set_lineup.py              build the preview (read-only; never writes to ESPN)
  python research/set_lineup.py --demo      same, but pretends nothing is set in ESPN yet (writes lineup_preview_example.json so the page can show a full example)
  python research/set_lineup.py --apply      NOT BUILT YET: needs the lineup-write request captured from Tommy's browser (see APPLY below)

What the plan means per day: `week_plan.json` teams[me].days[i].start = the starters (id, slot) the cap-aware plan wants, for the plan with NO adds. Everyone else on the active
roster should be on the bench; IR players stay put. A day the plan marks `locked` (cap already reached) is left alone. Today's players whose game has already tipped are
locked in ESPN and are never moved.

APPLY (to build once the request is captured): for each day with moves, POST the lineup change to ESPN for that scoring period with Tommy's cookies, one day at a time, after
saving a snapshot of the current lineups (ingest/research/data/lineup_snapshots/) so everything can be undone. Rules it must keep: only moves between slots (never an add or a
drop), never a locked player, a dry-run by default, a hash of the preview so it applies exactly what was shown, and a ledger line per change.
"""
import hashlib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from team_abbr import canon

ET = ZoneInfo("America/New_York")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
SLOT_NAME = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF", 8: "G/F", 9: "PF/C", 10: "F/C", 11: "UT", 12: "BE", 13: "IR"}


def load(name):
    return json.load(open(HUB / name, encoding="utf-8"))


def scoring_period(d, first_day):
    """ESPN scoring period 1 is the opener (2026-10-20); one period per day"""
    return (d - first_day).days + 1


def today_locked_teams(today):
    """NBA teams whose game today has already tipped off (those players are locked in ESPN)"""
    sched = load("nba_schedule.json")
    now = datetime.now(ET)
    out = set()
    for a, h, t in sched.get("games", {}).get(today.isoformat(), []):
        try:
            hh, mm = int(t[:2]), int(t[3:5])
        except (TypeError, ValueError):
            continue
        dt = datetime(today.year, today.month, today.day, hh, mm, tzinfo=ZoneInfo("UTC"))
        if hh < 10:
            dt += timedelta(days=1)
        if dt.astimezone(ET) <= now:
            out |= {canon(a), canon(h)}
    return out


SLOT_CAP = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UT": 3}


def minimal_moves(cur, elig, want, plays, is_today, locked_teams, team_of):
    """fewest moves that put the plan's starters in active slots. cur: pid -> (name, slot) for the day, elig: pid -> set of slot names he may fill,
    want: pid -> the slot the plan chose (a preference only: a starter already in ANY active slot is fine, since points do not depend on which slot).
    A player the plan does not start is benched only if he plays that day (the plan saves his game for the cap) or he holds a slot a starter needs; players with no game
    that day stay put. Locked players (game already tipped today) are never moved. Returns (moves, problems)."""
    locked = {pid for pid in cur if is_today and team_of.get(pid) in locked_teams}
    final = {pid: sl for pid, (nm, sl) in cur.items()}
    for pid, (nm, sl) in cur.items():                          # non-starters who play that day: bench them
        if pid not in want and sl in SLOT_CAP and pid in plays and pid not in locked:
            final[pid] = "BE"
    problems = []
    for pid, plan_slot in want.items():
        if pid not in cur or final[pid] in SLOT_CAP or pid in locked:
            continue                                           # missing from roster (reported by the caller), already in an active slot, or locked
        options = [plan_slot] + [s for s in SLOT_CAP if s != plan_slot and s in elig.get(pid, ())]
        if plan_slot not in elig.get(pid, set(SLOT_CAP)):
            options = [s for s in SLOT_CAP if s in elig.get(pid, ())]
        placed = False
        for sname in options:
            occ = [q for q, sl in final.items() if sl == sname]
            if len(occ) < SLOT_CAP[sname]:
                final[pid] = sname
                placed = True
                break
            evict = [q for q in occ if q not in want and q not in locked]      # a player with no game, or one the plan benches
            if evict:
                final[evict[0]] = "BE"
                final[pid] = sname
                placed = True
                break
        if not placed:
            problems.append(f"{cur[pid][0]}: no free or evictable slot he is eligible for (needs a swap chain)")
    moves = [{"id": pid, "name": cur[pid][0], "from": cur[pid][1], "to": final[pid]} for pid in cur if final[pid] != cur[pid][1]]
    return moves, problems


def build_preview(from_scratch=False):
    import config
    from espn_api.basketball import League
    W = load("week_plan.json")
    SP = load("schedule_plan.json")
    me = next(t for t in W["teams"] if t["abbrev"] == W["my_abbrev"])
    first_day = date.fromisoformat(SP["calendar"][0]["start"])
    lg = League(league_id=config.LEAGUE_ID, year=W["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    today = datetime.now(ET).date()
    locked_teams = today_locked_teams(today)
    team_of = {p["id"]: canon(p["team"]) for p in me["roster"]}
    days_out, total = [], 0
    for d in me["days"]:
        ds = d["date"]
        dd = date.fromisoformat(ds)
        sp = scoring_period(dd, first_day)
        raw = lg.espn_request.league_get(params={"view": ["mRoster"], "scoringPeriodId": sp})
        team = next(t for t in raw["teams"] if t["id"] == me["id"])
        cur, elig = {}, {}
        for e in team["roster"]["entries"]:
            slot = SLOT_NAME.get(e["lineupSlotId"], str(e["lineupSlotId"]))
            if from_scratch and slot not in ("IR",):
                slot = "BE"                                    # example mode: pretend nothing is set yet
            cur[e["playerId"]] = (e["playerPoolEntry"]["player"]["fullName"], slot)
            elig[e["playerId"]] = {SLOT_NAME.get(i) for i in e["playerPoolEntry"]["player"].get("eligibleSlots", [])} & set(SLOT_CAP)
        want = {x["id"]: x["slot"] for x in d["start"]}
        plays_today = {r["id"] for r in me["roster"] if ds in r.get("games", [])}
        moves, note = [], ""
        if d.get("locked"):
            note = "Games cap already reached by then: nothing counts, so the lineup is left alone."
        else:
            moves, probs = minimal_moves(cur, elig, want, plays_today, dd == today, locked_teams, team_of)
            if probs:
                note = "; ".join(probs)
            for pid in want:
                if pid not in cur:
                    moves.append({"id": pid, "name": next(x["name"] for x in d["start"] if x["id"] == pid), "from": "?", "to": want[pid], "problem": "not on your ESPN roster"})
        # order: bench moves first, then starters in slot order (the order an apply step would need to follow)
        moves.sort(key=lambda m: (m["to"] != "BE", m["to"]))
        total += len(moves)
        days_out.append({"date": ds, "sp": sp, "n_nba_games": d.get("nba_teams", 0) // 2, "plan": [{"id": x["id"], "name": x["name"], "slot": x["slot"], "ef": x["ef"]} for x in d["start"]],
                         "moves": moves, "note": note,
                         "current": [{"name": v[0], "slot": v[1]} for pid, v in cur.items() if v[1] not in ("BE", "IR")]})
    body = {"matchup": W["matchup"]["id"], "start": W["matchup"]["start"], "end": W["matchup"]["end"], "plan_generated": W["generated"], "days": days_out, "total_moves": total}
    body["hash"] = hashlib.md5(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]
    body["generated"] = datetime.now(timezone.utc).isoformat()
    body["apply_supported"] = False
    body["example"] = bool(from_scratch)
    body["plan_note"] = "Plan with no adds. Everyone not starting is benched; IR players stay on IR."
    return body


def apply(preview):
    raise SystemExit("--apply is not built yet: it needs the lineup-write request captured from a real ESPN lineup move (endpoint and body shape). "
                     "Nothing was changed in ESPN.")


if __name__ == "__main__":
    demo = "--demo" in sys.argv
    pv = build_preview(from_scratch=demo)
    (HUB / ("lineup_preview_example.json" if demo else "lineup_preview.json")).write_text(json.dumps(pv, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"matchup {pv['matchup']} ({pv['start']} to {pv['end']}): {pv['total_moves']} lineup moves over {len(pv['days'])} days (preview {pv['hash']})")
    for d in pv["days"]:
        print(f"  {d['date']} ({d['n_nba_games']} NBA games): {len(d['moves'])} moves" + (f"  [{d['note']}]" if d["note"] else ""))
        for m in d["moves"][:12]:
            print(f"      {m['name']:22s} {m['from']:>5s} -> {m['to']}" + (f"   !! {m['problem']}" if m.get("problem") else ""))
    if "--apply" in sys.argv:
        apply(pv)
