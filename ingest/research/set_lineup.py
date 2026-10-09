"""Set-the-lineup tool for the rest of the current matchup (SKELETON, 2026-10-08): compares the weekly plan's starters for EVERY remaining day against what ESPN
currently has for that day, and writes the list of moves to dashboard/lineup_preview.json (shown on the This week tab).

  python research/set_lineup.py              build the preview (read-only; never writes to ESPN)
  python research/set_lineup.py --demo      same, but pretends nothing is set in ESPN yet (writes lineup_preview_example.json so the page can show a full example)
  python research/set_lineup.py --mode apply --hash H --out DIR   apply exactly the preview with hash H (refuses if the preview changed). NOT SWITCHED ON YET
                                             (APPLY_READY=False): needs the lineup-write request captured from Tommy's browser (see APPLY below)
  --out DIR writes the preview/result files there (the set-lineup workflow publishes them to the lineup-data branch for the site)

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


APPLY_READY = True        # switched on 2026-10-09 after --selftest --write passed (swap + revert verified in ESPN with cookies alone)


SLOT_ID = {v: k for k, v in SLOT_NAME.items()}
WRITE_URL = "https://lm-api-writes.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/segments/0/leagues/{league}/transactions/"


def lineup_payload(team_id, sp, moves, current_sp):
    """the exact bodies ESPN's own lineup page sends (captured 2026-10-09 from real swaps): the CURRENT scoring period uses type ROSTER (with the member id = SWID), every
    LATER day uses type FUTURE_ROSTER (no member id). ESPN answers 409 'can only be executed in the current scoring period' if a future day is sent as ROSTER.
    One transaction per scoring period, one LINEUP item per player moved."""
    import config
    items = [{"playerId": m["id"], "type": "LINEUP", "fromLineupSlotId": SLOT_ID[m["from"]], "toLineupSlotId": SLOT_ID[m["to"]]} for m in moves]
    if sp < current_sp:
        raise RuntimeError("that day is already in the past")
    if sp == current_sp:
        return {"isLeagueManager": False, "teamId": team_id, "type": "ROSTER", "memberId": config.SWID, "scoringPeriodId": sp, "executionType": "EXECUTE", "items": items}
    return {"isLeagueManager": False, "teamId": team_id, "type": "FUTURE_ROSTER", "scoringPeriodId": sp, "executionType": "EXECUTE", "items": items}


def read_day(lg, team_id, sp):
    raw = lg.espn_request.league_get(params={"view": ["mRoster"], "scoringPeriodId": sp})
    team = next(t for t in raw["teams"] if t["id"] == team_id)
    return {e["playerId"]: SLOT_NAME.get(e["lineupSlotId"], str(e["lineupSlotId"])) for e in team["roster"]["entries"]}


def my_team_id():
    W = load("week_plan.json")
    return next(t for t in W["teams"] if t["abbrev"] == W["my_abbrev"])["id"]


def write_lineup(lg, sp, moves, team_id=None, dry=False):
    """Send one scoring period's lineup moves to ESPN, then re-read that day's lineup and confirm every moved player is where he was meant to be.
    Raises on any non-success response or any mismatch, so a failed day is never reported as applied."""
    import config
    import requests
    team_id = team_id or my_team_id()
    body = lineup_payload(team_id, sp, moves, lg.scoringPeriodId)
    if dry:
        print("DRY RUN, would POST:", json.dumps({**body, **({"memberId": "<SWID>"} if "memberId" in body else {})}))
        return
    url = WRITE_URL.format(season=load("week_plan.json")["season"], league=config.LEAGUE_ID)
    r = requests.post(url, json=body, cookies={"espn_s2": config.ESPN_S2, "SWID": config.SWID}, timeout=30,
                      headers={"Accept": "application/json", "X-Fantasy-Source": "kona", "Origin": "https://fantasy.espn.com", "Referer": "https://fantasy.espn.com/", "User-Agent": "Mozilla/5.0"})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"ESPN answered {r.status_code}: {r.text[:200]}")
    try:
        j = r.json()
        if isinstance(j, dict) and str(j.get("status", "")).upper() in ("FAILED", "ERROR", "INVALID"):
            raise RuntimeError(f"ESPN rejected the lineup: {json.dumps(j)[:200]}")
    except ValueError:
        pass
    now = read_day(lg, team_id, sp)
    bad = [m["name"] for m in moves if now.get(m["id"]) != m["to"]]
    if bad:
        raise RuntimeError(f"ESPN accepted the request but these players are not in the expected slot afterwards: {', '.join(bad)}")


def selftest(day_iso, do_write):
    """harmless end-to-end test of the write path: swap one player who has NO game that day (in an active slot) with a bench player who also has none, check ESPN shows it,
    swap back, check again. Without --write it only prints the payloads."""
    import config
    from espn_api.basketball import League
    W = load("week_plan.json")
    me = next(t for t in W["teams"] if t["abbrev"] == W["my_abbrev"])
    SP = load("schedule_plan.json")
    sp = scoring_period(date.fromisoformat(day_iso), date.fromisoformat(SP["calendar"][0]["start"]))
    lg = League(league_id=config.LEAGUE_ID, year=W["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    cur = read_day(lg, me["id"], sp)
    plays = {r["id"] for r in me["roster"] if day_iso in r.get("games", [])}
    name = {r["id"]: r["name"] for r in me["roster"]}
    act = [pid for pid, sl in cur.items() if sl in SLOT_CAP and pid not in plays]
    ben = [pid for pid, sl in cur.items() if sl == "BE" and pid not in plays]
    if not act or not ben:
        raise SystemExit("no suitable pair (an active-slot player and a bench player who both have no game that day)")
    A, B = act[0], ben[0]
    sA = cur[A]
    out = [{"id": A, "name": name.get(A, str(A)), "from": sA, "to": "BE"}, {"id": B, "name": name.get(B, str(B)), "from": "BE", "to": sA}]
    back = [{"id": A, "name": name.get(A, str(A)), "from": "BE", "to": sA}, {"id": B, "name": name.get(B, str(B)), "from": sA, "to": "BE"}]
    print(f"self-test on {day_iso} (scoring period {sp}): swap {out[0]['name']} ({sA}) with {out[1]['name']} (bench), neither plays that day")
    write_lineup(lg, sp, out, me["id"], dry=not do_write)
    if do_write:
        print("  swap applied and verified in ESPN")
    write_lineup(lg, sp, back, me["id"], dry=not do_write)
    if do_write:
        print("  swap reverted and verified in ESPN")
        final = read_day(lg, me["id"], sp)
        print("  lineup identical to the start:", final == cur)


def do_undo(out_dir):
    """reverse the most recent apply (kept in lineup_last_apply.json, carried between runs by the workflow): every moved player goes back to the slot he came from,
    day by day, each verified. Days whose games have already started will be refused by ESPN and reported."""
    f = out_dir / "lineup_last_apply.json"
    if not f.exists():
        return result_body(False, "Nothing to undo: no earlier apply was recorded.", "undo")
    import config
    from espn_api.basketball import League
    last = json.loads(f.read_text(encoding="utf-8"))
    lg = League(league_id=config.LEAGUE_ID, year=load("week_plan.json")["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    done = []
    for d in last["days"]:
        back = [{"id": m["id"], "name": m["name"], "from": m["to"], "to": m["from"]} for m in d["moves"]]
        try:
            write_lineup(lg, d["sp"], back, team_id=my_team_id())
            done.append(d["date"])
        except Exception as ex:
            return result_body(False, f"Undo stopped at {d['date']}: {ex}. Days already undone: {', '.join(done) or 'none'}.", "undo", None, done)
    f.rename(out_dir / "lineup_last_apply.undone.json")
    return result_body(True, f"Undone for {len(done)} day(s): {', '.join(done)}.", "undo", None, done)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else default


def result_body(ok, message, mode, pv=None, applied=None):
    return {"ok": bool(ok), "message": message, "mode": mode, "ts": datetime.now(timezone.utc).isoformat(), "preview_hash": pv["hash"] if pv else None,
            "applied": applied or [], "run": arg("--run-url")}


def do_apply(pv, want_hash, out_dir):
    """apply exactly the previewed moves: refuses unless the freshly built preview has the same hash the user reviewed"""
    if not want_hash or pv["hash"] != want_hash:
        return result_body(False, f"The lineup or plan changed since you looked (preview {want_hash or 'none'} vs now {pv['hash']}). Refresh the preview, review it, and apply again. Nothing was changed.", "apply", pv)
    if pv["total_moves"] == 0:
        return result_body(True, "Nothing to change: ESPN already has the planned lineup for every remaining day.", "apply", pv)
    if any(m.get("problem") for d in pv["days"] for m in d["moves"]):
        return result_body(False, "The preview has a move that cannot be done (a player missing from your roster). Nothing was changed.", "apply", pv)
    if not APPLY_READY:
        return result_body(False, "Apply is not switched on yet: it is waiting for the ESPN lineup request to be captured and tested. Nothing was changed.", "apply", pv)
    # snapshot of every day's current lineup, written before any change so it can be undone
    snap = {"taken": datetime.now(timezone.utc).isoformat(), "preview_hash": pv["hash"], "days": [{"date": d["date"], "sp": d["sp"], "current": d["current"], "moves": d["moves"]} for d in pv["days"]]}
    (out_dir / f"lineup_snapshot_{pv['hash']}.json").write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    import config
    from espn_api.basketball import League
    lg = League(league_id=config.LEAGUE_ID, year=load("week_plan.json")["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    applied = []
    for d in pv["days"]:
        if not d["moves"]:
            continue
        try:
            write_lineup(lg, d["sp"], d["moves"], team_id=load("week_plan.json")["teams"][0]["id"] and next(t for t in load("week_plan.json")["teams"] if t["abbrev"] == load("week_plan.json")["my_abbrev"])["id"])
            applied.append(d["date"])
        except Exception as ex:
            return result_body(False, f"Stopped at {d['date']}: {ex}. Days already set: {', '.join(applied) or 'none'}. The snapshot lineup_snapshot_{pv['hash']}.json can undo them.", "apply", pv, applied)
    last = {"ts": datetime.now(timezone.utc).isoformat(), "preview_hash": pv["hash"],
            "days": [{"date": d["date"], "sp": d["sp"], "moves": d["moves"]} for d in pv["days"] if d["date"] in applied]}
    (out_dir / "lineup_last_apply.json").write_text(json.dumps(last, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return result_body(True, f"Lineup set for {len(applied)} day(s): {', '.join(applied)}. Use Undo if you want it back the way it was.", "apply", pv, applied)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(arg("--day", "2026-10-20"), "--write" in sys.argv)
        raise SystemExit(0)
    demo = "--demo" in sys.argv
    mode = arg("--mode", "demo" if demo else "preview")
    out_dir = Path(arg("--out", str(HUB)))
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "undo":
        res = do_undo(out_dir)
        (out_dir / "lineup_result.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print("RESULT:", res["message"])
        sys.exit(0 if res["ok"] else 1)
    pv = build_preview(from_scratch=demo)
    pv["apply_supported"] = APPLY_READY
    (out_dir / ("lineup_preview_example.json" if demo else "lineup_preview.json")).write_text(json.dumps(pv, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"matchup {pv['matchup']} ({pv['start']} to {pv['end']}): {pv['total_moves']} lineup moves over {len(pv['days'])} days (preview {pv['hash']})")
    for d in pv["days"]:
        print(f"  {d['date']} ({d['n_nba_games']} NBA games): {len(d['moves'])} moves" + (f"  [{d['note']}]" if d["note"] else ""))
        for m in d["moves"][:12]:
            print(f"      {m['name']:22s} {m['from']:>5s} -> {m['to']}" + (f"   !! {m['problem']}" if m.get("problem") else ""))
    if mode == "apply" or "--apply" in sys.argv:
        res = do_apply(pv, arg("--hash"), out_dir)
    elif mode == "preview":
        res = result_body(True, "Preview refreshed.", "preview", pv)
    else:
        res = None
    if res and not demo:
        (out_dir / "lineup_result.json").write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print("RESULT:", res["message"])
        if not res["ok"] and mode == "apply":
            sys.exit(1)
