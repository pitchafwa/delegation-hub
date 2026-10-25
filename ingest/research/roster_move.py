"""One-tap moves (2026-10-25): make one of the plan's adds in ESPN from the site, through the roster-move workflow (same GitHub-token pattern as Set lineup).

  python research/roster_move.py --add PID [--drop PID] --kind fa|waiver --rid RID --out DIR [--run-url URL] [--dry]

Before anything is sent it checks, against ESPN right now and the newest plan:
  - the move is in the plan (a step of the add sequence with this add and this drop, or the IR stash suggestion)
  - the player to add is still available: a free agent for an add, on waivers for a claim (refuses the wrong kind instead of guessing)
  - the player to drop is still on your roster
  - you have adds left this matchup
Then it sends ONE transaction (add and drop together, so ESPN does both or neither) and re-reads ESPN to confirm: for an add, he is on your roster and the dropped
player is gone; for a claim, the claim is pending. The added player goes to your bench; the plan rebuilds afterwards and the lineup alert / Set lineup take it from there.

Request bodies are the ones ESPN's own pages send, as recorded in this league's transaction log (league_transactions_*.json): FREEAGENT with an ADD item
(fromTeamId 0 -> your team, to the bench slot 12) and a DROP item (your team -> 0); a waiver claim is type WAIVER with bidAmount 0. Same write endpoint and cookie
authentication as the lineup tool (set_lineup.py).
Writes move_result.json (and appends moves_log.json) in --out; the workflow publishes them to the move-data branch, which the site reads.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from set_lineup import SLOT_ID, WRITE_URL, load    # noqa: E402

BENCH = 12


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) else default


def player_status(lg, pid):
    """ESPN's current status of one player: FREEAGENT, WAIVERS or ONTEAM (+ team id)"""
    filt = {"players": {"filterIds": {"value": [int(pid)]}, "filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}}}
    raw = lg.espn_request.league_get(params={"view": "kona_player_info", "scoringPeriodId": lg.scoringPeriodId}, headers={"x-fantasy-filter": json.dumps(filt)})
    p = next((x for x in raw.get("players", []) if int(x.get("id", 0)) == int(pid)), None)
    if not p:
        return None, None, None
    return p.get("status"), p.get("onTeamId"), (p.get("player") or {}).get("fullName")


def my_roster(lg, team_id):
    raw = lg.espn_request.league_get(params={"view": ["mRoster", "mTeam"], "scoringPeriodId": lg.scoringPeriodId})
    t = next(x for x in raw["teams"] if x["id"] == team_id)
    roster = {e["playerId"]: e for e in t["roster"]["entries"]}
    used = ((t.get("transactionCounter") or {}).get("matchupAcquisitionTotals") or {})
    return roster, used


def in_plan(T, add, drop):
    for m in T.get("sequence") or []:
        if int(m["add"]["id"]) == add and int((m.get("drop") or {}).get("id") or 0) == (drop or 0):
            return m
    st = T.get("stash_suggest")
    if st and int(st["id"]) == add:
        return {"add": {"id": add, "name": st["name"]}, "drop": None, "stash": True}
    return None


def payload(kind, team_id, sp, add, drop, drop_slot):
    import config
    items = [{"playerId": add, "type": "ADD", "fromTeamId": 0, "toTeamId": team_id, "fromLineupSlotId": -1, "toLineupSlotId": BENCH if kind == "fa" else -1}]
    if drop:
        items.append({"playerId": drop, "type": "DROP", "fromTeamId": team_id, "toTeamId": 0, "fromLineupSlotId": drop_slot, "toLineupSlotId": -1})
    body = {"isLeagueManager": False, "teamId": team_id, "type": "FREEAGENT" if kind == "fa" else "WAIVER", "memberId": config.SWID,
            "scoringPeriodId": sp, "executionType": "EXECUTE", "items": items}
    if kind == "waiver":
        body["bidAmount"] = 0
    return body


def run():
    import config
    import requests
    from espn_api.basketball import League
    add, drop, kind = int(arg("--add")), int(arg("--drop") or 0) or None, arg("--kind", "fa")
    if kind not in ("fa", "waiver"):
        return False, "Unknown move type."
    W = load("week_plan.json")
    T = next(t for t in W["teams"] if t["abbrev"] == W["my_abbrev"])
    team_id = T["id"]
    step = in_plan(T, add, drop)
    if not step:
        return False, "That move is no longer in the plan (it may have changed since you looked). Refresh the page and check the plan. Nothing was changed."
    lg = League(league_id=config.LEAGUE_ID, year=W["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    status, on_team, name = player_status(lg, add)
    name = name or step["add"].get("name", str(add))
    if status == "ONTEAM":
        return False, f"{name} is no longer available ({'already on your team' if on_team == team_id else 'another team has him'}). Nothing was changed."
    if status is None:
        return False, f"ESPN did not return {name}'s status. Nothing was changed."
    if kind == "fa" and status == "WAIVERS":
        return False, f"{name} is on waivers: this needs a claim, not an add. Nothing was changed."
    if kind == "waiver" and status == "FREEAGENT":
        return False, f"{name} has cleared waivers and is a free agent now: use Add instead. Nothing was changed."
    roster, used = my_roster(lg, team_id)
    drop_name = (step.get("drop") or {}).get("name")
    if drop and drop not in roster:
        return False, f"{drop_name or drop} is not on your roster any more. Nothing was changed."
    limit = W["matchup"]["adds_limit"]
    if kind == "fa" and used.get(str(W["matchup"]["id"]), 0) >= limit:
        return False, f"All {limit} adds for this matchup are used. Nothing was changed."
    body = payload(kind, team_id, lg.scoringPeriodId, add, drop, roster[drop]["lineupSlotId"] if drop else None)
    if "--dry" in sys.argv:
        print("DRY RUN, would POST:", json.dumps({**body, "memberId": "<SWID>"}))
        return True, "Dry run: nothing sent."
    url = WRITE_URL.format(season=W["season"], league=config.LEAGUE_ID)
    r = requests.post(url, json=body, cookies={"espn_s2": config.ESPN_S2, "SWID": config.SWID}, timeout=30,
                      headers={"Accept": "application/json", "X-Fantasy-Source": "kona", "Origin": "https://fantasy.espn.com", "Referer": "https://fantasy.espn.com/", "User-Agent": "Mozilla/5.0"})
    if r.status_code not in (200, 201):
        return False, f"ESPN refused the move ({r.status_code}): {r.text[:220]}. Nothing was changed; you can make it in the ESPN app."
    try:
        j = r.json()
    except ValueError:
        j = {}
    st = str((j or {}).get("status", "")).upper() if isinstance(j, dict) else ""
    if st.startswith("FAIL") or st in ("ERROR", "INVALID", "CANCELED"):
        return False, f"ESPN did not make the move (status {st}). Nothing was changed; you can make it in the ESPN app."
    who = f"Added {name}" + (f", dropped {drop_name}" if drop else "") if kind == "fa" else f"Claim placed for {name}" + (f" (drop {drop_name} if it goes through)" if drop else "")
    if kind == "fa":
        roster2, _ = my_roster(lg, team_id)
        if add not in roster2 or (drop and drop in roster2):
            return False, f"ESPN answered OK (status {st or 'none'}) but your roster does not show the change yet. Check the ESPN app before trying again."
        return True, who + ". He is on your bench: the plan rebuilds now, then set your lineup."
    return True, who + f" (ESPN status {st or 'accepted'}). It is decided when waivers run."


if __name__ == "__main__":
    out = Path(arg("--out", "."))
    out.mkdir(parents=True, exist_ok=True)
    try:
        ok, msg = run()
    except Exception as ex:
        ok, msg = False, f"The move could not be checked or sent ({str(ex)[:180]}). Nothing was changed as far as we know; check the ESPN app."
    res = {"ok": ok, "message": msg, "rid": arg("--rid"), "add": arg("--add"), "drop": arg("--drop"), "kind": arg("--kind"), "ts": datetime.now(timezone.utc).isoformat(), "run": arg("--run-url")}
    (out / "move_result.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    log = out / "moves_log.json"
    hist = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    hist.append(res)
    log.write_text(json.dumps(hist[-200:], ensure_ascii=False), encoding="utf-8")
    print("RESULT:", ok, msg)
    sys.exit(0 if ok else 1)
