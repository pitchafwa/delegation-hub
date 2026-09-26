"""Build dashboard/schedule_plan.json: Phase 2 of the start/sit planner (schedule-aware roster planning).

Reads (all written by other scripts, no ESPN login needed):
  dashboard/nba_schedule.json   full-season NBA schedule (pull_nba_schedule.py)
  dashboard/week_plan.json      every team's roster (levels, statuses, slots) + the free-agent pool (build_week_plan.py)

Produces
  * calendar        the 22 fantasy matchup periods (ASSUMED until the season starts: 6-day opener, 7-day weeks, a 14-day All-Star matchup,
                    weeks 20-22 = playoffs, ending Mar 28)
  * nba             games and back-to-backs per NBA team per fantasy week (heat map), and the league average
  * teams           for each fantasy team: every player's games per week, extra games vs an average team over the next 4 weeks and the
                    playoff weeks, flags (light week / heavy playoffs), projected points for the next 4 weeks and playoff weeks 20-22
  * playoffs        projected points weeks 20-22 for every team, and for the requested team the best free-agent adds and trade targets
                    ranked by playoff-week points gained
Same lineup solver as Phase 1 (cap-aware DP over days, position-matched daily slots), with the "future game" availability rates measured in
availability_by_level.py (players' later-game play rate is lower than the same-day rate).
Run from ingest/:  uv run python research/build_schedule_plan.py
"""
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
SEASON_START = date(2026, 10, 20)
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
CAP_PER_7 = 40.0
LENGTHS = [6] + [7] * 15 + [14] + [7] * 2 + [7] * 3
PLAYOFF_WEEKS = (20, 21, 22)
FIX = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
OUT_STATUS = ("OUT", "INJURY_RESERVE", "SUSPENSION")
RETURN_FACTOR = 0.6      # ASSUMPTION: a player who is OUT now is back for weeks 3+ at 60% of the normal play rate (unknown return dates)
NEAR_WEEKS = 2           # ...and counts as absent for this many upcoming weeks
FA_ANCHOR, FA_SHRINK = 22.0, 0.6   # (fa_pool levels in week_plan.json are already shrunk)

canon = lambda t: FIX.get(t, t)

CAL = []
_d = SEASON_START
for i, n in enumerate(LENGTHS, start=1):
    CAL.append({"id": i, "start": _d, "end": _d + timedelta(days=n - 1), "days": n, "cap": round(CAP_PER_7 * n / 7.0, 1), "playoff": i in PLAYOFF_WEEKS})
    _d += timedelta(days=n)

sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))
GAMES = {}    # date -> set of teams playing
for ds, gl in sched["games"].items():
    GAMES[date.fromisoformat(ds)] = {canon(t) for g in gl for t in g[:2] if t != "TBD"}
wp = json.load(open(HUB / "week_plan.json", encoding="utf-8"))
from zoneinfo import ZoneInfo
today = datetime.now(ZoneInfo("America/New_York")).date()
cur = next((w for w in CAL if w["start"] <= max(today, SEASON_START) <= w["end"]), CAL[0])
FIRST = cur["id"]


def plays(team, d):
    return team in GAMES.get(d, ())


def b2b(team, d):
    return plays(team, d) and plays(team, d - timedelta(days=1))


def wk_days(w):
    return [w["start"] + timedelta(days=i) for i in range(w["days"])]


# ---------------- heat map data
nba_teams = sorted({t for s in GAMES.values() for t in s})
heat = {}
for t in nba_teams:
    row = []
    for w in CAL:
        ds = wk_days(w)
        row.append({"g": sum(plays(t, d) for d in ds), "b": sum(b2b(t, d) for d in ds)})
    heat[t] = row
avg_games = [round(sum(heat[t][i]["g"] for t in nba_teams) / len(nba_teams), 2) for i in range(len(CAL))]
games_seen = sum(len(s) for s in GAMES.values()) // 2
print(f"schedule: {len(GAMES)} game days, ~{games_seen} games; current fantasy week {FIRST}")


# ---------------- solver (same as Phase 1, parameterised by week)
def p_future(level):
    return 0.61 if level < 15 else (0.81 if level < 20 else 0.85)


def p_play(pl, d, wid):
    if pl["team"] and not plays(pl["team"], d):
        return 0.0
    if not pl["team"]:
        return 0.0
    p = p_future(pl["level"])
    if pl.get("back"):                               # stash candidate who is out now: nothing until his estimated return date, then 85% of the normal play rate
        if d.isoformat() < pl["back"]:
            return 0.0
        p *= 0.85
    elif pl["ir"] or pl["status"] in OUT_STATUS:
        if wid < FIRST + NEAR_WEEKS:
            return 0.0
        p *= RETURN_FACTOR
    elif b2b(pl["team"], d):
        p *= 0.89 / 0.94
    return p


def slot_matcher(cands):
    match = {}

    def try_assign(ci, seen):
        for si, s in enumerate(SLOTS):
            if s in cands[ci][1]["slots"] and si not in seen:
                seen.add(si)
                if si not in match or try_assign(match[si], seen):
                    match[si] = ci
                    return True
        return False

    return [ci for ci in range(len(cands)) if try_assign(ci, set())]


def day_options(roster, d, wid):
    cands = sorted(((p_play(pl, d, wid) * pl["level"], pl, p_play(pl, d, wid)) for pl in roster if p_play(pl, d, wid) > 0), key=lambda x: -x[0])
    acc = slot_matcher(cands)
    vals, cnt = [0.0], [0.0]
    for ci in acc:
        vals.append(vals[-1] + cands[ci][0])
        cnt.append(cnt[-1] + cands[ci][2])
    return vals, cnt


def week_points(roster, wid):
    w = CAL[wid - 1]
    days = wk_days(w)
    cap = w["cap"]
    opts = [day_options(roster, d, wid) for d in days]
    NS = int(cap) + 12
    memo = {}

    def best(di, c):
        if di == len(days) or c >= cap:
            return 0.0
        key = (di, c)
        if key in memo:
            return memo[key]
        vals, cnt = opts[di]
        top = 0.0
        for m in range(len(vals)):
            v = vals[m] + best(di + 1, min(int(round(c + cnt[m])), NS))
            if v > top:
                top = v
        memo[key] = top
        return top

    return best(0, 0)


def points_over(roster, weeks):
    return sum(week_points(roster, w) for w in weeks)


# ---------------- players
def mk(p):
    return {"id": p["id"], "name": p["name"], "team": canon(p.get("team") or ""), "slots": [s for s in p.get("slots", []) if s in set(SLOTS)] or ["UT"],
            "status": p.get("status", "ACTIVE"), "level": p.get("level") or 0.0, "ir": bool(p.get("ir")), "asset_rank": p.get("asset_rank"), "protected": bool(p.get("protected"))}


def games_row(pl):
    return [heat[pl["team"]][i]["g"] if pl["team"] in heat else 0 for i in range(len(CAL))]


NEXT4 = [w["id"] for w in CAL if FIRST <= w["id"] < FIRST + 4]
PLAY = list(PLAYOFF_WEEKS)
avg_next4 = sum(avg_games[i - 1] for i in NEXT4)
avg_play = sum(avg_games[i - 1] for i in PLAY)
teams_out = []
rosters = {}
for t in wp["teams"]:
    r = [mk(p) for p in t["roster"]]
    rosters[t["id"]] = r
    prow = []
    for pl in r:
        gr = games_row(pl)
        n4, npo = sum(gr[i - 1] for i in NEXT4), sum(gr[i - 1] for i in PLAY)
        flags = []
        if pl["level"] >= 24 and not pl["ir"]:
            wk_next = gr[FIRST - 1] if FIRST - 1 < len(gr) else 0
            if any(gr[i - 1] <= avg_games[i - 1] - 1.5 for i in NEXT4[:2]):
                flags.append("light week soon")
            if npo >= avg_play + 1.5:
                flags.append("heavy playoffs")
            elif npo <= avg_play - 1.5:
                flags.append("light playoffs")
        prow.append({"id": pl["id"], "name": pl["name"], "team": pl["team"], "level": pl["level"], "status": pl["status"], "ir": pl["ir"], "games": gr,
                     "x4": round(n4 - avg_next4, 1), "xp": round(npo - avg_play, 1), "flags": flags})
    active = [p for p in r]
    raw_proj = {w: week_points(active, w) for w in range(FIRST, len(CAL) + 1)}     # every remaining week (the playoff-odds simulation needs them all)
    wa = {int(k): v for k, v in (t.get("weekly_actual") or {}).items() if CAL[int(k) - 1]["days"] == 7 and int(k) not in (1, 17)}
    anchors = sorted(wa)[-3:]
    if anchors:       # BLEND (backtested: beats the raw solver, MAE 178 vs 207 one week ahead): real scoring in recent full weeks x how much lighter/heavier the target week is
        ratios = {w: sum(wa[a] * raw_proj[w] / max(week_points(active, a), 1.0) for a in anchors) / len(anchors) for w in raw_proj}
        proj = {str(w): round(ratios[w], 0) for w in raw_proj}
        basis = "blend"
    else:             # preseason: raw solver, scaled by 0.95 (it ran 5-7% high in the 2025-26 backtest)
        proj = {str(w): round(0.95 * raw_proj[w], 0) for w in raw_proj}
        basis = "solver"
    teams_out.append({"id": t["id"], "abbrev": t["abbrev"], "name": t["name"], "players": prow, "proj": proj, "proj_basis": basis})
    print(f"{t['abbrev']:5s} next4 {sum(proj[str(w)] for w in NEXT4):7.0f}  playoffs {sum(proj[str(w)] for w in PLAY):7.0f}", flush=True)

# ---------------- playoff planner for one team (default: mine): best FA adds and trade targets by playoff-week points
me = next((t for t in wp["teams"] if t["abbrev"] == wp["my_abbrev"]), wp["teams"][0])
mine = rosters[me["id"]]
base_po = points_over(mine, PLAY)
fa = [mk(dict(p, ir=False, protected=False)) for p in wp.get("fa_pool", [])]
non_ir = [p for p in mine if not p["ir"]]
droppable = sorted([p for p in non_ir if not p["protected"]], key=lambda p: p["level"])
adds = []
for f in fa:
    if f["team"] not in heat:
        continue
    drop = droppable[0] if len(non_ir) >= 15 and droppable else None
    new = [p for p in mine if p is not drop] + [f]
    g = points_over(new, PLAY) - base_po
    gr = games_row(f)
    adds.append({"name": f["name"], "team": f["team"], "level": f["level"], "gain": round(g, 0), "games": [gr[i - 1] for i in PLAY], "drop": drop["name"] if drop else None})
adds.sort(key=lambda x: -x["gain"])
targets = []
for t in wp["teams"]:
    if t["id"] == me["id"]:
        continue
    for pl in rosters[t["id"]]:
        if pl["level"] < 26 or pl["ir"] or pl["team"] not in heat:
            continue
        g = points_over(mine + [pl], PLAY) - base_po
        gr = games_row(pl)
        targets.append({"name": pl["name"], "team": pl["team"], "owner": t["abbrev"], "level": pl["level"], "gain": round(g, 0), "games": [gr[i - 1] for i in PLAY], "asset_rank": pl["asset_rank"]})
targets.sort(key=lambda x: -x["gain"])
print("best playoff FA adds:", [(a["name"], a["gain"]) for a in adds[:3]])

# streamers before a light week for your stars: FAs on teams with heavy next-week schedules
nxt = FIRST
streamers = sorted([{"name": f["name"], "team": f["team"], "level": f["level"], "games": games_row(f)[nxt - 1], "pts": round(f["level"] * games_row(f)[nxt - 1], 0)} for f in fa if f["team"] in heat],
                   key=lambda x: -x["pts"])[:12]

# ---------------- long-term adds (ROS stash): value over the REST OF THE SEASON of free agents the this-week search cannot see (out now, idle team) or who only help later
REMAIN = list(range(FIRST, len(CAL) + 1))
PLAYOFF_WT = 1.5
IR_OK_S = ("OUT", "INJURY_RESERVE")
base_w = {w: week_points(mine, w) for w in REMAIN}
mine_ir = sum(1 for p in mine if p["ir"])
stash = []
for src, pool in (("stash", wp.get("stash_pool", [])), ("fa", wp.get("fa_pool", []))):
    for x in pool:
        if x["team"] not in heat and canon(x["team"]) not in heat:
            continue
        if src == "stash" and x.get("out_now") and not x.get("back_date"):
            continue                                                 # out for the season
        f = mk(dict(x, ir=False, protected=False, status="ACTIVE"))
        if x.get("out_now"):
            f["back"] = x["back_date"]
        to_ir = bool(x.get("out_now")) and x.get("status") in IR_OK_S and mine_ir < 4
        drop = None if to_ir or len(non_ir) < 15 else (droppable[0] if droppable else None)
        new = [p for p in mine if p is not drop] + [f]
        gw = {w: week_points(new, w) - base_w[w] for w in REMAIN}
        reg = sum(g for w, g in gw.items() if w not in PLAY)
        po = sum(g for w, g in gw.items() if w in PLAY)
        n4 = sum(gw[w] for w in NEXT4)
        total_w = reg + PLAYOFF_WT * po
        stash.append({"form": x.get("form"), "id": x["id"], "name": x["name"], "team": f["team"], "level": x["level"], "src": src, "out_now": bool(x.get("out_now")), "status": x.get("status"), "injury": x.get("injury"),
                      "games_out": x.get("games_out"), "back_date": x.get("back_date"), "espn_return": x.get("espn_return"), "p_back_playoffs": x.get("p_back_playoffs"),
                      "age": x.get("age"), "asset": x.get("asset"), "asset_rank": x.get("asset_rank"), "market_rank": x.get("market_rank"), "kind": x.get("kind"), "slots": x.get("slots"),
                      "gain": round(total_w), "reg": round(reg), "po": round(po), "next4": round(n4), "drop": None if to_ir or not drop else drop["name"], "to_ir": to_ir})
keep = [e for e in stash if (e["src"] == "stash" and e["gain"] >= 60) or (e["src"] == "fa" and e["gain"] >= 100 and (e["gain"] - e["next4"]) / max(len(REMAIN) - 4, 1) >= 1.75 * max(e["next4"], 0) / 4)]      # healthy free agents only when the gain is back-loaded (the near-term ones are in the suggested moves)
keep.sort(key=lambda e: -e["gain"])
print("long-term (ROS) stash:", [(e["name"], e["gain"], e["next4"]) for e in keep[:6]])

out = {"generated": datetime.now(timezone.utc).isoformat(), "calendar_assumed": True, "schedule_provisional": True,
       "calendar": [{"id": w["id"], "start": w["start"].isoformat(), "end": w["end"].isoformat(), "days": w["days"], "cap": w["cap"], "playoff": w["playoff"]} for w in CAL],
       "current_week": FIRST, "next4": NEXT4, "playoff_weeks": PLAY, "avg_games": avg_games, "avg_next4": round(avg_next4, 1), "avg_playoffs": round(avg_play, 1),
       "nba": heat, "teams": teams_out, "my_abbrev": wp["my_abbrev"],
       "stash": keep[:15], "playoffs": {"base": round(base_po, 0), "fa_adds": adds[:10], "trade_targets": targets[:15], "streamers_next_week": streamers}}
(HUB / "schedule_plan.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote schedule_plan.json", round((HUB / "schedule_plan.json").stat().st_size / 1024), "KB")
