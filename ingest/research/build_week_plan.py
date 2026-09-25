"""Build dashboard/week_plan.json: the cap-aware "This week" plan for every team in the league.

For the current (or next) matchup it projects each player-game, then plans each team's lineups DAY BY DAY to maximize expected
fantasy points under this league's rules:
  * 10 starting slots (PG SG SF PF C G F + 3 UTIL), position-eligible, daily lineups
  * games-played cap: 40 started games per 7 days (scaled by matchup length). It is checked at the START of each day: begin a
    day under the cap and you may start anyone (and go over); begin it at/over the cap and the rest of the matchup scores nothing.
  * The objective is POINTS, not games: sometimes the best plan crosses the cap early (strong Saturday, thin Sunday), sometimes it sits
    weak starters to keep games for a better day. A small dynamic program over (day, starts so far) finds the best plan.
It also ranks free-agent adds (best add/drop pairs by weekly points gained, using the league's add limit) and projects the opponent.

Per-game expected points = P(plays) x level.
  level  = average of our season model (hub_data year0_ppg) and ESPN's projection; in season, blended with ESPN's last-15-game average
  P(play)= 0.94 healthy rotation player (short absences ~6%); x(0.89/0.94) on the second night of a back-to-back (measured, 10.9% vs 6.1%
           short absences); DAY_TO_DAY 0.55 (assumption, unmeasured); OUT / IR / SUSPENSION 0.
Nothing here uses Vegas lines: tested out-of-sample and they added nothing beyond recent form.
Run from ingest/:  uv run python research/build_week_plan.py
"""
import json
import math
import os
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
SEASON_ID = 2027
SEASON_START = date(2026, 10, 20)          # scoring period 1 (season opener, a Tuesday)
MY_ABBREV = "DRNK"
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
CAP_PER_7 = 40.0
ADDS_PER_DAY = 8 / 7
WEEK_SD = 253.0                             # measured week-to-week SD of a team's weekly points (2025-26)
FIX = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
ET = ZoneInfo("America/New_York")


def canon(t):
    return FIX.get(t, t)


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


# ---------- calendar (ESPN 2026-27 matchup dates are an ASSUMPTION until in-season: 6-day opener, 7-day weeks, one 14-day All-Star matchup #17)
LENGTHS = [6] + [7] * 15 + [14] + [7] * 2 + [7] * 3
BOUNDS, _d = [], SEASON_START
for _i, _n in enumerate(LENGTHS, start=1):
    BOUNDS.append((_i, _d, _d + timedelta(days=_n - 1)))
    _d += timedelta(days=_n)


def matchup_for(day):
    for i, a, b in BOUNDS:
        if a <= day <= b:
            return i, a, b
    return (1, *BOUNDS[0][1:]) if day < SEASON_START else BOUNDS[-1]


# ---------- data
lg = League(league_id=config.LEAGUE_ID, year=SEASON_ID, espn_s2=config.ESPN_S2, swid=config.SWID)
hub = json.load(open(HUB / "hub_data.json", encoding="utf-8"))
hub_by_id = {p["id"]: p for p in hub["players"]}
hub_by_name = {norm(p["player"]): p for p in hub["players"]}
id_map_path = Path(__file__).resolve().parent / "espn_id_map.json"
e2n = {int(k): int(v) for k, v in json.load(open(id_map_path)).items()} if id_map_path.exists() else {}


def hub_player(espn_id, name):
    nba = e2n.get(int(espn_id))
    if nba is not None:
        for pre in ("c", "p"):
            if f"{pre}{nba}" in hub_by_id:
                return hub_by_id[f"{pre}{nba}"]
    return hub_by_name.get(norm(name))


_ranked = sorted([p for p in hub["players"] if p.get("asset_k")], key=lambda p: -p["asset_k"][5])
ASSET_RANK = {p["id"]: i + 1 for i, p in enumerate(_ranked)}
# ---- who may be suggested as a DROP. Keeper league: only 5 players are kept, so a non-keeper's value is what he produces THIS season.
KEEPER_PROTECT = 6         # each team's top-6 dynasty assets (5 keepers + a margin) are never suggested as drops
PROTECT_LEVEL = 35         # projects 35+ pts/g: never dropped, valued or not (catches stars our model can't value, e.g. after a long injury)
ROS_WEEKS = 6              # if the player you drop is better than the one you add, count that gap for this many future weeks (before you could re-stream)
GAMES_PER_WEEK = 3.3
MIN_NET_GAIN = 10          # a move must be worth at least this many points AFTER subtracting the future cost
SOLID_LEVEL = 28           # drops at/above this projection are flagged 'solid contributor'
NEVER_DROP = {norm(n) for n in json.load(open(Path(__file__).resolve().parent / "never_drop.json")).get("names", [])} if (Path(__file__).resolve().parent / "never_drop.json").exists() else set()


def protected_ids(roster):
    valued = sorted([p for p in roster if p["asset_rank"]], key=lambda p: p["asset_rank"])
    keep = {p["espn_id"] for p in valued[:KEEPER_PROTECT]}
    keep |= {p["espn_id"] for p in roster if p["level"] >= PROTECT_LEVEL or norm(p["name"]) in NEVER_DROP}
    return keep


def future_cost(dr, add):
    """points of production lost in later weeks by swapping dr for add (only if dr is the better player). Injury/short schedule this week don't matter."""
    if dr is None:
        return 0.0
    return max(0.0, dr["level"] - add["level"]) * GAMES_PER_WEEK * ROS_WEEKS


def drop_flag(p):
    return "solid" if p["level"] >= SOLID_LEVEL else ""


today = datetime.now(ET).date()
mp_id, mp_start, mp_end = matchup_for(max(today, SEASON_START))
days = [mp_start + timedelta(days=i) for i in range((mp_end - mp_start).days + 1)]
plan_days = [d for d in days if d >= today]
if not plan_days:
    plan_days = days
n_days = len(days)
cap = CAP_PER_7 * n_days / 7.0
adds_limit = round(ADDS_PER_DAY * n_days)
print(f"matchup {mp_id}: {mp_start}..{mp_end} ({n_days} days), cap {cap:.1f}, adds limit {adds_limit}; planning {len(plan_days)} days from {plan_days[0]}")

# NBA schedule for the matchup (+ the day before, for back-to-back detection)
games = {}     # date -> {team: iso tipoff}
for d in [days[0] - timedelta(days=1)] + days:
    try:
        j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", params={"dates": d.strftime("%Y%m%d")}, timeout=30).json()
    except Exception:
        j = {}
    g = {}
    for ev in j.get("events", []):
        for c in ev["competitions"][0]["competitors"]:
            ab = c["team"].get("abbreviation")
            if ab:
                g[canon(ab)] = ev["date"]
    games[d] = g


def plays(team, d):
    return canon(team) in games.get(d, {})


def b2b(team, d):
    return plays(team, d) and plays(team, d - timedelta(days=1))


# ---------- players
RECENT_FORM_MAX = 0.45  # weight on the last-15-game average once 15 games are in (measured best: ~50/50 recent vs long-run). Interim until a true per-game projection source is found; 0 = ESPN as-is
STATUS_P = {"ACTIVE": 0.94, "DAY_TO_DAY": 0.55, "OUT": 0.0, "INJURY_RESERVE": 0.0, "SUSPENSION": 0.0}


def make_player(p, on_roster_slot=None):
    hp = hub_player(p.playerId, p.name)
    espn_proj = (p.stats.get(f"{SEASON_ID}_projected") or {}).get("applied_avg")
    ours = hp.get("year0_ppg") if hp else None
    base = espn_proj if espn_proj else (ours or 0.0)          # ESPN's projection first; our model only when ESPN has none
    src = "ESPN" if espn_proj else ("model" if ours else "none")
    l15 = p.stats.get(f"{SEASON_ID}_last_15") or {}
    gp15 = (l15.get("total") or {}).get("GP") or 0
    if gp15 >= 3 and l15.get("applied_avg"):
        w = RECENT_FORM_MAX * min(gp15, 15) / 15.0
        base = w * l15["applied_avg"] + (1 - w) * base
        src += "+recent"
    return {"espn_id": p.playerId, "name": p.name, "team": canon(p.proTeam or ""), "slots": [s for s in p.eligibleSlots if s in set(SLOTS)],
            "status": p.injuryStatus or "ACTIVE", "level": round(base, 1), "src": src, "espn_level": espn_proj, "model_level": ours, "espn_avg": (p.stats.get(f"{SEASON_ID}_projected") or {}).get("avg"), "hub_id": hp["id"] if hp else None,
            "ir": on_roster_slot == "IR", "asset_rank": ASSET_RANK.get(hp["id"]) if hp else None, "kind": hp.get("kind") if hp else None}


def level_on(pl, d):
    """projected points per game on date d: sportsbook-prop-derived when we have it for that day, else the base level"""
    return pl.get("lvl_by_date", {}).get(d.isoformat(), pl["level"])


def p_play(pl, d):
    if not plays(pl["team"], d):
        return 0.0
    if pl["ir"]:
        return 0.0
    p = STATUS_P.get(pl["status"], 0.94)
    if p and pl["status"] == "ACTIVE" and b2b(pl["team"], d):
        p *= 0.89 / 0.94
    return p


def slot_matcher(cands):
    """greedy (matroid) selection: players sorted by expected points; a player is added if the chosen set can still be matched to slots.
    Returns the accepted list in order with their slot assignment after each addition."""
    match = {}      # slot index -> candidate index

    def try_assign(ci, seen):
        for si, s in enumerate(SLOTS):
            if s in cands[ci][1]["slots"] and si not in seen:
                seen.add(si)
                if si not in match or try_assign(match[si], seen):
                    match[si] = ci
                    return True
        return False

    accepted = []
    for ci, (ef, pl, p) in enumerate(cands):
        if try_assign(ci, set()):
            accepted.append(ci)
    return accepted


def day_options(roster, d):
    """best achievable expected points when starting exactly m players (m=0..M), the players used, and expected starts."""
    cands = []
    for pl in roster:
        p = p_play(pl, d)
        if p > 0:
            cands.append((p * level_on(pl, d), pl, p))
    cands.sort(key=lambda x: -x[0])
    acc = slot_matcher(cands)
    vals, cnt, order = [0.0], [0], []
    for ci in acc:
        ef, pl, p = cands[ci]
        vals.append(vals[-1] + ef)
        cnt.append(cnt[-1] + p)
        order.append(ci)
    return cands, order, vals, cnt


def assign_slots(chosen):
    """final slot names for a set of (pl) via maximum matching (small)"""
    match = {}

    def go(i, seen):
        for si, s in enumerate(SLOTS):
            if s in chosen[i]["slots"] and si not in seen:
                seen.add(si)
                if si not in match or go(match[si], seen):
                    match[si] = i
                    return True
        return False

    for i in range(len(chosen)):
        go(i, set())
    out = {}
    for si, i in match.items():
        out[i] = SLOTS[si]
    return [out.get(i, "UT") for i in range(len(chosen))]


def plan_team(roster, c0=0.0, detail=False):
    opts = [day_options(roster, d) for d in plan_days]
    NS = int(cap) + 12
    NEG = -1e9
    # dp[day][c] = best expected points from this day on, given c starts so far (c integer, capped)
    memo = {}

    def best(di, c):
        if di == len(plan_days):
            return 0.0, None
        key = (di, c)
        if key in memo:
            return memo[key]
        cands, order, vals, cnt = opts[di]
        if c >= cap:                       # locked: nothing counts for the rest of the matchup
            memo[key] = (0.0, None)
            return memo[key]
        top = (NEG, None)
        for m in range(len(vals)):
            c2 = min(int(round(c + cnt[m])), NS)
            v = vals[m] + best(di + 1, c2)[0]
            if v > top[0] + 1e-9:
                top = (v, m)
        memo[key] = top
        return top

    total = best(0, int(round(c0)))[0]
    if not detail:
        return total
    rows, c = [], int(round(c0))
    naive_c, naive_total = c, 0.0
    for di, d in enumerate(plan_days):
        cands, order, vals, cnt = opts[di]
        locked = c >= cap
        m = 0 if locked else best(di, c)[1]
        chosen = [cands[order[k]][1] for k in range(m)]
        slots = assign_slots(chosen)
        starters = [{"id": pl["espn_id"], "name": pl["name"], "slot": s, "ef": round(cands[order[k]][0], 1), "p": round(cands[order[k]][2], 2), "props": d.isoformat() in pl.get("lvl_by_date", {})}
                    for k, (pl, s) in enumerate(zip(chosen, slots))]
        benched = [{"id": cands[order[k]][1]["espn_id"], "name": cands[order[k]][1]["name"], "ef": round(cands[order[k]][0], 1)} for k in range(m, len(order))]
        c_after = c + (cnt[m] if not locked else 0)
        rows.append({"date": d.isoformat(), "nba_teams": len(games.get(d, {})), "start": starters, "bench_playing": benched,
                     "pts": round(vals[m] if not locked else 0.0, 1), "starts_before": round(c, 1), "starts_after": round(c_after, 1),
                     "locked": bool(locked), "sat": (len(order) - m) if not locked else len(order)})
        # 'start everyone possible' comparison
        if naive_c < cap:
            naive_total += vals[-1]
            naive_c += cnt[-1]
        c = int(round(c_after))
    return total, rows, naive_total


def erf_win(diff, sd):
    return 0.5 * (1 + math.erf(diff / (sd * math.sqrt(2))))


# ---------- teams
raw = lg.espn_request.league_get(params={"view": ["mMatchupScore"]})
opp = {}
for m in raw.get("schedule", []):
    if m.get("matchupPeriodId") == mp_id and "away" in m:
        opp[m["home"]["teamId"]] = m["away"]["teamId"]
        opp[m["away"]["teamId"]] = m["home"]["teamId"]

# games started / points so far this matchup (in season only)
so_far = {t.team_id: {"starts": 0.0, "pts": 0.0} for t in lg.teams}
past = [d for d in days if d < today]
if past:
    for k, d in enumerate(past):
        sp = (d - SEASON_START).days + 1
        b = lg.espn_request.league_get(params={"view": "mBoxscore", "scoringPeriodId": sp})
        for m in b.get("schedule", []):
            if m.get("matchupPeriodId") != mp_id:
                continue
            for side in ("home", "away"):
                t = m.get(side, {})
                for e in (t.get("rosterForCurrentScoringPeriod") or {}).get("entries", []):
                    if (e.get("lineupSlotId") is not None) and e["lineupSlotId"] <= 11:
                        pl = e.get("playerPoolEntry", {}).get("player", {})
                        st = [s for s in pl.get("stats", []) if s.get("scoringPeriodId") == sp and s.get("statSourceId") == 0]
                        if st and st[0].get("stats", {}).get("42", 0):
                            so_far[t["teamId"]]["starts"] += 1
                            so_far[t["teamId"]]["pts"] += st[0].get("appliedTotal") or 0

counters = {tm["id"]: tm.get("transactionCounter") or {} for tm in lg.espn_request.league_get(params={"view": "mTeam"}).get("teams", [])}
_c0 = float(os.environ.get("WEEK_C0", 0))     # testing only: pretend every team already has this many starts
for _k in so_far:
    so_far[_k]["starts"] += _c0
rosters = {}
for t in lg.teams:
    rosters[t.team_id] = [make_player(p, p.lineupSlot) for p in t.roster]

fa_players = [make_player(p) for p in lg.free_agents(size=150)]
fa_players = [p for p in fa_players if p["level"] > 0 and p["team"] and any(plays(p["team"], d) for d in plan_days)]


def week_games(pl):
    return sum(1 for d in plan_days if p_play(pl, d) > 0)


fa_players.sort(key=lambda p: -(p["level"] * week_games(p)))
fa_players = fa_players[:60]

# ---------- sportsbook player props (optional): market projections for today/tomorrow, converted to league scoring
props_meta = {"enabled": False}
ODDS_KEY = os.environ.get("ODDS_API_KEY")
if ODDS_KEY:
    try:
        import props_projection as PP
        prop_dates = [d for d in plan_days if (d - today).days <= 1]
        props, pm = PP.fetch_player_props(ODDS_KEY, prop_dates)
        n_used = 0
        for pl in [x for rr in rosters.values() for x in rr] + fa_players:
            byd = props.get(norm(pl["name"]))
            if not byd:
                continue
            for dt, means in byd.items():
                lvl, used = PP.fantasy(means, pl.get("espn_avg"), pl.get("espn_level"))
                if lvl > 0 and used:
                    pl.setdefault("lvl_by_date", {})[dt] = round(lvl, 1)
                    n_used += 1
        props_meta = {"enabled": True, "events": pm["events"], "players_with_props": pm["players"], "player_days_used": n_used, "credits_remaining": pm["remaining"]}
        print("props:", props_meta)
    except Exception as ex:
        props_meta = {"enabled": False, "error": str(ex)[:120]}
        print("props unavailable:", ex)

# ---------- shadow log: today's projections from each source, so they can be scored against real results once games are played
import csv
LOG = HUB / "projection_log.csv"
seen_log = set()
if LOG.exists():
    with open(LOG, encoding="utf-8") as fh:
        seen_log = {(r["date"], r["espn_id"]) for r in csv.DictReader(fh)}
new_rows = []
for pl in [x for rr in rosters.values() for x in rr] + fa_players:
    key = (today.isoformat(), str(pl["espn_id"]))
    if today in days and plays(pl["team"], today) and key not in seen_log and pl["level"] > 0:
        new_rows.append({"date": today.isoformat(), "espn_id": pl["espn_id"], "name": pl["name"], "team": pl["team"], "status": pl["status"],
                         "espn": pl["espn_level"] or "", "model": pl["model_level"] or "", "props": pl.get("lvl_by_date", {}).get(today.isoformat(), "")})
    seen_log.add(key)
if new_rows:
    fresh = not LOG.exists()
    with open(LOG, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(new_rows[0].keys()))
        if fresh:
            w.writeheader()
        w.writerows(new_rows)

out_teams = []
plans = {}
for t in lg.teams:
    r = rosters[t.team_id]
    c0 = so_far[t.team_id]["starts"]
    total, rows, naive = plan_team(r, c0, detail=True)
    plans[t.team_id] = total + so_far[t.team_id]["pts"]
    tc = counters.get(t.team_id, {})
    non_ir = [p for p in r if not p["ir"]]
    moves = []
    for f in fa_players:
        if not any(p_play(f, d) > 0 for d in plan_days):
            continue
        drops = [None] if len(non_ir) < 15 else []
        prot = protected_ids(r)
        drops += [p for p in non_ir if p["espn_id"] not in prot]
        best_move = None
        for dr in drops:
            new = [p for p in r if p is not dr] + [f]
            wk = plan_team(new, c0) - total
            net = wk - future_cost(dr, f)
            if best_move is None or net > best_move[0]:
                best_move = (net, dr, wk)
        if best_move and best_move[0] >= MIN_NET_GAIN:
            g, dr, wk = best_move
            moves.append({"add": {"id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "status": f["status"],
                                  "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0]},
                          "drop": ({"id": dr["espn_id"], "name": dr["name"], "level": dr["level"], "asset_rank": dr["asset_rank"], "flag": drop_flag(dr)} if dr else None),
                          "gain": round(g, 1), "week_gain": round(wk, 1), "future_cost": round(wk - g, 1)})
    moves.sort(key=lambda x: -x["gain"])
    # greedy sequence: apply the best move, re-evaluate the rest against the new roster (moves interact: two adds can't fill the same idle slot)
    fa_by_id = {f["espn_id"]: f for f in fa_players}
    prot = protected_ids(r)
    seq, r2, cur, used = [], list(r), total, set()
    for step in range(4):
        best_step = None
        for mv in moves[:25]:
            f = fa_by_id[mv["add"]["id"]]
            if f["espn_id"] in used:
                continue
            nonir2 = [p for p in r2 if not p["ir"]]
            drops2 = ([None] if len(nonir2) < 15 else []) + [p for p in nonir2 if p["espn_id"] not in prot and p["espn_id"] not in used]
            for dr in drops2:
                wk = plan_team([p for p in r2 if p is not dr] + [f], c0) - cur
                g = wk - future_cost(dr, f)
                if best_step is None or g > best_step[0]:
                    best_step = (g, f, dr, wk)
        if not best_step or best_step[0] < MIN_NET_GAIN:
            break
        g, f, dr, wk = best_step
        r2 = [p for p in r2 if p is not dr] + [f]
        cur += wk
        used.add(f["espn_id"])
        seq.append({"add": {"id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"],
                            "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0]},
                    "drop": ({"id": dr["espn_id"], "name": dr["name"], "level": dr["level"], "asset_rank": dr["asset_rank"], "flag": drop_flag(dr)} if dr else None),
                    "gain": round(g, 1), "week_gain": round(wk, 1), "future_cost": round(wk - g, 1), "cum": round(cur - total, 1)})
    out_teams.append({"id": t.team_id, "abbrev": t.team_abbrev, "name": t.team_name.strip(), "opp": opp.get(t.team_id),
                      "starts_so_far": so_far[t.team_id]["starts"], "pts_so_far": round(so_far[t.team_id]["pts"], 1),
                      "adds_used": (tc.get("matchupAcquisitionTotals") or {}).get(str(mp_id), 0),
                      "expected": round(total, 1), "expected_total": round(total + so_far[t.team_id]["pts"], 1), "start_everyone": round(naive, 1),
                      "days": rows, "adds": moves[:10], "sequence": seq,
                      "roster": [{"id": p["espn_id"], "name": p["name"], "team": p["team"], "slots": p["slots"], "status": p["status"], "level": p["level"], "src": p["src"],
                                  "ir": p["ir"], "hub_id": p["hub_id"], "asset_rank": p["asset_rank"], "protected": p["espn_id"] in protected_ids(r), "model_level": p["model_level"], "espn_level": p["espn_level"], "has_props": bool(p.get("lvl_by_date")), "games": [d.isoformat() for d in plan_days if p_play(p, d) > 0 or (plays(p["team"], d))]}
                                 for p in r]})
    print(f"{t.team_abbrev:5s} expected {total:7.1f}  (start-everyone {naive:7.1f})  best add gain {moves[0]['gain'] if moves else 0}", flush=True)

for tm in out_teams:
    o = tm["opp"]
    if o in plans:
        diff = plans[tm["id"]] - plans[o]
        rem = len(plan_days) / n_days
        tm["opp_expected"] = round(plans[o], 1)
        tm["win_prob"] = round(erf_win(diff, WEEK_SD * math.sqrt(2) * math.sqrt(max(rem, 0.05))), 3)

out = {"generated": datetime.now(timezone.utc).isoformat(), "season": SEASON_ID, "my_abbrev": MY_ABBREV,
       "matchup": {"id": mp_id, "start": mp_start.isoformat(), "end": mp_end.isoformat(), "days": [d.isoformat() for d in days], "planned_days": [d.isoformat() for d in plan_days],
                   "cap": round(cap, 1), "adds_limit": adds_limit, "props": props_meta, "calendar_assumed": True,
                   "nba_games": {d.isoformat(): sorted(g.keys()) for d, g in games.items() if d in days}},
       "teams": out_teams}
OUTP = HUB / ("week_plan.json" if not _c0 else "week_plan_test.json")
OUTP.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote", OUTP.name, round(OUTP.stat().st_size / 1024), "KB")
