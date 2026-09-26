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
ASSET_HP = {p["id"]: p["asset_k"][5] for p in hub["players"] if p.get("asset_k") and len(p["asset_k"]) > 5}
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
# Measured (injury_status_playrate.py; NBA official 06:00 AM reports vs box scores, 2024-25 and 2025-26, n = 16,727 listed player-days):
# share of listed players who actually played.  Rotation = 20+ mpg over the last 15 games.
OFFICIAL_P = {"rotation": {"Available": 0.96, "Probable": 0.92, "Questionable": 0.57, "Doubtful": 0.04, "Out": 0.0},
              "bench": {"Available": 0.65, "Probable": 0.77, "Questionable": 0.37, "Doubtful": 0.03, "Out": 0.0}}
ROTATION_LEVEL = 24        # projected pts/g at which a player is treated as a rotation player (about 20 mpg)


def p_future(level):
    """chance a currently-healthy player suits up in a LATER scheduled game (measured, availability_by_level.py)"""
    return 0.61 if level < 15 else (0.81 if level < 20 else 0.85)


FA_ANCHOR, FA_SHRINK = 22.0, 0.6      # free agents are picked because they look good, so shrink their level toward replacement (backtest: predicted gain 93 -> 75 vs realized ~53)
try:
    import nba_injury_reports as NIR
    OFFICIAL, OFFICIAL_AT = NIR.latest_statuses()
    print(f"official NBA injury report: {len(OFFICIAL)} statuses from {OFFICIAL_AT}")
except Exception as ex:                       # never let a missing report break the plan
    OFFICIAL, OFFICIAL_AT = {}, None
    print("official injury report unavailable:", ex)
try:
    import availability_state as AV
    PRIOR = AV.prior_listings(today)
    print(f"availability: {len(PRIOR)} players listed in the previous 7 days' official reports")
except Exception as ex:                       # never let this break the plan: falls back to the flat rates in OFFICIAL_P
    AV, PRIOR = None, {}
    print("availability state unavailable:", ex)
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
            "status": p.injuryStatus or "ACTIVE", "level": round(base, 1), "src": src, "official": {},  "espn_level": espn_proj, "model_level": ours, "espn_avg": (p.stats.get(f"{SEASON_ID}_projected") or {}).get("avg"), "hub_id": hp["id"] if hp else None,
            "ir": on_roster_slot == "IR", "asset_rank": ASSET_RANK.get(hp["id"]) if hp else None, "kind": hp.get("kind") if hp else None}


def level_on(pl, d):
    """projected points per game on date d: sportsbook-prop-derived when we have it for that day, else the base level"""
    iso = d.isoformat()
    return pl.get("lvl_by_date", {}).get(iso, pl["level"] + pl.get("boost", {}).get(iso, 0.0))


def p_play(pl, d):
    if not plays(pl["team"], d):
        return 0.0
    if pl.get("from") and d < pl["from"]:         # add-timing what-ifs: he joins the roster on this date
        return 0.0
    if pl["ir"] or pl["status"] in ("OUT", "INJURY_RESERVE", "SUSPENSION"):
        return 0.0
    tier = "rotation" if pl["level"] >= ROTATION_LEVEL else "bench"
    off = pl.get("official", {}).get(d.isoformat())
    if off:                                       # the league's own designation for that game
        if AV is not None:                        # conditioned on whether he played his last game (5 seasons of reports): Questionable 81% / 42% / 45%
            v = AV.p_table(off, tier, pl.get("avail_state") if d == today else None)
            if v is not None:
                return v
        return OFFICIAL_P[tier].get(off, 0.9)
    if d == today:                                # status known this morning, no designation listed
        p = OFFICIAL_P[tier]["Questionable"] if pl["status"] == "DAY_TO_DAY" else 0.94
    else:                                         # later days: injuries/rest not yet known
        p = p_future(pl["level"])
    if pl["status"] == "ACTIVE" and b2b(pl["team"], d):
        p *= 0.89 / 0.94                          # second night of a back-to-back (measured)
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
        starters = [{"id": pl["espn_id"], "name": pl["name"], "slot": s, "ef": round(cands[order[k]][0], 1), "p": round(cands[order[k]][2], 2), "props": d.isoformat() in pl.get("lvl_by_date", {}),
                     "boost": round(pl.get("boost", {}).get(d.isoformat(), 0.0), 1)}
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

weekly_actual = {t.team_id: {} for t in lg.teams}      # completed matchups: real points per team (used to scale future-week projections)
for m in raw.get("schedule", []):
    if m.get("matchupPeriodId", 99) < mp_id and "away" in m and m.get("winner") not in (None, "UNDECIDED"):
        for side in ("home", "away"):
            tp = m[side].get("totalPoints")
            if tp is not None:
                weekly_actual[m[side]["teamId"]][str(m["matchupPeriodId"])] = round(tp, 1)
counters = {tm["id"]: tm.get("transactionCounter") or {} for tm in lg.espn_request.league_get(params={"view": "mTeam"}).get("teams", [])}
_c0 = float(os.environ.get("WEEK_C0", 0))     # testing only: pretend every team already has this many starts
for _k in so_far:
    so_far[_k]["starts"] += _c0
rosters = {}
for t in lg.teams:
    rosters[t.team_id] = [make_player(p, p.lineupSlot) for p in t.roster]

fa_players = [make_player(p) for p in lg.free_agents(size=150)]
for _p in fa_players:
    _p["level"] = round(FA_ANCHOR + FA_SHRINK * (_p["level"] - FA_ANCHOR), 1)
fa_players = [p for p in fa_players if p["level"] > 0 and p["team"] and any(plays(p["team"], d) for d in plan_days)]


# ---------- DYNASTY ADDS: free agents with long-term value (keeper-league asset value at 5 keepers), whether or not they help this week.
# Scans a wide free-agent list; remembers when each player first appeared so 'newly available' (dropped by a team) can be flagged.
def asset5(hp):
    return (hp["asset_k"][5] if hp and hp.get("asset_k") and len(hp["asset_k"]) > 5 else 0.0) or 0.0


DYN_MIN_ASSET = 6.0
dyn_pool = []
for _p in lg.free_agents(size=400):
    _hp = hub_player(_p.playerId, _p.name)
    _a = asset5(_hp)
    if _hp and _a >= DYN_MIN_ASSET:
        dyn_pool.append({"id": _p.playerId, "name": _p.name, "team": canon(_p.proTeam or ""), "pos": [x for x in _p.eligibleSlots if x in ("PG", "SG", "SF", "PF", "C")],
                         "status": _p.injuryStatus or "ACTIVE", "age": _hp.get("age"), "asset": round(_a, 1), "asset_rank": ASSET_RANK.get(_hp["id"]), "market_rank": _hp.get("market_rank"),
                         "kind": _hp.get("kind"), "p_break": _hp.get("p_break"), "level": round(_hp.get("year0_ppg") or 0, 1)})
dyn_pool.sort(key=lambda x: -x["asset"])
SEEN_PATH = HUB / "fa_seen.json"
_seen_old = json.load(open(SEEN_PATH, encoding="utf-8")) if SEEN_PATH.exists() else None
_now_ids = {str(x["id"]) for x in dyn_pool}
_seen_new = {k: v for k, v in (_seen_old or {}).items() if k in _now_ids}      # players who left the pool are forgotten; if they return they are 'new' again
for x in dyn_pool:
    k = str(x["id"])
    if k not in _seen_new:
        _seen_new[k] = today.isoformat()
    x["first_seen"] = _seen_new[k]
    x["new"] = bool(_seen_old is not None and (today - date.fromisoformat(_seen_new[k])).days <= 7)
SEEN_PATH.write_text(json.dumps(_seen_new), encoding="utf-8")
print(f"dynasty free agents: {len(dyn_pool)} with asset >= {DYN_MIN_ASSET}; top:", [(x['name'], x['asset']) for x in dyn_pool[:5]])


def week_games(pl):
    return sum(1 for d in plan_days if p_play(pl, d) > 0)


fa_players.sort(key=lambda p: -(p["level"] * week_games(p)))
fa_players = fa_players[:60]

_by_key = {}
for (_gd, _kk), _st in OFFICIAL.items():
    _by_key.setdefault(_kk, {})[_gd] = _st
for _pl in [x for rr in rosters.values() for x in rr] + fa_players:
    _pl["official"] = _by_key.get(norm(_pl["name"]).replace(" ", ""), {})
    _td = _pl["official"].get(today.isoformat())
    if AV is not None and _td in ("Questionable", "Probable", "Available", "Doubtful"):
        try:
            _pl["avail_state"] = AV.state_for(norm(_pl["name"]).replace(" ", ""), _pl["espn_id"], today, PRIOR)
        except Exception:
            _pl["avail_state"] = None

# ---------- USAGE FLOW: when a rotation player is out, his teammates pick up his production (usage_flow.py; 14 seasons of box scores, tested out of time on 2024-26).
# Adds a per-day boost to every teammate's level (rostered players and free agents alike), so lineups, adds, drops and timing all see it.
USAGE = {"enabled": False, "absent": 0, "boosted": 0}
OPPORTUNITIES = []
try:
    import usage_flow as UF
    all_pl = [x for rr in rosters.values() for x in rr] + fa_players
    kof = lambda name: norm(name).replace(" ", "")
    espn_by_key = {kof(pl["name"]): pl for pl in all_pl}
    # real NBA rosters from ESPN (the hub still lists retired players), matched by name to the hub's per-player projections
    nba_roster = {}
    try:
        _tj = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams", timeout=30).json()
        for _t in [x["team"] for x in _tj["sports"][0]["leagues"][0]["teams"]]:
            _rj = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{_t['id']}/roster", timeout=30).json()
            for _a in _rj.get("athletes", []):
                nba_roster[kof(_a["fullName"])] = canon(_t["abbreviation"])
    except Exception as _ex:
        print("ESPN NBA rosters unavailable, using the hub's teams:", _ex)
    team_players = {}
    for hp in hub["players"]:
        nsp = hp.get("next_season_proj") if hp.get("kind") == "current" else hp.get("rookie_proj")
        if not nsp or (nsp.get("MIN") or 0) < 8 or not hp.get("year0_ppg") or hp["year0_ppg"] < 6:
            continue
        key = kof(hp["player"])
        team = (espn_by_key.get(key) or {}).get("team") or nba_roster.get(key) or (canon(hp.get("team") or "") if not nba_roster else "")
        if not team or (nba_roster and key not in nba_roster and key not in espn_by_key):
            continue
        pos = UF.pos_probs(nsp["REB"] * 36, nsp["AST"] * 36, nsp["BLK"] * 36, nsp["STL"] * 36, nsp.get("FG3M", 0) * 36)
        team_players.setdefault(team, []).append({"id": key, "name": hp["player"], "fp": float(hp["year0_ppg"]), "mpg": float(nsp["MIN"]), "pos": pos})
    tp_by_key = {p["id"]: (tm, p) for tm, lst in team_players.items() for p in lst}
    listed_by_key = {}
    for (gd_, k_), st_ in OFFICIAL.items():
        listed_by_key.setdefault(k_, {})[gd_] = st_
    forced = {kof(n.strip()) for n in os.environ.get("USAGE_TEST_OUT", "").split(",") if n.strip()}       # testing only: pretend these players are OUT
    absent = []
    for k_ in set(listed_by_key) | {k for k, pl in espn_by_key.items() if pl["status"] in ("OUT", "INJURY_RESERVE") or pl["ir"]} | forced:
        if k_ not in tp_by_key:
            continue
        tm_, pp_ = tp_by_key[k_]
        if pp_["mpg"] < 12:
            continue
        lst = dict(listed_by_key.get(k_, {}))
        pl_ = espn_by_key.get(k_)
        today_st = lst.get(today.isoformat())
        espn_out = bool(pl_ and (pl_["status"] in ("OUT", "INJURY_RESERVE") or pl_["ir"]))
        if k_ in forced:
            for d_ in plan_days:
                lst[d_.isoformat()] = "Out"
            today_st = "Out"
        status0 = "Out" if (today_st == "Out" or (espn_out and today_st in (None, "Out"))) else (today_st if today_st in ("Doubtful", "Questionable") else "none")
        if status0 == "none" and not any(v in ("Out", "Doubtful", "Questionable") for v in lst.values()):
            continue
        streak = AV.out_streak(kof(pp_["name"]), today) if AV else 1
        if pl_ and pl_["ir"]:
            streak = max(streak, 15)
        elif espn_out and today_st is None:
            streak = max(streak, 8)       # ESPN says out and the report does not list him: a long-term absence
        absent.append({"id": k_, "team": tm_, "status0": status0, "streak": streak, "listed": lst, "p_today": 0.5 if today_st == "Questionable" else 0.0})
    tgn = lambda team, d: sum(1 for dd in plan_days if dd < d and plays(team, dd))
    boosts = UF.plan_boosts(plan_days, team_players, absent, plays, tgn)
    nb = 0
    for pl_ in all_pl:
        k_ = kof(pl_["name"])
        for d_ in plan_days:
            b_ = boosts.get((k_, d_.isoformat()))
            if b_ and pl_["level"] > 0:
                pl_.setdefault("boost", {})[d_.isoformat()] = round(b_["delta"], 1)
                pl_.setdefault("boost_why", set()).update(b_["because"])
                nb += 1
    owner_of = {kof(pp_["name"]): tm_ab for tm_ab, rr in [(t.team_abbrev, rosters[t.team_id]) for t in lg.teams] for pp_ in rr}
    for a_ in sorted(absent, key=lambda a: -tp_by_key[a["id"]][1]["fp"])[:8]:
        tm_, pp_ = tp_by_key[a_["id"]]
        if pp_["mpg"] < 22 or a_["status0"] == "none":
            continue
        ben = []
        for pl_ in all_pl:
            if pl_["team"] == tm_ and kof(pl_["name"]) != a_["id"] and pl_.get("boost"):
                mx_ = max(pl_["boost"].values())
                if mx_ >= 2.0 and pp_["name"] in pl_.get("boost_why", set()):
                    ben.append({"name": pl_["name"], "owner": owner_of.get(kof(pl_["name"])) or "FA", "delta": mx_, "level": pl_["level"], "espn_id": pl_["espn_id"]})
        ben.sort(key=lambda b: -b["delta"])
        OPPORTUNITIES.append({"absent": {"name": pp_["name"], "team": tm_, "status": a_["status0"], "streak": a_["streak"], "fp": round(pp_["fp"], 1)}, "beneficiaries": ben[:6]})
    USAGE = {"enabled": True, "absent": len(absent), "boosted": nb}
    print("usage flow:", USAGE)
except Exception as ex:                        # never let this break the plan
    print("usage flow unavailable:", repr(ex))

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

def search_moves(r, total, c0, steps=4, extra_protect=frozenset()):
    """best single adds and a greedy add/drop sequence for roster r (see the per-team notes in the module docstring)"""
    non_ir = [p for p in r if not p["ir"]]
    moves = []
    for f in fa_players:
        if not any(p_play(f, d) > 0 for d in plan_days):
            continue
        drops = [None] if len(non_ir) < 15 else []
        prot = protected_ids(r) | set(extra_protect)
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
                                  "boost": round(max(f.get("boost", {}).values(), default=0.0), 1), "boost_why": sorted(f.get("boost_why", [])),
                                  "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0]},
                          "drop": ({"id": dr["espn_id"], "name": dr["name"], "level": dr["level"], "asset_rank": dr["asset_rank"], "flag": drop_flag(dr)} if dr else None),
                          "gain": round(g, 1), "week_gain": round(wk, 1), "future_cost": round(wk - g, 1)})
    moves.sort(key=lambda x: -x["gain"])
    # greedy sequence: apply the best move, re-evaluate the rest against the new roster (moves interact: two adds can't fill the same idle slot)
    fa_by_id = {f["espn_id"]: f for f in fa_players}
    prot = protected_ids(r) | set(extra_protect)
    seq, r2, cur, used = [], list(r), total, set()
    for step in range(steps):
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
        # ADD TIMING for this step: its net gain if made on each remaining day (earlier steps assumed made already). Adds are capped per matchup and
        # unspent adds expire, so this shows what waiting costs
        base_r = [p for p in r2 if p is not dr]
        by_day = [{"date": d.isoformat(), "gain": round(plan_team(base_r + [dict(f, **{"from": d})], c0) - cur - future_cost(dr, f), 1)} for d in plan_days]
        r2 = [p for p in r2 if p is not dr] + [f]
        cur += wk
        used.add(f["espn_id"])
        seq.append({"add": {"id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"],
                            "boost": round(max(f.get("boost", {}).values(), default=0.0), 1), "boost_why": sorted(f.get("boost_why", [])),
                            "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0]},
                    "drop": ({"id": dr["espn_id"], "name": dr["name"], "level": dr["level"], "asset_rank": dr["asset_rank"], "flag": drop_flag(dr)} if dr else None),
                    "gain": round(g, 1), "week_gain": round(wk, 1), "future_cost": round(wk - g, 1), "cum": round(cur - total, 1), "by_day": by_day})
    return moves, seq


# ---------- INJURED-PLAYER ADVISOR (injury_advisor.py): how long will he be out, how does he come back, is he worth the roster spot
try:
    import injury_advisor as IA
    ESPN_INJ = IA.espn_injuries()
except Exception as _ex:
    IA, ESPN_INJ = None, {}
    print("injury advisor unavailable:", _ex)
TEAM_DATES = {}
_sp = HUB / "nba_schedule.json"
if _sp.exists():
    for _ds, _gl in json.load(open(_sp, encoding="utf-8"))["games"].items():
        for _g in _gl:
            for _t in _g[:2]:
                if _t != "TBD":
                    TEAM_DATES.setdefault(canon(_t), set()).add(_ds)
SEASON_END = max((max(v) for v in TEAM_DATES.values()), default=None)
PLAYOFF_START = next((a for i, a, b in BOUNDS if i == 20), None)
REPL_LEVEL = 24.0


def team_games(team, d0, d1):
    return sum(1 for ds in TEAM_DATES.get(team, ()) if d0 <= ds <= d1)


def advise_injuries(r, total, c0, key_of, ir_free):
    """one entry per injured player on the roster (out / IR / ESPN-listed out); the lists stay short"""
    if IA is None:
        return []
    out = []
    t0 = today.isoformat()
    pre = today < SEASON_START
    forced = {}
    for spec in os.environ.get("INJURY_TEST", "").split(";"):       # testing only: "Name|Type|Detail|Side|ReturnDate"
        f = [x.strip() for x in spec.split("|")]
        if len(f) == 5 and f[0]:
            forced[key_of(f[0])] = dict(status="Out", type=f[1], detail=f[2], side=f[3], return_date=f[4] or None, short="(test)", updated=None)
    for pl in r:
        info = forced.get(key_of(pl["name"])) or ESPN_INJ.get(int(pl["espn_id"]))
        long_out = pl["status"] in ("OUT", "INJURY_RESERVE") or pl["ir"] or (info and info.get("status") == "Out") or key_of(pl["name"]) in forced
        if not long_out:
            continue
        group, tier = IA.classify(info) if info else ("other", "moderate")
        streak = AV.out_streak(key_of(pl["name"]), today) if AV else 1
        if pre:
            streak = 1                                   # offseason: no games have been missed yet
        elif pl["ir"]:
            streak = max(streak, 8)
        curve, n_ref, cell = IA.out_curve(group, tier, streak)
        med, p80 = IA.median_games(curve), IA.quantile_games(curve, 0.8)
        team = pl["team"]
        g_season = team_games(team, t0, SEASON_END) if SEASON_END else 70
        g_play = team_games(team, t0, PLAYOFF_START.isoformat()) if PLAYOFF_START else 55
        g_espn = None
        if info and info.get("return_date"):
            rd = info["return_date"][:10]
            g_espn = team_games(team, t0, (date.fromisoformat(rd) - timedelta(days=1)).isoformat()) if rd > t0 else 0      # team games before the return date
            if rd >= (SEASON_END or "9999"):
                g_espn = g_season
        ours = med if med is not None else IA.KS[-1] + 10        # beyond the grid: 45+ games
        # in season take the LATER of our history and ESPN's date (team timelines run optimistic); in the offseason ESPN's date is the better guide
        plan_games = (g_espn if g_espn is not None else ours) if pre else (max(ours, g_espn) if g_espn is not None else ours)
        plan_games = min(plan_games, g_season)
        p_back = IA.p_back_within(curve, g_play)
        if g_espn is not None and info.get("return_date"):
            rd = info["return_date"][:10]
            if PLAYOFF_START and rd <= PLAYOFF_START.isoformat():
                p_back = max(p_back, 0.75)
            elif PLAYOFF_START and rd > PLAYOFF_START.isoformat():
                p_back = min(p_back, 0.10)
        if g_espn is not None and g_espn <= 0:
            p_back = 0.97                                     # ESPN's return date has already arrived
        hp = hub_by_id.get(pl["hub_id"]) if pl.get("hub_id") else None
        asset = ASSET_HP.get(pl["hub_id"], 0.0) if pl.get("hub_id") else 0.0
        after = max(0.0, g_season - plan_games)
        ramp = IA.ramp_summary(plan_games)
        above = max(0.0, pl["level"] * 0.88 - REPL_LEVEL)
        hold_value = above * after * 0.8 + asset * 73 * 0.10                # points from him after he returns (plays about 80% of games) plus a keeper-value term
        # cost of holding: a bench spot for the weeks he is out, unless an IR slot can take him
        cost = 0.0
        best_f = None
        if not pl["ir"]:
            rm = [q for q in r if q is not pl]
            for f in fa_players[:14]:
                g = plan_team(rm + [f], c0) - total
                if best_f is None or g > best_f[0]:
                    best_f = (g, f)
            weeks_out = min(plan_games / 3.3, 8.0)
            cost = 0.0 if ir_free > 0 and pl["status"] in IR_OK else max(0.0, best_f[0]) * weeks_out * 0.6 if best_f else 0.0
        season_end = plan_games >= g_season - 3
        if plan_games <= 1.5:
            verdict = "ACTIVATE (back within a game or two)" if pl["ir"] else "KEEP ACTIVE (back within a game or two)"
        elif pl["ir"]:
            verdict = "HOLD on IR" if hold_value > 0 else "HOLD (IR spot is free)"
        elif ir_free > 0 and pl["status"] in IR_OK:
            verdict = "MOVE TO IR, then hold"
        elif hold_value >= cost:
            verdict = "STASH" if plan_games <= 20 else "HOLD"
        else:
            verdict = "DROP"
        why = []
        if season_end:
            why.append("expected out for the rest of the season")
        elif pre and g_espn is not None:
            why.append(f"ESPN expects him back {info['return_date'][:10]} (about {g_espn} games missed)")
        else:
            why.append(f"about {plan_games:.0f} more games out (our history: median {('%.0f' % med) if med is not None else '45+'}" + (f"; ESPN: return {info['return_date'][:10]}" if g_espn is not None else "") + ")")
        why.append(f"about {hold_value:.0f} points above a replacement player once he is back" + (f"; the best free-agent swap now is worth about {best_f[0]:.0f} a week" if best_f and best_f[0] > 0 else ""))
        out.append({"id": pl["espn_id"], "name": pl["name"], "team": team, "level": pl["level"], "asset_rank": pl["asset_rank"], "on_ir": bool(pl["ir"]),
                    "injury": {"type": (info or {}).get("type"), "side": (info or {}).get("side"), "detail": (info or {}).get("detail"), "espn_return": (info or {}).get("return_date"),
                               "short": ((info or {}).get("short") or "")[:180], "group": group, "tier": tier},
                    "streak": streak, "median_games": None if med is None else round(med, 1), "p80_games": None if p80 is None else round(p80, 1), "espn_games": g_espn,
                    "plan_games": round(plan_games, 1), "p_back_playoffs": round(p_back, 2), "ramp": ramp, "hold_value": round(hold_value), "cost": round(cost),
                    "replacement": ({"name": best_f[1]["name"], "week_gain": round(best_f[0], 1)} if best_f and best_f[0] > 0 else None), "verdict": verdict, "why": why, "n_ref": n_ref})
    out.sort(key=lambda a: -a["level"])
    return out


IR_SLOTS = 4
IR_OK = ("OUT", "INJURY_RESERVE")     # ASSUMPTION: ESPN only lets players with an out/IR status into an IR slot (verify in season)
ROSTER_SPOTS = 15                     # 10 starters + 5 bench; IR slots are extra


def plan_ir(r):
    """Roster and IR housekeeping. Returns (roster used for planning, suggested moves).
    * a player who is OUT (or IR-status) sitting in a bench slot should move to an open IR slot: a free action that frees a roster spot
    * a player who is healthy (ACTIVE) but parked in an IR slot cannot play: activate him; if the roster is full, swap out the least useful
      non-protected player when the activated player is clearly better (3+ points a game)"""
    on_ir = [p for p in r if p["ir"]]
    free_ir = max(0, IR_SLOTS - len(on_ir))
    to_ir = sorted([p for p in r if not p["ir"] and p["status"] in IR_OK], key=lambda p: -p["level"])[:free_ir]
    ids_to_ir = {p["espn_id"] for p in to_ir}
    back = sorted([p for p in on_ir if p["status"] == "ACTIVE"], key=lambda p: -p["level"])
    plan = {p["espn_id"]: p for p in r}
    moves = []
    for p in to_ir:
        plan[p["espn_id"]] = dict(p, ir=True)
        moves.append({"id": p["espn_id"], "name": p["name"], "action": "to_ir", "status": p["status"], "level": p["level"]})
    prot = protected_ids(r)
    for p in back:
        spots = len([q for q in plan.values() if not q["ir"]])
        if spots < ROSTER_SPOTS:
            plan[p["espn_id"]] = dict(p, ir=False)
            moves.append({"id": p["espn_id"], "name": p["name"], "action": "activate", "status": p["status"], "level": p["level"]})
            continue
        cands = sorted([q for q in plan.values() if not q["ir"] and q["espn_id"] not in prot and q["espn_id"] != p["espn_id"]], key=lambda q: q["level"])
        if cands and p["level"] - cands[0]["level"] >= 3.0:
            d = cands[0]
            del plan[d["espn_id"]]
            plan[p["espn_id"]] = dict(p, ir=False)
            moves.append({"id": p["espn_id"], "name": p["name"], "action": "activate_swap", "status": p["status"], "level": p["level"],
                          "drop": {"id": d["espn_id"], "name": d["name"], "level": d["level"], "asset_rank": d["asset_rank"]}})
        else:
            moves.append({"id": p["espn_id"], "name": p["name"], "action": "activate_needs_spot", "status": p["status"], "level": p["level"]})
    return list(plan.values()), moves


out_teams = []
plans = {}
for t in lg.teams:
    r_orig = rosters[t.team_id]
    r, ir_moves = plan_ir(r_orig)
    c0 = so_far[t.team_id]["starts"]
    total, rows, naive = plan_team(r, c0, detail=True)
    plans[t.team_id] = total + so_far[t.team_id]["pts"]
    tc = counters.get(t.team_id, {})
    adds_left = max(0, adds_limit - (tc.get("matchupAcquisitionTotals") or {}).get(str(mp_id), 0))
    # NOTE: a suggested drop may be a player the IR advice activates. That is legal and can be right (activate him to free the IR slot, then drop him), so it is not blocked.
    moves, seq = search_moves(r, total, c0, steps=min(max(adds_left, 1), 8))
    _ir_free = max(0, 4 - len([p for p in r if p["ir"]]))
    advice = advise_injuries(r, total, c0, lambda n: norm(n).replace(" ", ""), _ir_free)
    if any(m["action"] in ("to_ir", "activate", "activate_swap") for m in ir_moves):
        total0 = plan_team(r_orig, c0)
        _m0, seq0 = search_moves(r_orig, total0, c0, steps=min(max(adds_left, 1), 8))
        unlock = (total + (seq[-1]["cum"] if seq else 0.0)) - (total0 + (seq0[-1]["cum"] if seq0 else 0.0))
        for m in ir_moves:
            m["unlocks"] = round(unlock, 1)
    # the lineup plan above assumes NO adds; also plan the week as if every suggested move were made, so the page can show both
    fa_map2 = {f["espn_id"]: f for f in fa_players}
    dropped = {m["drop"]["id"] for m in seq if m["drop"]}
    r_after = [p for p in r if p["espn_id"] not in dropped] + [fa_map2[m["add"]["id"]] for m in seq]
    total_after, rows_after, _n2 = plan_team(r_after, c0, detail=True) if seq else (total, rows, naive)
    out_teams.append({"id": t.team_id, "abbrev": t.team_abbrev, "name": t.team_name.strip(), "opp": opp.get(t.team_id), "weekly_actual": weekly_actual.get(t.team_id, {}),
                      "dynasty_adds": [dict(x, would_rank=1 + sum(1 for p in r if p["asset_rank"] is not None and (ASSET_HP.get(p["hub_id"]) or 0) > x["asset"])) for x in dyn_pool[:15]], "days_after": rows_after, "injury_advice": advice, "expected_after": round(total_after, 1),
                      "starts_so_far": so_far[t.team_id]["starts"], "pts_so_far": round(so_far[t.team_id]["pts"], 1),
                      "adds_used": (tc.get("matchupAcquisitionTotals") or {}).get(str(mp_id), 0),
                      "expected": round(total, 1), "expected_total": round(total + so_far[t.team_id]["pts"], 1), "start_everyone": round(naive, 1),
                      "days": rows, "adds": moves[:10], "sequence": seq, "ir_moves": ir_moves, "roster_spots": {"non_ir": len([p for p in r if not p["ir"]]), "of": ROSTER_SPOTS, "ir_used": len([p for p in r if p["ir"]]), "ir_of": IR_SLOTS},
                      "roster": [{"id": p["espn_id"], "name": p["name"], "team": p["team"], "slots": p["slots"], "status": p["status"], "level": p["level"], "src": p["src"],
                                  "ir": p["ir"], "hub_id": p["hub_id"], "official": p["official"].get(today.isoformat()), "avail_state": p.get("avail_state"), "boost": p.get("boost", {}), "boost_why": sorted(p.get("boost_why", [])), "asset_rank": p["asset_rank"], "protected": p["espn_id"] in protected_ids(r), "model_level": p["model_level"], "espn_level": p["espn_level"], "has_props": bool(p.get("lvl_by_date")), "games": [d.isoformat() for d in plan_days if p_play(p, d) > 0 or (plays(p["team"], d))]}
                                 for p in r]})
    print(f"{t.team_abbrev:5s} expected {total:7.1f}  (start-everyone {naive:7.1f})  best add gain {moves[0]['gain'] if moves else 0}", flush=True)

for tm in out_teams:
    o = tm["opp"]
    if o in plans:
        diff = plans[tm["id"]] - plans[o]
        rem = len(plan_days) / n_days
        tm["opp_expected"] = round(plans[o], 1)
        tm["win_prob"] = round(erf_win(diff, WEEK_SD * math.sqrt(2) * math.sqrt(max(rem, 0.05))), 3)

# ---- STASH POOL: free agents the this-week search cannot see -- out now (injured), or whose team has no game in the planning window.
# build_schedule_plan.py values each of them over the REST OF THE SEASON (when he is back, how many games, what he would replace) and merges them with the dynasty adds.
def build_stash():
    if IA is None:
        return []
    key_of = lambda n: norm(n).replace(" ", "")
    res = []
    t0 = today.isoformat()
    for _p in lg.free_agents(size=400):
        info = ESPN_INJ.get(int(_p.playerId))
        out_now = (_p.injuryStatus in IR_OK) or (info and info.get("status") == "Out")
        idle = canon(_p.proTeam or "") in TEAM_DATES and not any(plays(canon(_p.proTeam), d) for d in plan_days)
        if not (out_now or idle) or canon(_p.proTeam or "") not in TEAM_DATES:
            continue
        f = make_player(_p)
        f["level"] = round(FA_ANCHOR + FA_SHRINK * (f["level"] - FA_ANCHOR), 1)
        if f["level"] < 18 or not f["team"]:
            continue
        hp = hub_player(_p.playerId, _p.name)
        row = {"id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "status": f["status"], "out_now": bool(out_now),
               "age": hp.get("age") if hp else None, "asset": round(asset5(hp), 1), "asset_rank": f["asset_rank"], "market_rank": hp.get("market_rank") if hp else None, "kind": f.get("kind")}
        if out_now:
            group, tier = IA.classify(info) if info else ("other", "moderate")
            streak = 1 if today < SEASON_START else max(1, AV.out_streak(key_of(f["name"]), today) if AV else 1)
            curve, n_ref, cell = IA.out_curve(group, tier, streak)
            med = IA.median_games(curve)
            g_season = team_games(f["team"], t0, SEASON_END) if SEASON_END else 70
            g_espn = None
            if info and info.get("return_date"):
                rd = info["return_date"][:10]
                g_espn = team_games(f["team"], t0, (date.fromisoformat(rd) - timedelta(days=1)).isoformat()) if rd > t0 else 0
                if rd >= (SEASON_END or "9999"):
                    g_espn = g_season
            ours = med if med is not None else IA.KS[-1] + 10
            pre_ = today < SEASON_START
            pg = (g_espn if g_espn is not None else ours) if pre_ else (max(ours, g_espn) if g_espn is not None else ours)
            pg = min(pg, g_season)
            ds_ = sorted(d for d in TEAM_DATES.get(f["team"], ()) if d >= t0)
            i_ = int(round(pg))
            row.update({"games_out": round(pg, 1), "back_date": ds_[i_] if i_ < len(ds_) else None, "espn_return": (info or {}).get("return_date"),
                        "injury": " ".join(x for x in [((info or {}).get("side") or ""), ((info or {}).get("type") or (info or {}).get("detail") or "")] if x).strip() or None,
                        "p_back_playoffs": round(IA.p_back_within(curve, team_games(f["team"], t0, PLAYOFF_START.isoformat()) if PLAYOFF_START else 55), 2)})
        res.append(row)
    res.sort(key=lambda x: -x["level"])
    print(f"stash pool: {len(res)} free agents (out now or idle this window); top:", [(x['name'], x['level'], x.get('games_out')) for x in res[:6]])
    return res[:40]


try:
    STASH_POOL = build_stash()
except Exception as _ex:
    STASH_POOL = []
    print("stash pool failed:", _ex)


# ---- games to watch: for each team's matchup, the games (0-2 a day) whose players on either side matter most to the result
def key_games(tm, opp, wp):
    sch = json.load(open(HUB / "nba_schedule.json", encoding="utf-8")) if (HUB / "nba_schedule.json").exists() else {}
    tvmap = sch.get("tv", {})
    team_of = {p["id"]: p["team"] for t_ in (tm, opp) for p in t_["roster"]}
    lev = 1 - abs((wp if wp is not None else 0.5) - 0.5) * 0.8      # a close matchup is worth watching more than a blowout
    out = []
    mine = {r["date"]: r for r in (tm.get("days_after") or tm["days"])}
    theirs = {r["date"]: r for r in opp["days"]}
    for ds, r in mine.items():
        games_today = sch.get("games", {}).get(ds, [])
        cand = []
        for a, h, tip in games_today:
            a, h = canon(a), canon(h)
            m_ = [x for x in r["start"] if team_of.get(x["id"]) in (a, h)]
            o_ = [x for x in (theirs.get(ds) or {}).get("start", []) if team_of.get(x["id"]) in (a, h)]
            raw = math.sqrt(sum(x["ef"] ** 2 for x in m_ + o_))          # squares: two stars outweigh three role players
            if raw < 45 or not (m_ or o_):
                continue
            hh = int(tip[:2])
            dt = datetime.fromisoformat(ds).replace(hour=hh, minute=int(tip[3:5]), tzinfo=timezone.utc) + timedelta(days=1 if hh < 10 else 0)
            et = dt.astimezone(ZoneInfo("America/New_York"))
            cand.append({"date": ds, "away": a, "home": h, "tip": f"{(et.hour % 12) or 12}:{et.minute:02d} {'PM' if et.hour >= 12 else 'AM'} ET",
                         "tv": tvmap.get(ds, {}).get(a + "@" + h),
                         "score": round(raw * lev, 1), "mine": [[x["name"], round(x["ef"])] for x in sorted(m_, key=lambda z: -z["ef"])[:4]],
                         "theirs": [[x["name"], round(x["ef"])] for x in sorted(o_, key=lambda z: -z["ef"])[:4]]})
        cand.sort(key=lambda c: -c["score"])
        out += cand[:2]
    return out


_by_id = {t["id"]: t for t in out_teams}
for tm in out_teams:
    try:
        tm["key_games"] = key_games(tm, _by_id[tm["opp"]], tm.get("win_prob")) if tm["opp"] in _by_id else []
    except Exception as _ex:
        tm["key_games"] = []
        print("key games failed:", tm.get("abbrev"), _ex)

out = {"generated": datetime.now(timezone.utc).isoformat(), "season": SEASON_ID, "my_abbrev": MY_ABBREV,
       "matchup": {"id": mp_id, "start": mp_start.isoformat(), "end": mp_end.isoformat(), "days": [d.isoformat() for d in days], "planned_days": [d.isoformat() for d in plan_days],
                   "cap": round(cap, 1), "adds_limit": adds_limit, "props": props_meta, "calendar_assumed": True,
                   "nba_games": {d.isoformat(): sorted(g.keys()) for d, g in games.items() if d in days}},
       "fa_pool": [{"id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "status": f["status"], "boost": f.get("boost", {})} for f in fa_players],
       "usage": USAGE, "opportunities": OPPORTUNITIES, "stash_pool": STASH_POOL,
       "teams": out_teams}
OUTP = HUB / ("week_plan.json" if not _c0 else "week_plan_test.json")
OUTP.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote", OUTP.name, round(OUTP.stat().st_size / 1024), "KB")

# ---------- ACCOUNTABILITY LEDGER: what we predicted and recommended, appended every run from opening night (grade it against real results later; see ledger/README.md)
try:
    if (today >= SEASON_START - timedelta(days=1) or os.environ.get("LEDGER_FORCE")) and not _c0:
        LEDGER = HUB.parent / "ledger"
        LEDGER.mkdir(exist_ok=True)
        _lf = LEDGER / f"plan-{today.strftime('%Y-%m')}.jsonl"
        _hr = datetime.now(ET).hour
        _run = "overnight" if _hr < 10 else ("midday" if _hr < 17 else "evening")
        _me = next(t for t in out_teams if t["abbrev"] == MY_ABBREV)
        _mine = next(rosters[t.team_id] for t in lg.teams if t.team_abbrev == MY_ABBREV)
        _iso = today.isoformat()
        _recs = [{"kind": "plan", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": _run, "date": _iso, "matchup": mp_id, "team": MY_ABBREV,
                  "expected_total": _me["expected_total"], "expected_after_moves": _me.get("expected_after"), "win_prob": _me.get("win_prob"), "opp_expected": _me.get("opp_expected"),
                  "adds_used": _me["adds_used"], "adds_limit": adds_limit, "starts_so_far": _me["starts_so_far"], "cap": round(cap, 1),
                  "sequence": [{"add": m["add"]["name"], "add_id": m["add"]["id"], "drop": (m["drop"] or {}).get("name"), "gain": m["gain"], "week_gain": m["week_gain"], "boost": m["add"].get("boost", 0),
                                "by_day": [x["gain"] for x in m.get("by_day", [])]} for m in _me["sequence"]],
                  "ir_moves": [{"name": m["name"], "action": m["action"]} for m in _me["ir_moves"]],
                  "lineup": [{"id": x["id"], "name": x["name"], "slot": x["slot"], "ef": x["ef"], "p": x["p"], "boost": x.get("boost", 0)} for x in (_me["days"][0]["start"] if _me["days"] and _me["days"][0]["date"] == _iso else [])],
                  "players": [{"id": p["espn_id"], "name": p["name"], "status": p["status"], "official": p["official"].get(_iso), "state": p.get("avail_state"), "level": p["level"],
                               "boost": (p.get("boost") or {}).get(_iso, 0.0), "p_play": round(p_play(p, today), 3)} for p in _mine],
                  "fa_top": [{"id": f["espn_id"], "name": f["name"], "level": f["level"], "boost": (f.get("boost") or {}).get(_iso, 0.0), "p_play": round(p_play(f, today), 3)} for f in fa_players[:15]]}]
        if _me.get("injury_advice"):
            _recs.append({"kind": "injury", "ts": _recs[0]["ts"], "date": _iso, "items": [{"id": a["id"], "name": a["name"], "verdict": a["verdict"], "plan_games": a["plan_games"], "median": a["median_games"], "p80": a["p80_games"],
                                                                                          "espn_return": a["injury"].get("espn_return"), "p_back_playoffs": a["p_back_playoffs"], "streak": a["streak"]} for a in _me["injury_advice"]]})
        if OPPORTUNITIES:
            _recs.append({"kind": "usage", "ts": _recs[0]["ts"], "date": _iso, "items": [{"absent": o["absent"]["name"], "team": o["absent"]["team"], "status": o["absent"]["status"], "streak": o["absent"]["streak"],
                                                                                        "beneficiaries": [{"name": b["name"], "owner": b["owner"], "delta": round(b["delta"], 2)} for b in o["beneficiaries"]]} for o in OPPORTUNITIES]})
        with open(_lf, "a", encoding="utf-8") as fh:
            for r_ in _recs:
                fh.write(json.dumps(r_, ensure_ascii=False, separators=(",", ":")) + chr(10))
        print("ledger:", len(_recs), "records ->", _lf.name)
except Exception as _ex:
    print("ledger write failed:", _ex)
