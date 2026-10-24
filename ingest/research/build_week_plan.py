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
import random
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
from season import current_season_id
SEASON_ID = current_season_id()
import season_calendar as SCAL
_CAL = SCAL.load()                          # derived from the NBA schedule + ESPN's counts (season_calendar.py); refreshed below once ESPN is open
SEASON_START = SCAL.opener(_CAL)           # scoring period 1 (season opener)
MY_ABBREV = "DRNK"
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
CAP_PER_7 = 40.0
ADDS_PER_DAY = 8 / 7
WEEK_SD = 253.0                             # measured week-to-week SD of a team's weekly points (2025-26)
from team_abbr import canon
import late_swap as LS
ET = ZoneInfo("America/New_York")


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


# ---------- calendar (confirmed against ESPN's real scoreboard 2026-09-27: 6-day opener, 7-day weeks, one 14-day All-Star matchup #18 = Feb 15-28)
BOUNDS = SCAL.bounds(_CAL)                     # [(matchup id, first day, last day)] derived, no longer typed in (season_calendar.py); same values ESPN's scoreboard confirmed on 2026-09-27


def matchup_for(day):
    for i, a, b in BOUNDS:
        if a <= day <= b:
            return i, a, b
    return (1, *BOUNDS[0][1:]) if day < SEASON_START else BOUNDS[-1]


# ---------- data
lg = League(league_id=config.LEAGUE_ID, year=SEASON_ID, espn_s2=config.ESPN_S2, swid=config.SWID)
try:                                          # ESPN is open now: refresh/verify the calendar; if it changed, use the new one
    _newc = SCAL.refresh(lg)
    if _newc["periods"] != _CAL["periods"]:
        _CAL, SEASON_START, BOUNDS = _newc, SCAL.opener(_newc), SCAL.bounds(_newc)
    CAL_VERIFIED = SCAL.verified(_newc)
except Exception as _ex:
    print("season calendar refresh failed:", repr(_ex)[:100])
    CAL_VERIFIED = SCAL.verified(_CAL)
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
KEEPER_PROTECT = 6         # (see keeper_protect(): 5 keepers + a margin of 1, margin dropped for the last 3 weeks)
PROTECT_LEVEL = 35         # projects 35+ pts/g: never dropped, valued or not (catches stars our model can't value, e.g. after a long injury)
ROS_WEEKS = 6              # if the player you drop is better than the one you add, count that gap for this many future weeks (before you could re-stream)
GAMES_PER_WEEK = 3.3
RECLAIM_LOSE = float(os.environ.get("RECLAIM_LOSE", "0.15"))       # stream manager: a rostered player you drop for a rental can be re-added when the stream ends (he clears waivers in about 40 hours, then anyone can add him); the chance he is
                           # lost meanwhile (claimed by a team, or picked up as a free agent) is about 18% (waiver_intel.json: a good starter on waivers is claimed ~18% of the time) plus some free-agent exposure
CHAIN_STEP_MIN = float(os.environ.get("CHAIN_STEP_MIN", "5"))        # inside a stream chain (the slot's future cost is already paid) a further rental that only swaps one streamer for another is worth making from this many points: unspent adds are wasted
MIN_NET_GAIN = float(os.environ.get("MIN_NET_GAIN", "6"))         # a move must be worth at least this many points AFTER subtracting the future cost
SOLID_LEVEL = 28           # drops at/above this projection are flagged 'solid contributor'
NEVER_DROP = {norm(n) for n in json.load(open(Path(__file__).resolve().parent / "never_drop.json")).get("names", [])} if (Path(__file__).resolve().parent / "never_drop.json").exists() else set()


def protected_ids(roster):
    valued = sorted([p for p in roster if p["asset_rank"]], key=lambda p: p["asset_rank"])
    keep = {p["espn_id"] for p in valued[:keeper_protect()]}
    keep |= {p["espn_id"] for p in roster if (p["level"] >= PROTECT_LEVEL and weeks_after() > 0) or norm(p["name"]) in NEVER_DROP}     # championship week: a non-keeper has no value after it
    return keep


LAST_WEEK = _CAL["last_week"]      # last fantasy week (playoffs included), derived
KEEPERS_LATER = 5          # keepers per team in future years: the players who actually carry dynasty value


def weeks_after():
    """fantasy weeks left after the current matchup (playoffs included)"""
    return max(0, LAST_WEEK - mp_id)


def keeper_protect():
    """how many of your top dynasty assets are never suggested as drops: the 5 future keepers plus a margin of 1 (a keeper can get hurt, the 6th can pass the 5th) until the last 3 weeks,
    when the keeper list is settled and a player outside it is worth more as a contributor this week than as a hold"""
    return KEEPERS_LATER + (1 if weeks_after() >= 4 else 0)


def future_cost(dr, add):
    """points of production lost in later weeks by swapping dr for add (only if dr is the better player). Injury/short schedule this week don't matter."""
    if dr is None:
        return 0.0
    return max(0.0, dr["level"] - add["level"]) * GAMES_PER_WEEK * min(ROS_WEEKS, weeks_after())      # fewer future weeks to lose production in as the season ends (0 in the last week)


def drop_flag(p):
    return "solid" if p["level"] >= SOLID_LEVEL else ""


today = datetime.now(ET).date()
if os.environ.get("PLAN_TODAY"):                       # TEST HOOK: pretend today is this date (e.g. a Sunday) to exercise the lookahead; never set in production
    today = date.fromisoformat(os.environ["PLAN_TODAY"])
mp_id, mp_start, mp_end = matchup_for(max(today, SEASON_START))
days = [mp_start + timedelta(days=i) for i in range((mp_end - mp_start).days + 1)]
plan_days = [d for d in days if d >= today]
if not plan_days:
    plan_days = days
n_days = len(days)
cap = CAP_PER_7 * n_days / 7.0
adds_limit = round(ADDS_PER_DAY * n_days)
print(f"matchup {mp_id}: {mp_start}..{mp_end} ({n_days} days), cap {cap:.1f}, adds limit {adds_limit}; planning {len(plan_days)} days from {plan_days[0]}")

# SUNDAY LOOKAHEAD (2026-10-10). Adds are counted per matchup and unspent adds EXPIRE when it ends, so on the last two days of a matchup the stream manager may add players
# whose value is next matchup's first days (Mon-Wed). Those days are scored with their own fresh start count and cap; nothing changes mid-week.
# 2026-10-24 (Tommy): the whole next matchup, not just its first three days, so "he plays 4 games next week" counts in full. Adds made now come out of THIS week's
# allowance, which expires anyway, so setting up next week this way keeps next week's adds untouched.
LOOKAHEAD_DAYS = int(os.environ.get("LOOKAHEAD_DAYS", "7"))
LOOK_DAYS, NEXT_CAP = [], cap
if len(plan_days) <= 2 and plan_days[-1] == mp_end and mp_id < _CAL["last_week"]:
    _nx = next(((a_, b_) for i_, a_, b_ in BOUNDS if i_ == mp_id + 1), None)
    if _nx:
        LOOK_DAYS = [_nx[0] + timedelta(days=k) for k in range(min(LOOKAHEAD_DAYS, (_nx[1] - _nx[0]).days + 1))]
        NEXT_CAP = next(p_["cap"] for p_ in _CAL["periods"] if p_["id"] == mp_id + 1)
        print(f"lookahead on: also scoring next matchup's first days {[d.isoformat() for d in LOOK_DAYS]} (cap {NEXT_CAP})")

# NBA schedule for the matchup (+ the day before, for back-to-back detection)
games = {}     # date -> {team: iso tipoff}
opp_of = {}    # date -> {team: opponent abbrev} (for the matchup-context adjustment: who is he actually facing that day)
for d in [days[0] - timedelta(days=1)] + days + LOOK_DAYS:
    try:
        j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", params={"dates": d.strftime("%Y%m%d")}, timeout=30).json()
    except Exception:
        j = {}
    g = {}
    o = {}
    for ev in j.get("events", []):
        comp = ev["competitions"][0]["competitors"]
        abs_ = [canon(c["team"].get("abbreviation")) for c in comp if c["team"].get("abbreviation")]
        for c in comp:
            ab = c["team"].get("abbreviation")
            if ab:
                g[canon(ab)] = ev["date"]
        if len(abs_) == 2:
            o[abs_[0]], o[abs_[1]] = abs_[1], abs_[0]
    games[d] = g
    opp_of[d] = o


def plays(team, d):
    return canon(team) in games.get(d, {})


def opponent(team, d):
    return opp_of.get(d, {}).get(canon(team))


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


# ---- "why is he hot" (build_form_split.py, run by the daily local refresh): recent form split into minutes / shooting luck / volume / etc. with fitted persistence weights.
# Level adjustment: the split says how much of his recent change from baseline should last ("keep"); production level already credits w x the change, so add the difference (shrunk 30%, capped +-3).
FORM = {}
_fp = Path(os.environ["FORM_TEST_FILE"]) if os.environ.get("FORM_TEST_FILE") else HUB / "form_split.json"      # test: a form file built with FORM_AS_OF on last season
if _fp.exists():
    try:
        _fj = json.load(open(_fp, encoding="utf-8"))
        if os.environ.get("FORM_TEST_FILE") or (_fj.get("as_of") and (today - date.fromisoformat(_fj["as_of"])).days <= 4 and _fj.get("season", "")[:4] == str(SEASON_ID - 1)):
            FORM = _fj["players"]
    except Exception as _ex:
        print("form_split.json unreadable:", _ex)
FORM_ADJ_SHRINK, FORM_ADJ_CAP = 0.7, 3.0


def formkey(name):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(" jr.", "").replace(" jr", "").replace(" iii", "").replace(" ii", ""))

print("form split loaded for", len(FORM), "players")


def make_player(p, on_roster_slot=None):
    hp = hub_player(p.playerId, p.name)
    espn_proj = (p.stats.get(f"{SEASON_ID}_projected") or {}).get("applied_avg")
    ours = (hp.get("season_ppg") or hp.get("year0_ppg")) if hp else None      # DELCO season mode (rest-of-season settings); career mode stays in the keeper/dynasty numbers
    base = espn_proj if espn_proj else (ours or 0.0)          # ESPN's projection first; our model only when ESPN has none
    src = "ESPN" if espn_proj else ("model" if ours else "none")
    l15 = p.stats.get(f"{SEASON_ID}_last_15") or {}
    gp15 = (l15.get("total") or {}).get("GP") or 0
    if gp15 >= 3 and l15.get("applied_avg"):
        w = RECENT_FORM_MAX * min(gp15, 15) / 15.0
        base = w * l15["applied_avg"] + (1 - w) * base
        src += "+recent"
    form = FORM.get(formkey(p.name))
    level_pre = base
    if form and "keep" in form and base > 0:
        w_prod = RECENT_FORM_MAX * min(gp15, 15) / 15.0 if gp15 >= 3 and l15.get("applied_avg") else 0.0
        adj = max(-FORM_ADJ_CAP, min(FORM_ADJ_CAP, FORM_ADJ_SHRINK * (form["keep"] - w_prod * form["d"])))
        base = max(0.0, base + adj)
        form = dict(form, adj=round(adj, 1))
    return {"espn_id": p.playerId, "name": p.name, "team": canon(p.proTeam or ""), "slots": [s for s in p.eligibleSlots if s in set(SLOTS)], "form": form, "level_pre": round(level_pre, 1),
            "status": p.injuryStatus or "ACTIVE", "level": round(base, 1), "src": src, "official": {},  "espn_level": espn_proj, "model_level": ours, "espn_avg": (p.stats.get(f"{SEASON_ID}_projected") or {}).get("avg"), "hub_id": hp["id"] if hp else None, "hist_n": len((hp or {}).get("history_years") or []),
            "ir": on_roster_slot == "IR", "asset_rank": ASSET_RANK.get(hp["id"]) if hp else None, "kind": hp.get("kind") if hp else None,
            "model_career": hp.get("year0_ppg") if hp else None, "l15": round(l15["applied_avg"], 1) if gp15 >= 3 and l15.get("applied_avg") else None, "gp15": gp15}


def level_on(pl, d):
    """projected points per game on date d: sportsbook-prop-derived when we have it for that day, else the base level; discounted by the
    post-return ramp if he is still shaking off a recent long absence (ramp_by_date, set below)"""
    iso = d.isoformat()
    if iso in pl.get("lvl_by_date", {}):
        base = pl["lvl_by_date"][iso]
    elif pl.get("dfs_carry"):                       # a free agent the daily table has priced for a day already posted: his current role is the better guide for the days not posted yet
        base = DFSP.FA_CARRY * pl["dfs_carry"] + (1 - DFSP.FA_CARRY) * pl["level"] + pl.get("boost", {}).get(iso, 0.0)
    else:
        base = pl["level"] + pl.get("boost", {}).get(iso, 0.0)
    base += pl.get("matchup_by_date", {}).get(iso, 0.0)
    r = pl.get("ramp_by_date", {}).get(iso)
    return base * r if r is not None else base


def p_play(pl, d):
    p = _p_play0(pl, d) * pl.get("p_scale", 1.0)
    return p * DFSP_UNLISTED_P if d.isoformat() in pl.get("dfs_unlisted", ()) else p


DFSP_UNLISTED_P = 0.1      # see dfs_proj.UNLISTED_P


def _p_play0(pl, d):
    if not plays(pl["team"], d):
        return 0.0
    if pl.get("from") and d < pl["from"]:         # add-timing what-ifs: he joins the roster on this date
        return 0.0
    if pl.get("until") and d > pl["until"]:       # stream manager: he is dropped after this date (last day he is on the roster)
        return 0.0
    if pl["ir"] or pl["status"] in ("OUT", "INJURY_RESERVE", "SUSPENSION") or d.isoformat() in pl.get("dfs_out", ()):
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
    # WHO GETS A STARTING SLOT is decided by what a slot is worth with the late-swap safety net, not by level x chance of playing: before a game tips you can see who is playing and swap in a
    # bench player, so a 50%-to-play 50-point player is worth 0.5 x 50 + 0.5 x (the bench player who would take his slot), which beats a 90%-to-play 35-pointer. The cover value R is the
    # expected points of the best player left out of the 10 slots (0 when nobody is left: then nothing covers a scratch and the plain level x chance ordering applies).
    # Only the ORDER of selection changes; ef (level x chance) and the point totals are still accounted the old way, whose calibration was measured on that accounting.
    R = cands[len(SLOTS)][0] if len(cands) > len(SLOTS) else 0.0
    if R > 0:
        cands.sort(key=lambda x: -(x[0] + (1 - x[2]) * R))
    acc = slot_matcher(cands)
    vals, cnt, order = [0.0], [0], []
    for ci in acc:
        ef, pl, p = cands[ci]
        vals.append(vals[-1] + ef)
        cnt.append(cnt[-1] + p)
        order.append(ci)
    return cands, order, vals, cnt


def starts_with_cover(cands, n_draw=300):
    """expected starts for a day if you slot a bench player in for anyone who turns out to be out (what you actually do before tip-off). plan_team's own
    count, `cnt`, sums each chosen starter's chance of playing and gives bench players no credit, so it understates starts on days with spare players.
    DISPLAY ONLY: the points forecast and add/drop search still use the original accounting (their calibration was measured on it)."""
    if not cands:
        return 0.0
    rng = random.Random(len(cands) * 7919 + int(cands[0][0] * 10))
    tot = 0
    for _ in range(n_draw):
        avail = [c for c in cands if rng.random() < c[2]]
        tot += len(slot_matcher(avail))
    return tot / n_draw


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


def plan_team(roster, c0=0.0, detail=False, days_=None, cap_=None, lateswap=False):
    pdays = plan_days if days_ is None else days_       # days to score (default: the rest of this matchup) and the games capv that applies to them
    capv = cap if cap_ is None else cap_
    opts = [day_options(roster, d) for d in pdays]
    NS = int(capv) + 12
    NEG = -1e9
    # dp[day][c] = best expected points from this day on, given c starts so far (c integer, capped)
    memo = {}

    def best(di, c):
        if di == len(pdays):
            return 0.0, None
        key = (di, c)
        if key in memo:
            return memo[key]
        cands, order, vals, cnt = opts[di]
        if c >= capv:                       # locked: nothing counts for the rest of the matchup
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
    for di, d in enumerate(pdays):
        cands, order, vals, cnt = opts[di]
        locked = c >= capv
        m = 0 if locked else best(di, c)[1]
        chosen = [cands[order[k]][1] for k in range(m)]
        slots = assign_slots(chosen)
        ls_info = None
        if lateswap and chosen and not locked:
            try:
                tipof = lambda pl: games.get(d, {}).get(canon(pl["team"]), "z")
                st_ = [{"i": pl["espn_id"], "slots": [x for x in pl["slots"]], "p": cands[order[k]][2], "pts": cands[order[k]][0] / max(cands[order[k]][2], 1e-6), "tip": tipof(pl)} for k, pl in enumerate(chosen)]
                chosen_ids = {pl["espn_id"] for pl in chosen}
                bn_ = [{"i": c_[1]["espn_id"], "slots": [x for x in c_[1]["slots"]], "p": c_[2], "pts": c_[0] / max(c_[2], 1e-6), "tip": tipof(c_[1])} for c_ in cands if c_[1]["espn_id"] not in chosen_ids]
                new_, ls_info = LS.best_assignment(st_, bn_, {st["i"]: sl for st, sl in zip(st_, slots)})
                if ls_info and ls_info.get("changed"):
                    slots = [new_[pl["espn_id"]] for pl in chosen]
                if ls_info and any(x["p"] < 0.985 for x in st_):
                    nm_ = {b["i"]: next(c_[1]["name"] for c_ in cands if c_[1]["espn_id"] == b["i"]) for b in bn_}
                    cov_ = LS.cover_for(st_, bn_, {pl["espn_id"]: sl for pl, sl in zip(chosen, slots)})
                    ls_info["cover"] = {str(k_): [nm_[x] for x in v_] for k_, v_ in cov_.items()}
            except Exception as ex_:
                print("late-swap planning skipped:", repr(ex_)[:100])
        starters = [{"id": pl["espn_id"], "name": pl["name"], "slot": s, "ef": round(cands[order[k]][0], 1), "p": round(cands[order[k]][2], 2), "props": d.isoformat() in pl.get("lvl_by_date", {}) and d.isoformat() not in pl.get("dfs_dates", ()), "tonight": d.isoformat() in pl.get("dfs_dates", ()),
                     "boost": round(pl.get("boost", {}).get(d.isoformat(), 0.0), 1), "tip": games.get(d, {}).get(canon(pl["team"]))}
                    for k, (pl, s) in enumerate(zip(chosen, slots))]
        benched = [{"id": cands[order[k]][1]["espn_id"], "name": cands[order[k]][1]["name"], "ef": round(cands[order[k]][0], 1)} for k in range(m, len(order))]
        c_after = c + (cnt[m] if not locked else 0)
        rows.append({"date": d.isoformat(), "nba_teams": len(games.get(d, {})), "start": starters, "bench_playing": benched,
                     "pts": round(vals[m] if not locked else 0.0, 1), "starts_before": round(c, 1), "starts_after": round(c_after, 1),
                     "locked": bool(locked), "sat": (len(order) - m) if not locked else len(order), "late_swap": ls_info,
                     "starts_cover": round(starts_with_cover(cands), 1) if not locked else 0.0})
        # 'start everyone possible' comparison
        if naive_c < capv:
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

import waivers as WV
WM = WV.waiver_map(lg)        # players on waivers right now -> when the nightly run releases/awards them (waivers.py). Everyone else unrostered is an instant free agent.
print(f"{len(WM)} players on waivers right now: tagged, and only usable from the day the run clears them")


def _tag_waivers(players):
    """ESPN's free_agents() returns free agents AND waiver players. A waiver player can only be claimed (processed ~3am ET, at least 24h after the drop), so he joins the roster
    on his clear date: the planner's `from` mechanism then keeps him out of earlier days, and the UI shows a WAIVERS tag with the date."""
    for f in players:
        w = WM.get(f["espn_id"])
        if w:
            f["waiver"] = WV.export(w)
            f["from"] = w["clear"].date()
    return players


fa_players = _tag_waivers([make_player(p) for p in lg.free_agents(size=400)])   # matches stash_pool/dyn_pool's pull size (was 150: a real, valuable, healthy
# free agent could rank outside ESPN's global top-150-by-default-relevance in this specific league and so never surface here at all -- found
# 2026-09-27 via Jalen Green missing "This week" info on his player card entirely. The later sort-and-trim (see below) also had to widen for the
# same reason -- 400 alone wasn't enough once he was still getting cut at that later step.
for _p in fa_players:
    _p["is_fa"] = True
    _p["level_raw"] = _p["level"]                                             # true level, before the weekly-decision shrink below
    _p["level"] = round(FA_ANCHOR + FA_SHRINK * (_p["level"] - FA_ANCHOR), 1)  # shrink toward replacement for THIS WEEK's add decision only (see FA_SHRINK's comment) -- a
                                                                                # long-horizon evaluation (Long-term adds) should use level_raw instead; see build_schedule_plan.py
fa_players = [p for p in fa_players if p["level"] > 0 and p["team"] and any(plays(p["team"], d) for d in plan_days + LOOK_DAYS)]


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
                         "kind": _hp.get("kind"), "p_break": _hp.get("p_break"), "level": round(_hp.get("season_ppg") or _hp.get("year0_ppg") or 0, 1), "level_raw": round(_hp.get("season_ppg") or _hp.get("year0_ppg") or 0, 1), "form": FORM.get(formkey(_p.name))})
dyn_pool.sort(key=lambda x: -x["asset"])
for _x in dyn_pool:
    _x["waiver"] = WV.export(WM.get(_x["id"]))
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
    return sum(1 for d in plan_days + LOOK_DAYS if p_play(pl, d) > 0)


fa_players.sort(key=lambda p: -(p["level"] * week_games(p)))
fa_players = fa_players[:100]   # widened from 60 (2026-09-27): a real, well-known, healthy scorer (Jalen Green) ranked #87 here purely because ESPN's
# default free-agent relevance sort for this specific league buried him below other names -- he was getting cut before ever reaching fa_pool/"This
# week". search_moves()'s per-team simulation is the real cost driver here (O(fa_players x roster drops)), not this trim size itself; 100 stays cheap.

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

# ---------- POST-RETURN RAMP: a player who missed 5+ games gives less than his normal level for his first several games back
# (injury_return_model.json's "ramp", ~1,000 real returns). Applies to anyone currently ACTIVE who is still inside that window; a player
# still OUT right now is not touched here (his return is handled by the injury advisor separately, below).
try:
    import injury_advisor as IA_R
except Exception as _ex:
    IA_R = None
    print("post-return ramp unavailable (injury_advisor import failed):", _ex)
_ramp_team_dates = {}
try:
    for _ds, _gl in json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))["games"].items():
        if SEASON_START.isoformat() <= _ds <= today.isoformat():           # hard floor at this season's opener: a return ramp never reaches back into last season, however the schedule file is scoped
            for _g in _gl:
                for _t in _g[:2]:
                    if _t != "TBD":
                        _ramp_team_dates.setdefault(canon(_t), []).append(_ds)
    for _t in _ramp_team_dates:
        _ramp_team_dates[_t].sort()
except Exception as _ex:
    print("post-return ramp: schedule unavailable:", _ex)


def _ramp_streak(tdates, dates):
    """Scan backward through this SEASON's team games only. Returns (games_since_return, absence_len) for the most recent absence of 5+
    games, treating any shorter gap (a single rest day, etc.) as a blip that does not reset the streak -- only stops counting further back
    at a genuine 5+ game absence. (None, None) if he hasn't had one, or is out right now."""
    n = len(tdates)
    if n == 0 or dates.get(tdates[-1], 0) <= 0:            # out right now: handled by the injury advisor, not here
        return None, None
    i = n - 1
    while i >= 0:
        if dates.get(tdates[i], 0) > 0:
            i -= 1
            continue
        j = i
        while j >= 0 and dates.get(tdates[j], 0) <= 0:
            j -= 1
        gap = i - j
        if gap >= 5:
            since = sum(1 for k in range(j + 1, n) if dates.get(tdates[k], 0) > 0)
            return since, gap
        i = j                                              # short gap: see through it, keep scanning
    return None, None


_n_ramping = 0
if IA_R is not None and AV is not None:
    for _pl in [x for rr in rosters.values() for x in rr] + fa_players:
        if _pl["status"] != "ACTIVE" or _pl.get("ir"):
            continue
        _tdates = _ramp_team_dates.get(_pl["team"]) or []
        if len(_tdates) < 6:
            continue
        try:
            _dates = AV.game_dates(_pl["espn_id"])
        except Exception:
            _dates = None
        if not _dates:
            continue
        _since, _absent = _ramp_streak(_tdates, _dates)
        if _since is None or _since == 0 or _since > 10 or _absent < 5:
            continue
        _rb, _k = {}, _since
        for _d in plan_days:
            if plays(_pl["team"], _d):
                _k += 1
                _rr = IA_R.ramp_ratio(_absent, _k)
                if abs(_rr - 1.0) >= 0.02:
                    _rb[_d.isoformat()] = round(_rr, 3)
        if _rb:
            _pl["ramp_by_date"] = _rb
            _pl["ramp_note"] = f"back from a {_absent}-game absence, {_since} game{'s' if _since != 1 else ''} in — still ramping up"
            _n_ramping += 1
print(f"post-return ramp: {_n_ramping} player(s) currently ramping up from a long absence")

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
        team_players.setdefault(team, []).append({"id": key, "name": hp["player"], "fp": float(hp.get("season_ppg") or hp["year0_ppg"]), "mpg": float(nsp["MIN"]), "pos": pos})
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
    # confirmed tonight's starting lineups (pull_lineups.py -- RotoWire, local daily refresh only, same-day only, UNVALIDATED against a real
    # game as of 2026-09-30, see that file's docstring). Matched by re-normalizing ITS raw names through this file's own norm()/kof(), since
    # the two files' normalizers don't strip suffixes (Jr/Sr/III) the same way. Fails soft to "no data" like every other live feed here.
    will_start = {}
    try:
        lu = json.load(open(HUB / "lineups_today.json", encoding="utf-8"))
        if lu.get("has_data"):
            for team_players_lu in lu["teams"].values():
                for p_lu in team_players_lu.values():
                    k_lu = kof(p_lu["name"])
                    if k_lu in tp_by_key:
                        will_start[k_lu] = p_lu["starting"]
            print(f"lineup intel: {len(will_start)} of today's players matched to a confirmed start/bench status")
        else:
            print("lineup intel: no data today (real off day, preseason not covered, or fetch/parse failed -- see pull_lineups.py)")
    except Exception as _ex:
        print("lineup intel unavailable:", _ex)
    boosts = UF.plan_boosts(plan_days, team_players, absent, plays, tgn, will_start=will_start)
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

# ---------- MATCHUP: opponent defense (overall + by position), opponent missing production, opponent back-to-back, and this specific matchup's
# expected pace -- e.g. two fast teams meeting (matchup_context.py, matchup_model.json; RESEARCH_gamelevel.md -- a small, honestly-tested effect
# that rides on top of the level above, day by day, by real opponent).
MATCHUP = {"enabled": False, "n_teams": 0}
try:
    import matchup_context as MC
    team_matchup = json.load(open(HUB / "team_matchup.json", encoding="utf-8"))["teams"]
    all_pl_mu = [x for rr in rosters.values() for x in rr] + fa_players
    # opponent's missing rotation production right now, league-wide (reuses the usage-flow roster/absence lists when they built successfully)
    team_missing_now = {}
    try:
        for a_ in absent:
            p_out = 1.0 if a_["status0"] == "Out" else (0.5 if a_["status0"] in ("Doubtful", "Questionable") else a_.get("p_today", 0.0))
            tm_, pp_ = tp_by_key[a_["id"]]
            team_missing_now[tm_] = team_missing_now.get(tm_, 0.0) + p_out * pp_["fp"]
    except NameError:
        pass       # usage-flow didn't build (e.g. ESPN rosters unavailable that run): matchup still works, just without the "opponent missing" piece
    league_avg_missing = MC.LEAGUE_AVG["opp_missing_fp"]
    n_mu = 0
    for pl in all_pl_mu:
        hp = hub_by_id.get(pl.get("hub_id"))
        nsp = (hp.get("next_season_proj") if hp and hp.get("kind") == "current" else (hp.get("rookie_proj") if hp else None)) if hp else None
        pos = UF.pos_probs(nsp["REB"] * 36, nsp["AST"] * 36, nsp["BLK"] * 36, nsp["STL"] * 36, nsp.get("FG3M", 0) * 36) if nsp and (nsp.get("MIN") or 0) > 0 else (1 / 3, 1 / 3, 1 / 3)
        own_tm = team_matchup.get(pl["team"])
        mb = {}
        for d in plan_days:
            opp_team = opponent(pl["team"], d)
            if not opp_team:
                continue
            tm = team_matchup.get(opp_team)
            if not tm:
                continue
            posdef = pos[0] * tm["fpC"] + pos[1] * tm["fpF"] + pos[2] * tm["fpG"]
            missing = team_missing_now.get(opp_team, league_avg_missing)
            b2b_opp = bool(plays(opp_team, d) and plays(opp_team, d - timedelta(days=1)))
            exp_pace = MC.expected_pace(own_tm["pace"], tm["pace"]) if own_tm else None
            adj = MC.adjustment(drtg_opp=tm["drtg"], posdef_opp=posdef, missing_opp=missing, b2b_opp=b2b_opp, exp_pace=exp_pace)
            if abs(adj) >= 0.15:
                mb[d.isoformat()] = round(adj, 2)
        if mb:
            pl["matchup_by_date"] = mb
            n_mu += 1
    MATCHUP = {"enabled": True, "n_teams": len(team_matchup), "n_players": n_mu}
    print("matchup context:", MATCHUP)
except Exception as ex:
    print("matchup context unavailable:", repr(ex))

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

# ---------- DailyFantasyFuel per-day projections: the standard 'what will he do tonight' number. Overrides the per-day level (after the sportsbook props) for every date the table lists.
dfs_meta = {"enabled": False}
try:
    import dfs_proj as DFSP
    dfs_meta = DFSP.apply([x for rr in rosters.values() for x in rr] + fa_players, plan_days + LOOK_DAYS, fa_ids={f["espn_id"] for f in fa_players})
    print("dfs projections:", dfs_meta)
except Exception as ex:
    dfs_meta = {"enabled": False, "error": str(ex)[:120]}
    print("dfs projections unavailable:", repr(ex)[:120])

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
                         "espn": pl["espn_level"] or "", "model": pl["model_level"] or "", "props": pl.get("lvl_by_date", {}).get(today.isoformat(), ""),
                         # added 2026-10-24 for the projection-blend study (blend_study.py): career-mode DELCO, ESPN's last-15 average, and the blended level the plan used
                         "model_career": pl.get("model_career") or "", "l15": pl.get("l15") or "", "gp15": pl.get("gp15") or "", "level": pl.get("level") or "", "is_fa": int(bool(pl.get("is_fa")))})
    seen_log.add(key)
if new_rows:
    fields = list(new_rows[0].keys())
    if LOG.exists():
        with open(LOG, encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
        if old and set(fields) - set(old[0].keys()):          # the log gained columns: rewrite it once with the wider header (old rows get blanks)
            with open(LOG, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(old)
    fresh = not LOG.exists()
    with open(LOG, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
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
            moves.append({"add": {"form": f.get("form"), "ramp": f.get("ramp_note"), "id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "status": f["status"],
                                  "boost": round(max(f.get("boost", {}).values(), default=0.0), 1), "boost_why": sorted(f.get("boost_why", [])),
                                  "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0], "waiver": f.get("waiver")},
                          "drop": ({"form": dr.get("form"), "ramp": dr.get("ramp_note"), "id": dr["espn_id"], "name": dr["name"], "level": dr["level"], "asset_rank": dr["asset_rank"], "flag": drop_flag(dr)} if dr else None),
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
        by_day = [{"date": d.isoformat(), "gain": round(plan_team(base_r + [dict(f, **{"from": max(d, f["from"]) if f.get("from") else d})], c0) - cur - future_cost(dr, f), 1)} for d in plan_days]
        r2 = [p for p in r2 if p is not dr] + [f]
        cur += wk
        used.add(f["espn_id"])
        seq.append({"add": {"form": f.get("form"), "ramp": f.get("ramp_note"), "id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"],
                            "boost": round(max(f.get("boost", {}).values(), default=0.0), 1), "boost_why": sorted(f.get("boost_why", [])),
                            "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0], "waiver": f.get("waiver")},
                    "drop": ({"form": dr.get("form"), "ramp": dr.get("ramp_note"), "id": dr["espn_id"], "name": dr["name"], "level": dr["level"], "asset_rank": dr["asset_rank"], "flag": drop_flag(dr)} if dr else None),
                    "gain": round(g, 1), "week_gain": round(wk, 1), "future_cost": round(wk - g, 1), "cum": round(cur - total, 1), "by_day": by_day})
    return moves, seq


# ---------- STREAM MANAGER (built 2026-10-10): the week's adds planned DAY BY DAY as rentals, not one-way swaps.
# search_moves() above judges each add on its own and only ever adds (every add burns one of the few droppable players, and an add worse than the dropped player is charged six
# weeks of lost production, which blocked cheap rental streamers that fill the empty slot-days). stream_search() instead:
#   * considers every day a free agent plays as a possible ADD DAY and holds him through his last game that week (a rental window);
#   * lets a new add replace an EARLIER STREAMER (drop him the day the next one is needed) so one slot rotates through the week inside the add budget;
#   * checks the 15-man roster limit on every single day;
#   * charges a real contributor you drop only the gap to the free agent you could re-stream with (not six weeks of the add's own level);
#   * only uses true free agents any day, and waiver players only from the day they clear (their `from`, set in _tag_waivers).
# Greedy: each round picks the single best (player, add day, who to drop) by points gained this week minus that future cost, until nothing is worth MIN_NET_GAIN or the adds run out.
def long_level(p):
    # the LONG-RUN level used for 'what is he worth after this week': our model (DELCO) and ESPN averaged, since both see the whole season; a free agent is shrunk toward replacement, but only a little
    # when he has a long record and the two agree (a veteran at 34 on both is not a lucky streak), more when the record is short. This week's points still use the day-specific numbers.
    vals = [v for v in (p.get("espn_level"), p.get("model_level")) if v]
    base = sum(vals) / len(vals) if vals else p["level"]
    if not p.get("is_fa"):
        return base
    agree = len(vals) == 2 and abs(vals[0] - vals[1]) <= 6
    sh = 0.9 if (p.get("hist_n", 0) >= 3 and agree) else (0.75 if p.get("hist_n", 0) >= 3 else 0.6)
    return FA_ANCHOR + sh * (base - FA_ANCHOR)


# ---------- LEVERAGE (2026-10-24, Tommy). What a move is for is WINS, not points. A point this week is worth what it does to this matchup's win chance; a point in a
# later week is worth an average close week's rate. Both are put in the same currency by measuring this week's change in win chance against what the same points
# would do at the start of a coin-flip week, using the live tracker's win model (win_prob_model.json, calibrated on 2025-26):
#   win chance = Phi((margin so far + BETA x expected remaining margin) / sigma(days left))
#   points-equivalent of a this-week gain g = [Phi(x + BETA g / sigma) - Phi(x)] / (BETA x phi(0) / sigma_7)
# Early in a close week this is about g (1x); on a close last day about 2-3x; in a week that is out of reach either way, close to 0. Next week's points and the
# long-run cost of a drop stay at face value. LEV is filled in before the team loop (it needs both teams' projections); NO_LEVERAGE=1 turns it off.
WPM = json.load(open(Path(__file__).resolve().parent / "win_prob_model.json"))
WP_BETA = WPM["beta"]
WP_SIG = {int(k): v for k, v in WPM["sigma"].items()}
LEV = None


def wp_sigma(days_left):
    if days_left < 1:
        return WP_SIG[1] * math.sqrt(max(days_left, 0.25))
    lo = min(7, int(days_left))
    hi = min(7, lo + 1)
    return WP_SIG[lo] + (WP_SIG[hi] - WP_SIG[lo]) * (days_left - lo)


def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


THIS_WEEK_COUNTS = True     # False in a playoff week with no matchup for my team (a bye, or eliminated): this week's points count for nothing
NEXT_WEEK_P = 1.0           # chance my team plays NEXT matchup (playoff weeks: made-minus-bye before week 20, winning this round during the playoffs)


def this_week_value(g, shift=0.0):
    """points-equivalent of g more expected points THIS matchup, given where the matchup stands (shift = this-week points the plan has already added)"""
    if not THIS_WEEK_COUNTS:
        return 0.0
    if LEV is None or os.environ.get("NO_LEVERAGE") or abs(g) < 1e-9:
        return g
    sg = LEV["sigma"]
    x0 = (LEV["margin"] + WP_BETA * (LEV["exp_diff"] + shift)) / sg
    x1 = x0 + WP_BETA * g / sg
    return (_Phi(x1) - _Phi(x0)) / (WP_BETA * 0.3989422804 / WP_SIG[7])


def p_taken(ll, days_ahead):
    # chance another team takes an available player before you add him (2024-26: players dropped after 4+ recent games, taken within ~48 hours: 25% at 34+ ppg, 16% at 27-34, 12% below), compounding per 2 days.
    # Used ONLY to rank and time moves; whether a move makes the plan is decided on its full value, and the plan is rebuilt several times a day, so the risk of a move shrinks to zero on its own day.
    if days_ahead <= 0:
        return 0.0
    p48 = 0.25 if ll >= 33 else (0.16 if ll >= 27 else 0.12)
    return 1 - (1 - p48) ** (days_ahead / 2.0)


STREAM_HOLD_EVAL = {}      # candidate id -> best 'add now and KEEP him' option the stream search saw on its first round (read by the pickup alert so it agrees with the plan)


def stream_search(r, total, c0, steps=6, extra_protect=frozenset(), n_cand=35):
    global STREAM_HOLD_EVAL
    STREAM_HOLD_EVAL = {}
    prot = protected_ids(r) | set(extra_protect)
    look = LOOK_DAYS
    all_days = plan_days + look

    def val2(ros):                                                 # (this matchup's points, next matchup's points when the lookahead is on: own fresh start count and cap)
        return plan_team(ros, c0), (plan_team(ros, 0.0, days_=look, cap_=NEXT_CAP) if look else 0.0)

    def val(ros):
        a_, b_ = val2(ros)
        return a_ + b_
    cand = [f for f in fa_players if any(p_play(f, d) > 0 for d in all_days)][:n_cand]

    LL = long_level
    fl = sorted(LL(f) for f in fa_players if not f.get("waiver"))
    restream = fl[-3] if len(fl) >= 3 else REPL_LEVEL
    wk_left = min(ROS_WEEKS, weeks_after())

    def lose_p(dr, wait_days=2):
        """chance the player you drop is gone before you can re-add him: about 15% for the usual two days on waivers, more the longer he sits (same per-2-day rate as p_taken)"""
        return max(RECLAIM_LOSE, p_taken(LL(dr), max(2, wait_days)))

    def fut(dr, wait_days=2):
        # the gap to the free agent you could re-stream with, for the rest of the season, but only the share we expect to actually lose: you can re-add the player you drop once the stream is over
        return max(0.0, LL(dr) - restream) * GAMES_PER_WEEK * wk_left * lose_p(dr, wait_days) if dr is not None else 0.0

    def roster_ok(ros):
        for d in all_days:
            n = sum(1 for x in ros if not x["ir"] and not (x.get("from") and x["from"] > d) and not (x.get("until") and x["until"] < d))
            if n > ROSTER_SPOTS:
                return False
        return True

    def fut_hold(tgt, f):
        # keeping the add for the rest of the season instead of the player dropped: the level gap for the weeks that remain (a negative cost = he is better than the player dropped; credited at 60% because free-agent levels are noisy)
        gap = LL(tgt) - LL(f)
        return gap * GAMES_PER_WEEK * wk_left * (1.0 if gap > 0 else 0.6)

    def option(f, s_, e_):
        if e_ is None:                                              # a HOLD: he stays on the roster past this matchup
            return dict(f, **{"from": max(s_, f["from"]) if f.get("from") else s_, "until": None, "_stream": False, "_hold": True})
        return dict(f, **{"from": max(s_, f["from"]) if f.get("from") else s_, "until": e_, "_stream": True})

    def build(cur, fs, tgt, s_):
        new = [dict(x, until=s_ - timedelta(days=1)) if (tgt is not None and x is tgt) else x for x in cur]
        new.append(fs)
        if tgt is not None and not tgt.get("_stream") and not tgt.get("_rejoin") and not fs.get("_hold"):
            # a rostered player you drop can be claimed back: he clears waivers at the first 3am run at least 24 hours after the drop (drop in the morning of s_ -> back for s_ + 2).
            # You re-add him when the rental ends (that day at the earliest), and the chance he is still there falls the longer he sits (lose_p).
            back = max(s_ + timedelta(days=2), (fs["until"] + timedelta(days=1)) if fs.get("until") else s_ + timedelta(days=2))
            new.append(dict(tgt, **{"from": back, "until": None, "p_scale": 1.0 - lose_p(tgt, (back - s_).days), "_rejoin": True}))
        return new

    def run(forced):
        """one greedy pass. forced = a roster player whose slot is turned into a rotating stream slot: the FIRST add must drop him, his future cost (the gap to the free agent you could
        re-stream with, for the rest of the season) is charged ONCE for the whole chain, and later adds replace the earlier streamer for free. A single add that cannot beat his own future cost
        (+30 this week but -109 later) is correctly rejected by the one-at-a-time rule; a week of rotating streams through that slot can still be worth it, and this is how the plan sees it."""
        cur, used, infos = list(r), set(), []
        cur_this, cur_next = total, (plan_team(r, 0.0, days_=look, cap_=NEXT_CAP) if look else 0.0)
        shift = 0.0                                                 # this-week points already added by earlier steps (moves the matchup, so later steps see the new margin)
        for step in range(steps):
            first_forced = forced is not None and step == 0
            best = None
            for f in cand:
                if f["espn_id"] in used:
                    continue
                fd = [d for d in plan_days if p_play(f, d) > 0]
                fdl = [d for d in look if p_play(f, d) > 0]
                if not fd and not fdl:
                    continue
                e_last = (fdl or fd)[-1]                                # held through his last game in view (possibly next week's Mon-Wed)
                starts = list(fd) + ([plan_days[-1]] if fdl and plan_days[-1] not in fd else [])
                variants = {(s0, ee) for s0 in starts for ee in {e_last, next((d for d in (fd + fdl) if d >= s0), e_last)}}   # rent him to his last game, or release him after his first game (one-day rentals)
                variants |= {(s0, None) for s0 in starts}                                                                          # or KEEP him for the rest of the season (a pickup, not a rental)
                for s_, e_ in variants:
                    fs = option(f, s_, e_)
                    targets = [(None, "open")]
                    for x in cur:
                        if x["ir"]:
                            continue
                        if x.get("_stream"):
                            if x["from"] < s_ and (not x.get("until") or x["until"] >= s_):
                                targets.append((x, "streamer"))
                        elif x["espn_id"] not in prot and not x.get("until") and not x.get("_rejoin") and not x.get("_hold"):
                            targets.append((x, "roster"))
                    if e_ is None:
                        targets = [t_ for t_ in targets if t_[1] != "open"]
                    if first_forced:
                        targets = [(forced, "roster")]
                    for tgt, kind in targets:
                        new = build(cur, fs, tgt, s_)
                        if not roster_ok(new):
                            continue
                        th, nx = val2(new)
                        d_this, d_next = th - cur_this, nx - cur_next
                        wk = this_week_value(d_this, shift) + NEXT_WEEK_P * d_next          # this week in win terms, next week at face value x the chance I play it
                        wait = ((fs["until"] + timedelta(days=1)) - s_).days if fs.get("until") else 2
                        fc = (fut_hold(tgt, f) if e_ is None else fut(tgt, wait)) if kind == "roster" else 0.0
                        net = wk if first_forced else wk - fc
                        risk = p_taken(LL(f), (max(s_, f.get("from") or s_) - today).days)
                        adj = net - risk * max(0.0, wk)                # if he is gone by then this week's gain is lost (the plan would rebuild and pick a backup, so this is the pessimistic case)
                        if step == 0 and forced is None and e_ is not None:      # best RENTAL of each free agent against today's roster (the "other free agents" table)
                            h = STREAM_HOLD_EVAL.setdefault(f["espn_id"], {"name": f["name"], "net": -1e9, "waiver": f.get("waiver"), "now": None, "ll": round(LL(f), 1)})
                            if h.get("rent") is None or adj > h["rent"]["adj"]:
                                h["rent"] = {"net": round(float(net), 1), "adj": round(float(adj), 1), "wk": round(float(wk), 1), "fut": round(float(fc), 1), "drop": tgt["name"] if tgt else None,
                                             "on": s_.isoformat(), "until": e_.isoformat(), "risk": round(risk, 2)}
                        if step == 0 and forced is None and e_ is None and kind == "roster":
                            h = STREAM_HOLD_EVAL.setdefault(f["espn_id"], {"name": f["name"], "net": -1e9, "waiver": f.get("waiver"), "now": None, "ll": round(LL(f), 1)})
                            if net > h["net"]:
                                h.update({"net": round(float(net), 1), "adj": round(float(adj), 1), "wk": round(float(wk), 1), "fut": round(float(fc), 1), "drop": tgt["name"], "drop_level": tgt["level"], "on": s_.isoformat(), "team": f["team"]})
                            if s_ == plan_days[0] and (h["now"] is None or net > h["now"]["net"]):      # the same, but only 'add him today' (what the pickup alert is asking)
                                h["now"] = {"net": round(float(net), 1), "wk": round(float(wk), 1), "fut": round(float(fc), 1), "drop": tgt["name"], "drop_level": tgt["level"]}
                        if best is None or adj > best["adj"]:
                            best = {"net": net, "adj": adj, "risk": risk, "wk": wk, "f": f, "s": s_, "e": e_, "tgt": tgt, "kind": kind, "new": new, "fut": fc, "d_this": d_this, "d_next": d_next}
            if os.environ.get("STREAM_DEBUG"):
                print("  [stream]", ("chain via " + forced["name"] + " ") if forced is not None else "", "round best:", None if best is None else (best["f"]["name"], "add day", best["s"].isoformat(), "hold to", best["e"].isoformat() if best["e"] else "KEEP", "drop", (best["tgt"] or {}).get("name"), best["kind"], "week", round(best["wk"], 1), "future", round(best["fut"], 1), "net", round(best["net"], 1)))
            if best is None or best["net"] < (1.0 if first_forced else (CHAIN_STEP_MIN if (forced is not None and best["kind"] != "roster") else MIN_NET_GAIN)):
                break
            f, tgt = best["f"], best["tgt"]
            by_day = []
            for d in plan_days:                                         # what the same swap is worth if made on each other day
                fd2 = option(f, d, best["e"])
                try:
                    nn = build(cur, fd2, tgt if (tgt is None or tgt.get("_stream") is None or tgt["from"] < d) else None, d)
                    if roster_ok(nn):
                        th2, nx2 = val2(nn)
                        v = this_week_value(th2 - cur_this, shift) + NEXT_WEEK_P * (nx2 - cur_next) - best["fut"]
                    else:
                        v = -999.0
                except Exception:
                    v = -999.0
                if v > -900:
                    by_day.append({"date": d.isoformat(), "gain": round(float(v), 1)})
            infos.append({"f": f, "s": best["s"], "e": best["e"], "tgt_id": tgt["espn_id"] if tgt else None, "kind": best["kind"], "net": best["net"], "wk": best["wk"], "fut": best["fut"], "by_day": by_day, "risk": best["risk"],
                          "wk_next": best["d_next"], "this_raw": best["d_this"], "this_val": best["wk"] - best["d_next"]})
            cur, cur_this, cur_next = best["new"], cur_this + best["d_this"], cur_next + best["d_next"]
            shift += best["d_this"]
            used.add(f["espn_id"])
        return infos, cur

    infos, cur = run(None)
    best_total = sum(i["net"] for i in infos)
    # chains: turn the cheapest-to-lose roster player's slot into a stream slot for the week (a slot rotating one-day rentals), and keep it if the whole chain beats the best one-at-a-time plan
    pool = sorted([x for x in r if not x["ir"] and x["espn_id"] not in prot and not x.get("until") and not x.get("_stream")], key=fut)[:3]
    for T in pool:
        ci, cc = run(T)
        if len(ci) < 2:
            continue
        fT = fut(T)
        chain_total = sum(i["wk"] for i in ci) - fT
        if os.environ.get("STREAM_DEBUG"):
            print("  [stream] chain via", T["name"], "steps", len(ci), "weeks", round(sum(i["wk"] for i in ci), 1), "slot cost", round(fT, 1), "net", round(chain_total, 1), "vs best so far", round(best_total, 1))
        if chain_total >= MIN_NET_GAIN and chain_total > best_total:
            wsum = sum(i["wk"] for i in ci)
            for i in ci:                                                # share the one-time slot cost across the steps in proportion to what each earns, so every step shows a net gain and they add up to the chain's net
                i["net"] = i["wk"] * chain_total / wsum
                i["fut"] = i["wk"] - i["net"]
                i["chain"] = T["name"]
            infos, cur, best_total = ci, cc, chain_total
    # chronological order, with the running total recomputed in that order
    infos.sort(key=lambda i: i["s"])
    ros, seq = list(r), []
    for i in infos:
        f = i["f"]
        tgt = next((x for x in ros if x["espn_id"] == i["tgt_id"]), None) if i["tgt_id"] else None
        # BACKUPS: if this player is taken before you add him, who is next best for the same day and the same drop (judged against the roster as it stands at that step)
        backups = []
        try:
            base_t, base_n = val2(ros)
            shift_b = base_t - total
            used_ids = {x["f"]["espn_id"] for x in infos}
            for f2 in cand:
                if f2["espn_id"] in used_ids or not any(p_play(f2, d) > 0 for d in all_days):
                    continue
                new2 = build(ros, option(f2, i["s"], i["e"]), tgt, i["s"])
                if not roster_ok(new2):
                    continue
                t2, n2 = val2(new2)
                net2 = this_week_value(t2 - base_t, shift_b) + NEXT_WEEK_P * (n2 - base_n) - ((fut_hold(tgt, f2) if i["e"] is None else fut(tgt)) if (tgt is not None and not tgt.get("_stream")) else 0.0)
                backups.append({"name": f2["name"], "team": f2["team"], "net": round(float(net2), 1)})
            backups = sorted(backups, key=lambda b: -b["net"])[:3]
        except Exception:
            backups = []
        ros = build(ros, option(f, i["s"], i["e"]), tgt, i["s"])
        seq.append({"add": {"form": f.get("form"), "ramp": f.get("ramp_note"), "id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "status": f["status"],
                            "boost": round(max(f.get("boost", {}).values(), default=0.0), 1), "boost_why": sorted(f.get("boost_why", [])),
                            "ll": round(LL(f), 1), "games": [d.isoformat() for d in plan_days if d >= i["s"] and (i["e"] is None or d <= i["e"]) and p_play(f, d) > 0], "waiver": f.get("waiver")},
                    "drop": ({"form": tgt.get("form"), "ramp": tgt.get("ramp_note"), "id": tgt["espn_id"], "name": tgt["name"], "level": tgt["level"], "asset_rank": tgt["asset_rank"], "flag": drop_flag(tgt),
                              "streamer": bool(tgt.get("_stream"))} if tgt else None),
                    "gain": round(i["net"], 1), "week_gain": round(i["wk"] - i.get("wk_next", 0.0), 1), "next_gain": round(i.get("wk_next", 0.0), 1), "future_cost": round(i["fut"], 1),
                    "games_next": [d.isoformat() for d in look if i["s"] <= d and (i["e"] is None or d <= i["e"]) and p_play(f, d) > 0], "cum": round(plan_team(ros, c0) - total, 1), "by_day": i["by_day"],
                    "on": i["s"].isoformat(), "hold_through": i["e"].isoformat() if i["e"] else None, "keep": i["e"] is None, "backups": backups, "p_taken": round(i.get("risk", 0.0), 2),
                    "this_raw": round(i.get("this_raw", 0.0), 1), "this_val": round(i.get("this_val", 0.0), 1), "purpose": "next" if i.get("wk_next", 0.0) > max(0.0, i.get("this_val", 0.0)) else "this", "gain_adj": round(i["net"] - i.get("risk", 0.0) * max(0.0, i["wk"]), 1), "release": None, "chain": i.get("chain"), "reclaim": bool(tgt and not tgt.get("_stream") and i["e"] is not None)})
    for k, i in enumerate(infos):                                   # a streamer who is replaced later: say when he goes
        if i["tgt_id"]:
            for q in seq[:k]:
                if q["add"]["id"] == i["tgt_id"]:
                    q["release"] = i["s"].isoformat()
    return seq, ros


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
PLAYOFF_START = next((a for i, a, b in BOUNDS if i == _CAL["playoff_weeks"][0]), None)
REPL_LEVEL = 24.0


def team_games(team, d0, d1):
    return sum(1 for ds in TEAM_DATES.get(team, ()) if d0 <= ds <= d1)


# ---- long-term priority: dynasty value in season-point units, the same exchange rate as the Keepers tool (73 games x 0.25 weight on next year's top-5 asset total)
DYN_W = 0.25


def _a5th(roster):
    a = sorted((ASSET_HP.get(p["hub_id"]) or 0.0 for p in roster if p.get("hub_id")), reverse=True)
    return a[4] if len(a) >= 5 else 0.0


def dyn_pts(asset, a5):
    """points-equivalent of adding a player with this keeper asset: what he adds to your next-year top-5 asset total (only if he beats your 5th best) plus 10% of his asset for trade/bench option value"""
    return round(DYN_W * 73 * (max(0.0, asset - a5) + 0.1 * asset))


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


# OPPONENT HABITS (opponent_habits_study.py): an opponent's realistic rest-of-week points = the plan's perfect-lineup projection x his lineup efficiency (team_usage.json, from his cap use)
try:
    _TU_EFF = json.load(open(HUB / "team_usage.json", encoding="utf-8")).get("eff") or {}
except Exception:
    _TU_EFF = {}


def owner_eff(team_id, abbrev=None):
    if abbrev == MY_ABBREV:
        return 1.0
    return float((_TU_EFF.get(str(team_id)) or {}).get("e", 0.957))


# leverage inputs for MY matchup: margin so far, both teams' expected rest of the week (as the plan sees them now), and how much time is left
try:
    _me_t = next(t for t in lg.teams if t.team_abbrev == MY_ABBREV)
    _op_id = opp.get(_me_t.team_id)
    if _op_id is not None and today in plan_days:
        _rm, _ = plan_ir(rosters[_me_t.team_id])
        _ro, _ = plan_ir(rosters[_op_id])
        _em = plan_team(_rm, so_far[_me_t.team_id]["starts"])
        _eo = plan_team(_ro, so_far[_op_id]["starts"])
        _hr = datetime.now(ET).hour + datetime.now(ET).minute / 60.0
        _dl = len(plan_days) - (min(1.0, max(0.0, (_hr - 18.5) / 4.0)) if not os.environ.get("PLAN_TODAY") else 0.0)    # tonight's games under way: part of today is gone
        LEV = {"margin": so_far[_me_t.team_id]["pts"] - so_far[_op_id]["pts"], "exp_diff": _em - _eo * owner_eff(_op_id), "sigma": wp_sigma(_dl), "days_left": round(_dl, 2)}
        _wp0 = _Phi((LEV["margin"] + WP_BETA * LEV["exp_diff"]) / LEV["sigma"])
        print(f"leverage: margin so far {LEV['margin']:+.0f}, expected rest {LEV['exp_diff']:+.0f}, {_dl:.2f} days left, win chance {_wp0:.0%}; "
              f"+20 this week is worth {this_week_value(20.0):.1f} points-equivalent")
        LEV["wp"] = round(_wp0, 3)
except Exception as _ex:
    LEV = None
    print("leverage unavailable:", repr(_ex)[:120])
# PLAYOFFS (study #10, 2026-10-24): a playoff week with no matchup for me (bye / eliminated) counts for nothing; next week's points count only as much as the chance I play it
try:
    _po_weeks = set(_CAL.get("playoff_weeks") or [])
    _me_t2 = next(t for t in lg.teams if t.team_abbrev == MY_ABBREV)
    _odds = json.load(open(HUB / "playoff_odds.json", encoding="utf-8")) if (HUB / "playoff_odds.json").exists() else {}
    _mo = next((x for x in _odds.get("teams", []) if x.get("abbrev") == MY_ABBREV), {})
    if mp_id in _po_weeks and opp.get(_me_t2.team_id) is None:
        THIS_WEEK_COUNTS = False
    if LOOK_DAYS and (mp_id + 1) in _po_weeks:
        if mp_id not in _po_weeks:                                  # last regular-season week -> playoff round 1: seeds 1-2 sit out
            NEXT_WEEK_P = max(0.0, float(_mo.get("made", 1.0)) - float(_mo.get("bye", 0.0)))
        elif not THIS_WEEK_COUNTS:                                  # no matchup now: a bye (play next round) or eliminated
            NEXT_WEEK_P = 1.0 if float(_mo.get("bye", 0.0)) >= 0.5 else 0.0
        else:
            NEXT_WEEK_P = float((LEV or {}).get("wp", 0.5))         # must win this round to play the next
    if not THIS_WEEK_COUNTS or NEXT_WEEK_P < 1.0:
        print(f"playoffs: this week counts {THIS_WEEK_COUNTS}, chance of playing next week {NEXT_WEEK_P:.0%}")
except Exception as _ex:
    print("playoff context unavailable:", repr(_ex)[:100])

out_teams = []
plans = {}
for t in lg.teams:
    r_orig = rosters[t.team_id]
    r, ir_moves = plan_ir(r_orig)
    c0 = so_far[t.team_id]["starts"]
    total, rows, naive = plan_team(r, c0, detail=True, lateswap=(t.team_abbrev == MY_ABBREV))
    plans[t.team_id] = total * owner_eff(t.team_id, t.team_abbrev) + so_far[t.team_id]["pts"]      # realistic total: perfect-lineup projection x the owner's lineup habits
    tc = counters.get(t.team_id, {})
    adds_left = max(0, adds_limit - (tc.get("matchupAcquisitionTotals") or {}).get(str(mp_id), 0))
    # NOTE: a suggested drop may be a player the IR advice activates. That is legal and can be right (activate him to free the IR slot, then drop him), so it is not blocked.
    moves, seq = search_moves(r, total, c0, steps=min(max(adds_left, 1), 8))
    r_stream = None
    tonight_picks = None
    pickup_checks = None
    if t.team_abbrev == MY_ABBREV and today in plan_days:
        try:                                                  # single-day pickups for TONIGHT ranked by the day-specific projection: what each free agent adds to tonight's lineup (needs a drop; not netted)
            _base = plan_team(r, c0)
            _tp = []
            for f in fa_players:
                if f.get("waiver") or p_play(f, today) <= 0:
                    continue
                g1 = plan_team(r + [dict(f, **{"from": today, "until": today})], c0) - _base
                _tp.append({"id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "status": f["status"], "tonight": round(level_on(f, today), 1), "dfs": today.isoformat() in f.get("dfs_dates", ()),
                            "p": round(p_play(f, today), 2), "gain": round(g1, 1), "games": [d.isoformat() for d in plan_days if p_play(f, d) > 0]})
            tonight_picks = sorted(_tp, key=lambda x: -x["gain"])[:8]
        except Exception as _ex:
            print("tonight pickups skipped:", repr(_ex)[:100])
    if t.team_abbrev == MY_ABBREV:
        try:
            _rs, _tot = r, total
            if os.environ.get("STREAM_TEST_WEAKEN"):      # TEST HOOK: pretend these rostered players (comma-separated surnames) are 14 points worse, to exercise the chain logic
                _w = [x.strip().lower() for x in os.environ["STREAM_TEST_WEAKEN"].split(",")]
                _rs = [dict(x, level=round(x["level"] - 14, 1)) if any(k in x["name"].lower() for k in _w) else x for x in r]
                _tot = plan_team(_rs, c0)
            seq, r_stream = stream_search(_rs, _tot, c0, steps=min(max(adds_left, 1), 8))
            pickup_checks = {str(k): v for k, v in STREAM_HOLD_EVAL.items()}
        except Exception as _ex:
            print("stream manager failed, keeping the greedy sequence:", repr(_ex)[:160])
            r_stream = None
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
    r_after = r_stream if r_stream is not None else [p for p in r if p["espn_id"] not in dropped] + [fa_map2[m["add"]["id"]] for m in seq]
    total_after, rows_after, _n2 = plan_team(r_after, c0, detail=True, lateswap=(t.team_abbrev == MY_ABBREV)) if seq else (total, rows, naive)
    out_teams.append({"id": t.team_id, "abbrev": t.team_abbrev, "name": t.team_name.strip(), "tonight_picks": tonight_picks, "pickup_checks": pickup_checks if t.team_abbrev == MY_ABBREV else None, "opp": opp.get(t.team_id), "weekly_actual": weekly_actual.get(t.team_id, {}),
                      "asset_5th": round(_a5th(r), 1), "dynasty_adds": [dict(x, would_rank=1 + sum(1 for p in r if p["asset_rank"] is not None and (ASSET_HP.get(p["hub_id"]) or 0) > x["asset"]), dyn_pts=dyn_pts(x["asset"], _a5th(r))) for x in dyn_pool[:15]], "days_after": rows_after, "injury_advice": advice, "expected_after": round(total_after, 1),
                      "starts_so_far": so_far[t.team_id]["starts"], "pts_so_far": round(so_far[t.team_id]["pts"], 1),
                      "adds_used": (tc.get("matchupAcquisitionTotals") or {}).get(str(mp_id), 0),
                      "expected": round(total, 1), "expected_total": round(total + so_far[t.team_id]["pts"], 1), "start_everyone": round(naive, 1),
                      "days": rows, "adds": moves[:10], "sequence": seq, "ir_moves": ir_moves, "roster_spots": {"non_ir": len([p for p in r if not p["ir"]]), "of": ROSTER_SPOTS, "ir_used": len([p for p in r if p["ir"]]), "ir_of": IR_SLOTS},
                      "roster": [{"id": p["espn_id"], "name": p["name"], "team": p["team"], "slots": p["slots"], "status": p["status"], "level": p["level"], "src": p["src"],
                                  "ir": p["ir"], "hub_id": p["hub_id"], "tonight": round(level_on(p, today), 1), "official": p["official"].get(today.isoformat()), "avail_state": p.get("avail_state"), "boost": p.get("boost", {}), "boost_why": sorted(p.get("boost_why", [])), "asset_rank": p["asset_rank"], "protected": p["espn_id"] in protected_ids(r), "model_level": p["model_level"], "espn_level": p["espn_level"], "ramp": p.get("ramp_note"), "matchup": p.get("matchup_by_date", {}).get(today.isoformat()), "form": p.get("form"), "level_pre": p.get("level_pre"), "has_props": bool(p.get("lvl_by_date")), "games": [d.isoformat() for d in plan_days if p_play(p, d) > 0 or (plays(p["team"], d))]}
                                 for p in r]})
    print(f"{t.team_abbrev:5s} expected {total:7.1f}  (start-everyone {naive:7.1f})  best add gain {moves[0]['gain'] if moves else 0}", flush=True)

for tm in out_teams:
    o = tm["opp"]
    if o in plans:
        diff = plans[tm["id"]] - plans[o]
        rem = len(plan_days) / n_days
        tm["opp_expected"] = round(plans[o], 1)
        tm["eff"] = round(owner_eff(tm["id"], tm["abbrev"]), 3)
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
        f["is_fa"] = True
        f["level_raw"] = f["level"]                                            # true level, before the weekly-decision shrink -- see build_week_plan.py's fa_players comment
        f["level"] = round(FA_ANCHOR + FA_SHRINK * (f["level"] - FA_ANCHOR), 1)
        if f["level"] < 18 or not f["team"]:
            continue
        hp = hub_player(_p.playerId, _p.name)
        row = {"form": f.get("form"), "ramp": f.get("ramp_note"), "id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "level_raw": f["level_raw"], "ll": round(long_level(f), 1), "status": f["status"], "out_now": bool(out_now),
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
        row["waiver"] = WV.export(WM.get(f["espn_id"]))      # on waivers: a claim, not an instant add
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
                   "cap": round(cap, 1), "adds_limit": adds_limit, "props": props_meta, "dfs": dfs_meta, "calendar_assumed": not CAL_VERIFIED, "look_days": [d.isoformat() for d in LOOK_DAYS],
           "leverage": ({"x20": round(this_week_value(20.0) / 20.0, 2), "wp": LEV.get("wp"), "days_left": LEV.get("days_left")} if LEV else None),
                   "nba_games": {d.isoformat(): sorted(g.keys()) for d, g in games.items() if d in days}},
       "fa_pool": [{"form": f.get("form"), "ramp": f.get("ramp_note"), "id": f["espn_id"], "name": f["name"], "team": f["team"], "slots": f["slots"], "level": f["level"], "tonight": round(level_on(f, today), 1), "level_raw": f.get("level_raw", f["level"]), "ll": round(long_level(f), 1), "status": f["status"], "waiver": f.get("waiver"), "boost": f.get("boost", {}),
                     "age": (hub_by_id.get(f["hub_id"]) or {}).get("age"), "asset": round(asset5(hub_by_id.get(f["hub_id"])), 1), "asset_rank": ASSET_RANK.get(f["hub_id"]), "market_rank": (hub_by_id.get(f["hub_id"]) or {}).get("market_rank"), "kind": (hub_by_id.get(f["hub_id"]) or {}).get("kind")} for f in fa_players],
       "usage": USAGE, "opportunities": OPPORTUNITIES, "stash_pool": STASH_POOL,
       "teams": out_teams}
# form records are exported once (out["form"], keyed by normalised name); every player dict carries just the key
_FORM_OUT = {}


def _slim(o):
    if isinstance(o, dict):
        for k_, v_ in list(o.items()):
            if k_ == "form" and isinstance(v_, dict):
                key_ = formkey(o.get("name") or "")
                _FORM_OUT[key_] = {kk: vv for kk, vv in v_.items() if kk not in ("team", "n")}
                o[k_] = key_
            else:
                _slim(v_)
    elif isinstance(o, list):
        for v_ in o:
            _slim(v_)


_slim(out)
out["form"] = _FORM_OUT
OUTP = HUB / ("week_plan.json" if not _c0 else "week_plan_test.json")
OUTP.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote", OUTP.name, round(OUTP.stat().st_size / 1024), "KB")

# ---------- ACCOUNTABILITY LEDGER: what we predicted and recommended, appended every run from opening night (grade it against real results later; see ledger/README.md)
try:
    if (today >= SEASON_START - timedelta(days=1) or os.environ.get("LEDGER_FORCE")) and not _c0 and not os.environ.get("PLAN_TODAY"):      # a simulated day (PLAN_TODAY) must never write the real ledger
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
                  "players": [{"id": p["espn_id"], "name": p["name"], "status": p["status"], "official": p["official"].get(_iso), "state": p.get("avail_state"), "level": p["level"], "level_pre": p.get("level_pre"), "form_d": (p.get("form") or {}).get("d"), "form_keep": (p.get("form") or {}).get("keep"),
                               "boost": (p.get("boost") or {}).get(_iso, 0.0), "p_play": round(p_play(p, today), 3), "tonight": round(level_on(p, today), 1), "dfs": _iso in p.get("dfs_dates", ())} for p in _mine],
                  "fa_top": [{"id": f["espn_id"], "name": f["name"], "level": f["level"], "level_pre": f.get("level_pre"), "form_d": (f.get("form") or {}).get("d"), "form_keep": (f.get("form") or {}).get("keep"), "boost": (f.get("boost") or {}).get(_iso, 0.0), "p_play": round(p_play(f, today), 3), "tonight": round(level_on(f, today), 1)} for f in fa_players[:15]]}]
        _recs.append({"kind": "league", "ts": _recs[0]["ts"], "date": _iso, "matchup": mp_id, "run": _run,
                      "teams": [{"abbrev": t["abbrev"], "id": t["id"], "opp": t.get("opp"), "pts_so_far": t["pts_so_far"], "expected": t["expected"], "expected_total": t["expected_total"],
                                 "eff": t.get("eff"), "win_prob": t.get("win_prob"), "starts_so_far": t["starts_so_far"], "adds_used": t["adds_used"]} for t in out_teams]})
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
