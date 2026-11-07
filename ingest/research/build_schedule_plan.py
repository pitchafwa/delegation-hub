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
import os
if os.environ.get("SP_HUB"):                       # replaying an old snapshot (rebuild_odds_history.py): read and write that folder instead
    HUB = Path(os.environ["SP_HUB"])
import season_calendar as SCAL
_CAL = SCAL.load()
SEASON_START = SCAL.opener(_CAL)
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
CAP_PER_7 = 40.0
PLAYOFF_WEEKS = tuple(_CAL["playoff_weeks"])
from team_abbr import canon
OUT_STATUS = ("OUT", "INJURY_RESERVE", "SUSPENSION")
RETURN_FACTOR = 0.6      # FALLBACK ONLY (no injury details in week_plan.json): a player who is OUT now is back for weeks 3+ at 60% of the normal play rate
NEAR_WEEKS = 2           # ...and counts as absent for this many upcoming weeks
RAMP_BACK = 0.9          # first weeks back below his normal level (injury_return_model.json ramp)
try:
    import injury_advisor as IA   # measured return curves (title_odds_study.py, 2026-11-07: better than the flat rule above at telling teams apart)
except Exception as _ex:
    IA = None
    print("injury return curves unavailable, using the flat rule:", _ex)
FA_ANCHOR, FA_SHRINK = 22.0, 0.6   # (fa_pool levels in week_plan.json are already shrunk)

CAL = [{"id": p["id"], "start": date.fromisoformat(p["start"]), "end": date.fromisoformat(p["end"]), "days": p["days"], "cap": p["cap"], "playoff": p["playoff"]} for p in _CAL["periods"]]
SPECIAL_WEEKS = {w["id"] for w in CAL if w["days"] != 7}      # the short opener and the 14-day All-Star matchup: not comparable to a normal 7-day week

sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))
GAMES = {}    # date -> set of teams playing
for ds, gl in sched["games"].items():
    GAMES[date.fromisoformat(ds)] = {canon(t) for g in gl for t in g[:2] if t != "TBD"}
wp = json.load(open(HUB / "week_plan.json", encoding="utf-8"))
from zoneinfo import ZoneInfo
today = date.fromisoformat(os.environ["SP_TODAY"]) if os.environ.get("SP_TODAY") else datetime.now(ZoneInfo("America/New_York")).date()
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


_tg_cache = {}


def team_games_upto(team, d):
    """his team's games from today through d"""
    k = (team, d)
    if k not in _tg_cache:
        _tg_cache[k] = sum(1 for i in range((d - today).days + 1) if plays(team, today + timedelta(days=i)))
    return _tg_cache[k]


# LATER-WEEK DECAY (horizon_decay_study.py, 2026-11-07): a player healthy today plays fewer of his games, and scores a little less, the further out the week:
# 97% of this week's rate 1 week ahead, 93% at 4, 90% at 8, 85% at 16+ (box scores 2010-24; about the same at every level). Without it, forecasts for later
# weeks ran 75-183 points a week high on the 2025-26 replay; with it, 10-28. The current week is not touched (the in-season factor calibrates it).
_DEC = json.load(open(Path(__file__).resolve().parent / "horizon_decay.json"))
_DEC_FROM = None


def decay(d, level):
    global _DEC_FROM
    if _DEC_FROM is None:
        _DEC_FROM = max(today, SEASON_START)
        _DEC_FROM = next((w["start"] for w in CAL if w["start"] <= _DEC_FROM <= w["end"]), _DEC_FROM)     # weeks ahead counted from this matchup's start
    h = min(max(0, (d - _DEC_FROM).days // 7), _DEC["hmax"])
    if h == 0:
        return 1.0
    b = next((i for i, (lo, hi) in enumerate(_DEC["bands"]) if lo <= level < hi), 0)
    return _DEC["f"][h][b]


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
    elif pl.get("inj") and IA is not None:
        # out now: ESPN's return date when it gives one, else the chance he is back by this game from the measured return curve (injury type, games missed so far)
        j = pl["inj"]
        if j.get("rd"):
            if d.isoformat() < j["rd"]:
                return 0.0
            p *= RAMP_BACK
        else:
            curve = IA.out_curve(j["g"], j["t"], j["s"])[0]
            p *= IA.p_back_within(curve, team_games_upto(pl["team"], d) - 0.5) * RAMP_BACK
    elif pl["ir"] or pl["status"] in OUT_STATUS:
        if wid < FIRST + NEAR_WEEKS:
            return 0.0
        p *= RETURN_FACTOR
    elif b2b(pl["team"], d):
        p *= 0.89 / 0.94
    return p * decay(d, pl["level"])


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
            "status": p.get("status", "ACTIVE"), "level": p.get("level") or 0.0, "ir": bool(p.get("ir")), "asset_rank": p.get("asset_rank"), "protected": bool(p.get("protected")),
            "inj": p.get("inj")}


def games_row(pl):
    return [heat[pl["team"]][i]["g"] if pl["team"] in heat else 0 for i in range(len(CAL))]


NEXT4 = [w["id"] for w in CAL if FIRST <= w["id"] < FIRST + 4]
PLAY = list(PLAYOFF_WEEKS)
avg_next4 = sum(avg_games[i - 1] for i in NEXT4)
avg_play = sum(avg_games[i - 1] for i in PLAY)
# saved start-of-week forecasts and real weekly points (full 7-day weeks only), for the in-season factor below
SHRINK_K = 5
FC_PATH = HUB / "week_forecasts.json"
FCST = {int(w): v for w, v in (json.load(open(FC_PATH, encoding="utf-8"))["weeks"] if FC_PATH.exists() else {}).items()}
ACT = {t["id"]: {int(k): v for k, v in (t.get("weekly_actual") or {}).items() if int(k) not in SPECIAL_WEEKS and CAL[int(k) - 1]["days"] == 7} for t in wp["teams"]}
_pairs = [(ACT[int(tid)][w], f) for w, v in FCST.items() for tid, f in v.items() if int(tid) in ACT and w in ACT[int(tid)]]
LEAGUE_F = sum(a for a, _ in _pairs) / sum(f for _, f in _pairs) if _pairs else None
NEW_FC = {}
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
    # In season: solver x a "beat-the-model" factor. A team's factor in a past week = its real points / the solver's forecast saved at the start of
    # that week (week_forecasts.json: the roster it really had, not today's). The league-wide factor fixes the solver's overall bias; the team's own
    # factor counts only as weeks pile up, weight n / (n + 5). Backtest on 2025-26 (weekly_proj_study.py, 2026-11-02), mean miss per team-week:
    # this 168, the old blend (last 3 weeks' own factors, unshrunk) 177, solver alone 202; after one week played 156 vs the old blend's 217.
    wk = [w for w in sorted(FCST) if str(t["id"]) in FCST[w] and w in ACT.get(t["id"], {})]
    if wk and LEAGUE_F:
        tf = sum(ACT[t["id"]][w] for w in wk) / sum(FCST[w][str(t["id"])] for w in wk)
        mult = LEAGUE_F * (tf / LEAGUE_F) ** (len(wk) / (len(wk) + SHRINK_K))
        proj = {str(w): round(mult * raw_proj[w], 0) for w in raw_proj}
        basis = "blend"
    else:             # preseason: raw solver, scaled by 0.95 (it ran 5-7% high in the 2025-26 backtest)
        proj = {str(w): round(0.95 * raw_proj[w], 0) for w in raw_proj}
        basis = "solver"
    NEW_FC[str(t["id"])] = round(raw_proj[FIRST], 1) if FIRST in raw_proj else None
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
# REPLACEMENT FILL. A roster this thin leaves slot-days empty, and an empty slot-day is not worth zero: you can stream a free agent into it (the best three free agents are about 27-29 points a game).
# Valuing a stash against an EMPTY slot made every injured scrub who can sit on a free IR slot look like +300 or more (Nae'Qwan Tomlin, a 17-point player, 'gained' 342) while a real 33-point player
# who costs a drop looked smaller. So every candidate is valued against the roster plus the best three free agents standing in as the streamers you would otherwise use.
_fa_sorted = sorted([x for x in wp.get("fa_pool", []) if not x.get("waiver") and x.get("status") in ("ACTIVE", "DAY_TO_DAY") and x["level"] > 0], key=lambda x: -(x.get("ll") or x["level"]))
_FILL_N = 6
_FILL_SRC = _fa_sorted[:_FILL_N + 1]
def _fillers(exclude_id):
    return [mk(dict(x, ir=False, protected=False, status="ACTIVE", level=x.get("ll") or x["level"])) for x in _FILL_SRC if x["id"] != exclude_id][:_FILL_N]   # same long-run level as the candidates
_FILL = _fillers(None)
_REPL_FILL = sum(f["level"] for f in _FILL) / max(1, len(_FILL))
base_w = {w: week_points(mine + _FILL, w) for w in REMAIN}
mine_ir = sum(1 for p in mine if p["ir"])
stash = []
_seen_ids = {x["id"] for x in wp.get("stash_pool", [])} | {x["id"] for x in wp.get("fa_pool", [])}
# dynasty_adds carries the real dynasty asset value (asset, asset_rank, market_rank, age) that stash_pool/fa_pool entries don't compute at all -- so a
# player good enough to appear in BOTH (e.g. a strong starter who happens to be a free agent here) used to lose that data entirely: whichever pool's
# loop reached him first "won" his row, and if that was stash_pool/fa_pool, his asset/dyn_pts silently stayed at None/0 even though he's clearly a
# real asset (found 2026-09-27, e.g. Walker Kessler, Ivica Zubac, Nickeil Alexander-Walker, Ryan Rollins all showed up with no asset score). Now
# looked up by id and merged onto every row, regardless of which pool it came from.
dyn_by_id = {d["id"]: d for d in (me.get("dynasty_adds") or [])}
# dynasty_adds carries only a real ESPN status, not games_out/back_date (that detail is only tracked for stash_pool/fa_pool) -- so a dynasty prospect
# who's actually out (e.g. a season-ending injury) can't be safely simulated as if he plays every remaining game. Skip him here rather than assume
# out_now=False (a real bug found 2026-09-27: this hardcoding is a second way an injured player could slip into the ROS gain calc uncaught, alongside
# the ESPN-injuries-feed fallback bug in injury_advisor.espn_injuries). He still surfaces under the DYNASTY tag alone, just without a fabricated ROS
# points estimate on top.
IR_OK_DYN = ("OUT", "INJURY_RESERVE")
_dyn = [dict(d, slots=[q for q in d["pos"]] + ["UT"], out_now=False)
        for d in (me.get("dynasty_adds") or []) if d["id"] not in _seen_ids and d["level"] > 0 and d.get("status") not in IR_OK_DYN]
for src, pool in (("stash", wp.get("stash_pool", [])), ("fa", wp.get("fa_pool", [])), ("fa", _dyn)):
    for x in pool:
        if x["team"] not in heat and canon(x["team"]) not in heat:
            continue
        if src == "stash" and x.get("out_now") and not x.get("back_date"):
            continue                                                 # out for the season
        dd = dyn_by_id.get(x["id"])
        # the ROS-gain simulation and the displayed level should both use his TRUE level (level_raw), not the shrink meant for THIS WEEK's noisy,
        # small-sample streaming decisions (FA_SHRINK's backtest was about a single week's realized gain, not a 70-game season -- see FA_SHRINK's
        # comment in build_week_plan.py; found 2026-09-27 via Jalen Green showing "level" 31.1 here against his real ~37-42 projection elsewhere).
        level = x.get("ll") or x.get("level_raw", x["level"])       # the same long-run level the weekly plan uses (ESPN and DELCO averaged, light shrink for veterans)
        f = mk(dict(x, ir=False, protected=False, status="ACTIVE", level=level))
        if x.get("out_now"):
            f["back"] = x["back_date"]
        to_ir = bool(x.get("out_now")) and x.get("status") in IR_OK_S and mine_ir < 4
        drop = None if to_ir or len(non_ir) < 15 else (droppable[0] if droppable else None)
        _fl = _fillers(x["id"]) if any(y["id"] == x["id"] for y in _FILL_SRC[:_FILL_N]) else _FILL
        new = [p for p in mine if p is not drop] + [f] + _fl
        _bw = base_w if _fl is _FILL else {w: week_points(mine + _fl, w) for w in REMAIN}
        gw = {w: week_points(new, w) - _bw[w] for w in REMAIN}
        reg = sum(g for w, g in gw.items() if w not in PLAY)
        po = sum(g for w, g in gw.items() if w in PLAY)
        n4 = sum(gw[w] for w in NEXT4)
        total_w = reg + PLAYOFF_WT * po
        if level < _REPL_FILL:                                   # a player below what you would stream into the same slot adds nothing over the rest of the season, however many empty slot-days the roster has
            reg = po = n4 = total_w = 0
        asset = x.get("asset") if x.get("asset") is not None else (dd.get("asset") if dd else None)
        stash.append({"form": x.get("form"), "id": x["id"], "name": x["name"], "team": f["team"], "level": level, "src": src, "out_now": bool(x.get("out_now")), "status": x.get("status"), "injury": x.get("injury"),
                      "games_out": x.get("games_out"), "back_date": x.get("back_date"), "espn_return": x.get("espn_return"), "p_back_playoffs": x.get("p_back_playoffs"),
                      "age": x.get("age") if x.get("age") is not None else (dd.get("age") if dd else None), "asset": asset, "dyn_pts": round(0.25 * 73 * (max(0.0, (asset or 0) - me.get("asset_5th", 0)) + 0.1 * (asset or 0))),
                      "asset_rank": x.get("asset_rank") if x.get("asset_rank") is not None else (dd.get("asset_rank") if dd else None),
                      "market_rank": x.get("market_rank") if x.get("market_rank") is not None else (dd.get("market_rank") if dd else None),
                      "kind": x.get("kind") if x.get("kind") is not None else (dd.get("kind") if dd else None), "slots": x.get("slots"),
                      "gain": round(total_w), "reg": round(reg), "po": round(po), "next4": round(n4), "drop": None if to_ir or not drop else drop["name"], "to_ir": to_ir})
keep = [e for e in stash if e["gain"] >= 100]       # anyone who adds real points over the rest of the season (near-term ones are also in the suggested moves; the tag is the same)
keep.sort(key=lambda e: -e["gain"])
print("long-term (ROS) stash:", [(e["name"], e["gain"], e["next4"]) for e in keep[:6]])

out = {"generated": datetime.now(timezone.utc).isoformat(), "calendar_assumed": not SCAL.verified(_CAL), "schedule_provisional": True,
       "calendar": [{"id": w["id"], "start": w["start"].isoformat(), "end": w["end"].isoformat(), "days": w["days"], "cap": w["cap"], "playoff": w["playoff"]} for w in CAL],
       "current_week": FIRST, "next4": NEXT4, "playoff_weeks": PLAY, "avg_games": avg_games, "avg_next4": round(avg_next4, 1), "avg_playoffs": round(avg_play, 1),
       "nba": heat, "teams": teams_out, "my_abbrev": wp["my_abbrev"],
       "stash": keep[:25], "stash_all": {str(e["id"]): [e["gain"], e["dyn_pts"]] for e in stash}, "playoffs": {"base": round(base_po, 0), "fa_adds": adds[:10], "trade_targets": targets[:15], "streamers_next_week": streamers}}
(HUB / "schedule_plan.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote schedule_plan.json", round((HUB / "schedule_plan.json").stat().st_size / 1024), "KB")
# this week's start-of-week forecast, kept once (the first run that sees the week) for the in-season factor above
if FIRST not in SPECIAL_WEEKS and FIRST not in FCST and all(v is not None for v in NEW_FC.values()):
    FCST[FIRST] = NEW_FC
    FC_PATH.write_text(json.dumps({"weeks": {str(w): v for w, v in sorted(FCST.items())}}, separators=(",", ":")), encoding="utf-8")
    print(f"saved week {FIRST}'s start-of-week forecasts")
