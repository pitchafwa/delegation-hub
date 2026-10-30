"""Live matchup tracker data: writes live_matchup.json (score so far, expected remaining points, win probability, and its history through the week).

Runs every 20 minutes from the alerts workflow (needs ESPN_S2 / SWID) and publishes the file to the `live-data` branch, which the page reads from raw.githubusercontent.com.
  score          ESPN's live matchup totals for the user's team and this week's opponent
  score          ESPN's LIVE totals (totalPointsLive: includes games in progress)
  expected rest  players in each team's ACTUAL lineup today: the unplayed share of their game (from ESPN's NBA scoreboard clock) x their level (tonight's projection when the plan has one)
                 x chance of playing for players not in yet, plus the plan's expected points for the days after today
  win chance     Phi( (margin now + 0.72 x expected remaining margin) / sigma(days left) ), fit and checked on 2025-26 (win_prob_study.py: calibrated within ~2 points)
  the cap        starts so far vs the games-played cap (a locked lineup ends scoring for the week)
Test on last season:  LIVE_YEAR=2026 LIVE_DATE=2026-01-15 LIVE_MP=13 uv run python research/live_matchup.py --dry
"""
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent.parent
DASH = ROOT / "dashboard"
RAW_URL = "https://raw.githubusercontent.com/pitchafwa/delegation-hub/live-data/live_matchup.json"
from season import current_season_id
YEAR = int(os.environ.get("LIVE_YEAR", current_season_id()))
import season_calendar as SCAL
SEASON_START = SCAL.opener() if YEAR == current_season_id() else {2026: date(2025, 10, 21)}.get(YEAR, SCAL.opener())      # derived for this season; last season's opener only for the --dry replay test
# the fantasy day runs until 6am ET: games that tip at 10:30pm still belong to that day's scoring period, so after midnight keep tracking them (2026-10-29)
TODAY = date.fromisoformat(os.environ["LIVE_DATE"]) if os.environ.get("LIVE_DATE") else (datetime.now(ET) - timedelta(hours=6)).date()
NOW = datetime.now(ET)
DRY = "--dry" in sys.argv
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "ingest" / "live_out" / "live_matchup.json"
MODEL = json.load(open(Path(__file__).resolve().parent / "win_prob_model.json"))
BETA = MODEL["beta"]
SIG = {int(k): v for k, v in MODEL["sigma"].items()}
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
CAP7 = 40.0


def sigma(rem):
    if rem <= 0:
        return 1e-6
    if rem < 1:
        return SIG[1] * math.sqrt(rem)
    if rem <= 7:
        lo = int(rem)
        hi = min(7, lo + 1)
        return SIG[lo] + (SIG[hi] - SIG[lo]) * (rem - lo)
    return SIG[7] * math.sqrt(rem / 7.0)


ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
GAME_INFO = {}             # canon team -> {state pre/in/post, clock text, tip} for today's games (filled by game_progress)


def game_progress():
    """canon NBA team -> share of its game still to play today (1.0 = not started, 0.0 = final), from ESPN's public NBA scoreboard (the NBA's own live feed blocks scripts).
    {} if unavailable: then a player counts as 'to play' until ESPN marks him as having played, which is the old behaviour."""
    from team_abbr import canon
    evs = None
    for _try in range(3):                  # a failed fetch made every starter count as playing today: one-point win-chance drops of ~15 (2026-10-27/28)
        try:
            evs = requests.get(ESPN_SB, params={"dates": TODAY.strftime("%Y%m%d")}, timeout=20).json().get("events", [])
            if evs:
                break
        except Exception as ex:
            print("NBA scoreboard unavailable:", repr(ex)[:80])
        time.sleep(4)
    if not evs:
        return {}
    out = {}
    GAME_INFO.clear()
    for e in evs:
        c = e["competitions"][0]
        st = c["status"]
        state = st["type"]["state"]
        info = {"state": state, "clock": (st.get("type") or {}).get("shortDetail") or "", "tip": e.get("date"), "event": e.get("id")}
        for t in c["competitors"]:
            GAME_INFO[canon(t["team"]["abbreviation"])] = info
        if state == "pre":
            left = 1.0
        elif state == "post":
            left = 0.0
        else:
            per = int(st.get("period") or 1)
            secs = float(st.get("clock") or 0.0)
            left = max(0.0, ((4 - per) * 720 + secs) / 2880.0) if per <= 4 else max(0.0, secs / 2880.0)    # overtime: at most five more minutes
        for t in c["competitors"]:
            out[canon(t["team"]["abbreviation"])] = left
    return out


_BOX = {}


def ruled_out(event):
    """ESPN athlete ids listed as Out on the game's own page (ESPN game summary 'injuries'), which updates at tip-off even when the fantasy injury tag lags.
    ESPN athlete ids are the fantasy player ids. Empty set if unavailable."""
    if not event:
        return set()
    if event not in _BOX:
        try:
            j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary", params={"event": event}, timeout=20).json()
            _BOX[event] = {int(i["athlete"]["id"]) for t in j.get("injuries", []) for i in t.get("injuries", []) if str(i.get("status", "")).lower() == "out"}
        except Exception as ex:
            print("game summary unavailable:", repr(ex)[:80])
            _BOX[event] = set()
    return _BOX[event]


def load(name):
    p = DASH / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def main():
    if TODAY < SEASON_START and not os.environ.get("LIVE_YEAR"):
        print("before the season: nothing to track")
        return
    W = load("week_plan.json") or {}
    plan_ok = W.get("season") == YEAR
    my_ab = W.get("my_abbrev", "DRNK")
    lg = League(league_id=config.LEAGUE_ID, year=YEAR, espn_s2=config.ESPN_S2, swid=config.SWID)
    raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mTeam"]})
    teams = {t["id"]: (t.get("abbrev") or t.get("name") or str(t["id"])) for t in raw["teams"]}
    names = {t["id"]: ((t.get("location", "") + " " + t.get("nickname", "")).strip() or t.get("name", "")) for t in raw["teams"]}
    my_id = next((i for i, a in teams.items() if a == my_ab), None)
    if my_id is None:
        print("cannot find my team")
        return
    # the matchup that contains the fantasy day (late on the last night of a matchup ESPN may already show the next one)
    mp = int(os.environ.get("LIVE_MP") or next((i for i, a_, b_ in SCAL.bounds() if a_ <= TODAY <= b_), None) or raw.get("status", {}).get("currentMatchupPeriod") or 0)
    m = next((x for x in raw["schedule"] if x.get("matchupPeriodId") == mp and "away" in x and my_id in (x["home"]["teamId"], x["away"]["teamId"])), None)
    if m is None:
        print("no matchup found for period", mp)
        return
    mine_side, opp_side = ("home", "away") if m["home"]["teamId"] == my_id else ("away", "home")
    opp_id = m[opp_side]["teamId"]
    pts = {my_id: float(m[mine_side].get("totalPointsLive") or m[mine_side].get("totalPoints") or 0), opp_id: float(m[opp_side].get("totalPointsLive") or m[opp_side].get("totalPoints") or 0)}
    by_sp = {my_id: m[mine_side].get("pointsByScoringPeriod") or {}, opp_id: m[opp_side].get("pointsByScoringPeriod") or {}}
    if os.environ.get("LIVE_DATE"):                              # testing on a finished season: only count periods up to the test date
        lim = (TODAY - SEASON_START).days + 1
        for tid in pts:
            pts[tid] = float(sum(v for k, v in by_sp[tid].items() if int(k) <= lim))
    plan_match = plan_ok and W["matchup"]["id"] == mp
    days = [date.fromisoformat(d) for d in W["matchup"]["days"]] if plan_match else None
    if days is None:                                        # no matching plan: infer the week from the scoring periods ESPN has scored so far
        sp_keys = sorted(int(k) for k in by_sp[my_id])
        first = SEASON_START + timedelta(days=(sp_keys[0] - 1)) if sp_keys else TODAY
        days = [first + timedelta(days=i) for i in range(7)]
    cap = W["matchup"]["cap"] if plan_match else round(CAP7 * len(days) / 7.0, 1)
    sp_of = lambda d: (d - SEASON_START).days + 1
    today_sp = sp_of(TODAY)

    def box(sp):
        b = lg.espn_request.league_get(params={"view": "mBoxscore", "scoringPeriodId": sp})
        res = {my_id: [], opp_id: []}
        for mm in b.get("schedule", []):
            if mm.get("matchupPeriodId") != mp:
                continue
            for side in ("home", "away"):
                t = mm.get(side, {})
                if t.get("teamId") not in res:
                    continue
                for e in (t.get("rosterForCurrentScoringPeriod") or {}).get("entries", []):
                    pl = e.get("playerPoolEntry", {}).get("player", {})
                    st = [s for s in pl.get("stats", []) if s.get("scoringPeriodId") == sp and s.get("statSourceId") == 0]
                    played = bool(st and st[0].get("stats", {}).get("42", 0))
                    res[t["teamId"]].append(dict(id=pl.get("id"), name=pl.get("fullName"), slot=e.get("lineupSlotId"), played=played, pts=(st[0].get("appliedTotal") if st else None) or 0.0,
                                                 status=pl.get("injuryStatus"), team=pl.get("proTeamId")))
        return res

    # ---- starts so far (past days + players who already played today)
    starts = {my_id: 0, opp_id: 0}
    sb = {my_id: 0, opp_id: 0}                       # starts at the START of today: ESPN checks the games cap at the start of each day
    today_box = None
    for d in days:
        if d > TODAY:
            break
        b = box(sp_of(d))
        if d == TODAY:
            today_box = b
        for tid in (my_id, opp_id):
            n_ = sum(1 for e in b[tid] if e["slot"] is not None and e["slot"] <= 11 and e["played"])
            starts[tid] += n_
            if d < TODAY:
                sb[tid] += n_
    # ---- expected points still to come
    plan_team = {t["id"]: t for t in W.get("teams", [])} if plan_match else {}
    plan_roster = {tid: {p["id"]: p for p in plan_team[tid]["roster"]} for tid in plan_team}
    nba_today = set((W.get("matchup", {}).get("nba_games", {}) or {}).get(TODAY.isoformat(), [])) if plan_match else set()
    rem_today = {my_id: 0.0, opp_id: 0.0}
    left_today = {my_id: [], opp_id: []}
    p_start = {my_id: [], opp_id: []}                       # chance each not-yet-played starter adds a start today (for the cap lockout odds)
    full_day = {my_id: 0.0, opp_id: 0.0}
    from espn_api.basketball.constant import PRO_TEAM_MAP
    from team_abbr import canon
    prog = game_progress() if not os.environ.get("LIVE_DATE") else {}
    from espn_api.basketball.constant import POSITION_MAP as _PM
    lineup = {my_id: [], opp_id: []}
    for tid in (my_id, opp_id):
        for e in (today_box or {}).get(tid, []):
            if e["slot"] is None or e["slot"] > 11:
                continue
            pr = plan_roster.get(tid, {}).get(e["id"], {})
            tcanon = canon(PRO_TEAM_MAP.get(e["team"]) or "")
            gi = GAME_INFO.get(tcanon)
            started = bool(gi and gi["state"] in ("in", "post"))
            not_dressed = bool(gi) and not e["played"] and e["id"] is not None and int(e["id"]) in ruled_out(gi.get("event"))      # ruled out on the game page
            if gi is not None or not prog:
                outp = e["status"] in ("OUT", "INJURY_RESERVE", "SUSPENSION") or not_dressed
                stt = ("out" if outp and not e["played"] else ("final" if gi and gi["state"] == "post" else ("live" if gi and gi["state"] == "in" else "pre"))) if gi else ("final" if e["played"] else "pre")
                lineup[tid].append({"name": e["name"], "slot": _PM.get(e["slot"], ""), "pts": round(e["pts"], 1) if e["played"] else (0.0 if started and not outp else None), "st": stt,
                                    "clock": gi["clock"] if gi else "", "tip": gi["tip"] if gi else None, "proj": round(float(pr.get("tonight") or pr.get("level") or 0), 1), "played": bool(e["played"])})
            lvl = pr.get("tonight") or pr.get("level", 22.0)
            boost = (pr.get("boost") or {}).get(TODAY.isoformat(), 0.0) if isinstance(pr.get("boost"), dict) else 0.0
            left = prog.get(canon(PRO_TEAM_MAP.get(e["team"]) or ""), None)
            if left is None:
                if prog or (nba_today and tcanon not in nba_today):   # his team is not on the NBA board (or, without it, not in the plan's games for today): no game today
                    continue
                left = 0.0 if e["played"] else 1.0                   # no board at all: fall back to "to play until ESPN marks him as played"
            p_play = 0.0 if (e["status"] in ("OUT", "INJURY_RESERVE", "SUSPENSION") or not_dressed) else 0.94
            full_day[tid] += (lvl + boost) * p_play
            if left <= 0.0 or p_play <= 0.0:
                continue
            if e["played"]:
                ef = (lvl + boost) * left                            # on the floor tonight: the unplayed share of his game is still to come
                left_today[tid].append({"name": e["name"], "ef": round(ef, 1), "live": round(e["pts"], 1)})
            else:
                ef = (lvl + boost) * p_play * left                   # not in yet: late in a game he is likely sitting, so scale by what is left
                left_today[tid].append({"name": e["name"], "ef": round(ef, 1)})
                p_start[tid].append(p_play * (1.0 if left >= 1.0 else left))
            rem_today[tid] += ef
    # OPPONENT'S UNSET LINEUP (2026-10-30): before his games tip, the opponent's ESPN lineup may not be set yet (in the morning it had 4 of his players with games
    # in it; the plan expects ~8 and already scales for how well he usually sets lineups, eff). Until a game starts, his expected points and starts from games
    # not yet started are at least the plan's. Once a game tips, its lineup is locked and the real lineup counts.
    pre_n = {my_id: None, opp_id: None}
    _ot = plan_team.get(opp_id) or {}
    _orow = next((x for x in _ot.get("days", []) if x["date"] == TODAY.isoformat()), None)
    if _orow and sb[opp_id] < cap:
        _now = datetime.now(timezone.utc)
        _pre = [x for x in _orow.get("start", []) if x.get("tip") and datetime.fromisoformat(x["tip"].replace("Z", "+00:00")) > _now]
        _plan_pts = sum(x.get("ef", 0.0) for x in _pre) * float(_ot.get("eff") or 1.0)
        _act_pre = [x for x in left_today[opp_id] if "live" not in x]
        _act_pts = sum(x["ef"] for x in _act_pre)
        if _plan_pts > _act_pts:
            rem_today[opp_id] += _plan_pts - _act_pts
            pre_n[opp_id] = max(len(_act_pre) * 0.94, sum(x.get("p", 0.94) for x in _pre))
            print(f"opponent lineup not set yet: {len(_act_pre)} starters still to play vs {len(_pre)} in his plan; using the plan's {_plan_pts:.0f} (lineup {_act_pts:.0f})")
    # GAMES CAP, for BOTH teams: a team that began today at or over the cap cannot start anyone today or later (nothing more counts), and a team projected to cross it by the end of
    # today loses the days after. The plan assumes every team uses the cap perfectly and is only refreshed a few times a day, so without this a locked opponent still "scores" in the forecast.
    lock_note = {}
    cum = {}
    for tid in (my_id, opp_id):
        if sb[tid] >= cap:
            rem_today[tid], left_today[tid] = 0.0, []
            lock_note[tid] = "locked"
        started_today = sum(1 for e in (today_box or {}).get(tid, []) if e["slot"] is not None and e["slot"] <= 11 and e["played"])
        cum[tid] = sb[tid] + started_today + (0 if sb[tid] >= cap else (pre_n[tid] if pre_n[tid] is not None else sum(1 for x in left_today[tid] if "live" not in x) * 0.94))   # players already on the floor are counted in started_today
    # LOCKOUT ODDS (2026-10-24): whether tomorrow is still open depends on how many of today's starters actually play. A team at 25 starts with 10 starters today and a
    # 34.3 cap locks itself out of tomorrow only if all 10 play (about 54%). The day after today is weighted by that chance instead of being treated as all or nothing.
    def p_open_tomorrow(tid):
        base = sb[tid] + sum(1 for e in (today_box or {}).get(tid, []) if e["slot"] is not None and e["slot"] <= 11 and e["played"])
        if sb[tid] >= cap:
            return 0.0
        dist = [1.0]                                            # distribution of today's remaining starts (each starter plays or not)
        for q in p_start[tid]:
            nd = [0.0] * (len(dist) + 1)
            for k, v in enumerate(dist):
                nd[k] += v * (1 - q)
                nd[k + 1] += v * q
            dist = nd
        return sum(v for k, v in enumerate(dist) if base + k < cap)
    p_open = {tid: p_open_tomorrow(tid) for tid in (my_id, opp_id)}
    next_day = next((x for x in days if x > TODAY), None)
    cum_end = dict(cum)                                         # expected starts at the end of today (before tomorrow's games)
    fut = {my_id: 0.0, opp_id: 0.0}
    fut_days = 0
    # OPPONENT STREAMING (opponent_adds_net.py): his usual net gain from in-week adds, per day still open to him, capped at ~30 points per add he had left when the plan was built
    # (an add made since then is not in the plan's days yet, so the estimate keeps standing in for it until the plan refreshes). MY days are the plan's days after its suggested moves.
    _tus = (load("team_usage.json") or {}).get("stream") or {}
    o_rate = float((_tus.get(str(opp_id)) or {}).get("r", 41.0)) / 7.0
    o_adds = (plan_team.get(opp_id) or {}).get("adds_left")
    o_cap = 30.0 * o_adds if o_adds is not None else 1e9
    stream = 0.0
    # BOTH teams re-plan (2026-10-29, Tommy): the plan's days already keep each team's last day open under the cap. Running a start or two ahead of the plan
    # used to push the projected count over the cap and wipe out the whole last day (-200 points, win chance 73% -> 52%), when the team would simply bench
    # one or two cheap starts earlier. So each team's plan decides which later days are open, the extra starts cost what that team's cheapest planned starts
    # are worth, and a lockout counts only once it is real (at the cap when a day begins) or set by tonight's lineups (tomorrow's odds, p_open).
    rows_ = {tid: ((plan_team.get(tid) or {}).get("days_after" if tid == my_id else "days") or (plan_team.get(tid) or {}).get("days") or []) for tid in (my_id, opp_id)}
    today_ = {tid: next((x for x in rows_[tid] if x["date"] == TODAY.isoformat()), None) for tid in (my_id, opp_id)}
    replan = {tid: bool(today_[tid]) and sb[tid] < cap for tid in (my_id, opp_id)}
    extra = {tid: max(0.0, cum_end[tid] - today_[tid]["starts_after"]) if replan[tid] else 0.0 for tid in (my_id, opp_id)}
    for d in days:
        if d <= TODAY:
            continue
        got = False
        for tid in (my_id, opp_id):
            t = plan_team.get(tid)
            dd = next((x for x in (t or {}).get("days_after" if tid == my_id and (t or {}).get("days_after") else "days", []) if x["date"] == d.isoformat()), None)
            w_open = p_open[tid] if d == next_day else (1.0 if cum[tid] < cap else 0.0)
            if replan[tid] and dd and d != next_day:
                w_open = 1.0 if dd.get("starts_before", 0) + extra[tid] < cap + 1.0 else 0.0      # only a clear overshoot (a full start beyond what benching can absorb) locks
            if w_open <= 0.0:                                     # certainly at the cap when this day begins: the day is locked
                lock_note.setdefault(tid, "locks after today" if d == next_day else "locked from " + d.strftime("%a"))
                continue
            if d == next_day and w_open < 0.95:
                lock_note.setdefault(tid, f"may lock after today ({1 - w_open:.0%} chance)")
            if dd:
                fut[tid] += w_open * dd["pts"] * (float(t.get("eff") or 1.0) if tid != my_id else 1.0)      # the plan's days assume perfect lineups: scale by the owner's real habits
                if tid == opp_id:
                    stream += w_open * o_rate
                cum[tid] += w_open * max(0.0, dd.get("starts_after", 0) - dd.get("starts_before", 0))
                got = got or dd["pts"] > 0
            elif not plan_match:                             # crude fallback: this team's average points per day so far
                played_days = max(1, len(by_sp[tid]))
                fut[tid] += pts[tid] / played_days
                got = True
        fut_days += 1 if got else 0
    for tid in (my_id, opp_id):                              # the starts each team now has to give up, at the value of its cheapest remaining planned starts
        per = [x["pts"] / (x["starts_after"] - x["starts_before"]) for x in rows_[tid] if x["date"] > TODAY.isoformat() and x.get("starts_after", 0) > x.get("starts_before", 0) + 0.5]
        if extra[tid] > 0 and per:
            fut[tid] = max(0.0, fut[tid] - extra[tid] * 0.8 * min(per) * (float((plan_team.get(tid) or {}).get("eff") or 1.0) if tid != my_id else 1.0))
    stream = max(0.0, min(stream, o_cap))
    fut[opp_id] += stream
    frac_today = 0.0
    if full_day[my_id] + full_day[opp_id] > 0:
        frac_today = min(1.0, (rem_today[my_id] + rem_today[opp_id]) / (full_day[my_id] + full_day[opp_id]))
    elif TODAY in days:
        frac_today = 0.5
    rem_days = frac_today + fut_days
    exp_rem = {tid: rem_today[tid] + fut[tid] for tid in (my_id, opp_id)}
    margin = pts[my_id] - pts[opp_id]
    mu = margin + BETA * (exp_rem[my_id] - exp_rem[opp_id])
    wp = Phi(mu / sigma(rem_days)) if rem_days > 0.02 else (1.0 if margin > 0 else 0.0 if margin < 0 else 0.5)
    # ---- history (kept on the live-data branch between runs)
    hist = []
    try:
        prev = requests.get(RAW_URL + f"?t={int(NOW.timestamp())}", timeout=20).json()
        if prev.get("matchup") == mp and prev.get("season") == YEAR:
            hist = prev.get("history", [])
    except Exception:
        pass
    hist.append({"t": datetime.now(timezone.utc).isoformat(timespec="minutes"), "wp": round(wp, 3), "margin": round(margin, 1), "me": round(pts[my_id], 1), "opp": round(pts[opp_id], 1)})
    # the whole matchup for the week chart (2026-10-28; it kept only the last ~2.5 days): every point from the last 12 hours, one per 30 minutes before that
    _cut = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(timespec="minutes")
    _seen, _thin = set(), []
    for h_ in hist:
        b_ = h_["t"][:14] + ("00" if int(h_["t"][14:16]) < 30 else "30")
        if h_["t"] >= _cut or b_ not in _seen:
            _thin.append(h_)
        _seen.add(b_)
    # drop blips: a run of 1-4 readings (up to ~20 minutes) whose win chance sits ~10+ points off the readings on BOTH sides, in the same direction, while
    # the readings on either side agree with each other and the score margin moved under 25 (worth only a few points of win chance). Bad reads (a failed scoreboard fetch, 2026-10-27/28; the
    # midnight day switch, 2026-10-29) look like that; real news (injury, plan rebuild) does not snap straight back. Re-checked every run, so it heals itself.
    drop_ = set()
    # known-bad stretches too long for the blip rule: 2026-10-30 02:41-03:45 UTC, my last day wrongly counted as locked out (see "MY team re-plans" above)
    for i_, h_ in enumerate(_thin):
        if "2026-10-30T02:41" <= h_["t"] <= "2026-10-30T03:45" or "2026-10-30T09:39" <= h_["t"] <= "2026-10-30T14:15":     # and the morning the opponent's unset lineup counted as final
            drop_.add(i_)
    for i_ in range(1, len(_thin) - 1):
        for k_ in range(1, 5):
            if i_ + k_ >= len(_thin):
                break
            a_, z_, seg_ = _thin[i_ - 1], _thin[i_ + k_], _thin[i_:i_ + k_]
            if (abs(a_["wp"] - z_["wp"]) < 0.05 and abs(a_["margin"] - z_["margin"]) < 25
                    and all((x_["wp"] - a_["wp"]) * (x_["wp"] - z_["wp"]) >= 0.01 and abs(x_["margin"] - a_["margin"]) < 25 for x_ in seg_)):
                drop_.update(range(i_, i_ + k_))
                break
    hist = [h_ for i_, h_ in enumerate(_thin) if i_ not in drop_][-1500:]
    my_plan = plan_team.get(my_id, {})
    adds_used = my_plan.get("adds_used")
    best_add = next((a for a in (my_plan.get("sequence") or [])[:1]), None)
    advice = []
    if lock_note.get(opp_id):
        _ln = lock_note[opp_id]
        if _ln == "locked":
            advice.append("Your opponent has reached the games cap: nothing more counts for them, and your win probability reflects that.")
        elif _ln == "locks after today":
            advice.append("Your opponent will reach the games cap after today: nothing counts for them after that, and your win probability reflects that.")
        elif _ln.startswith("may lock"):
            pct = _ln[_ln.index("(") + 1:_ln.index("%")]
            advice.append(f"Your opponent has enough starters today to hit the games cap: if all of them play ({pct}% likely) nothing counts for them tomorrow. Your win chance counts that risk.")
        else:
            advice.append(f"Your opponent is projected to reach the games cap before {_ln.replace('locked from ', '')}: days after that should not count for them, and your win probability reflects that (it is a projection, not yet true).")
    locked = starts[my_id] >= cap
    if locked:
        advice.append("You have reached the games cap: nothing else counts this week.")
    elif TODAY == days[-1] and sb[my_id] < cap:
        advice.append("Last day and you started it under the cap: every start today counts, so start everyone who plays.")
    elif starts[my_id] >= cap - 4:
        advice.append(f"You are {cap - starts[my_id]:.0f} starts from the cap: a day you start at or over it counts nothing, so save starts for your best remaining games.")
    if wp >= 0.9:
        advice.append("The week is nearly won: avoid needless risk, and do not spend adds you do not need.")
    elif wp <= 0.1 and rem_days > 0.5:
        need = -mu
        advice.append(f"A long shot: you need about {need:.0f} points more than expected. Only moves that add expected points matter; variance choices barely change the odds.")
    elif rem_days > 0.5:
        _adds_left = (W["matchup"]["adds_limit"] - adds_used) if (plan_match and adds_used is not None) else None
        advice.append(f"Expected final margin {mu:+.0f}. Every extra started game is worth about 36 points, so fill every open slot"
                      + (" and use your remaining adds well." if _adds_left is None or _adds_left > 0 else "."))
    if best_add and rem_days > 1 and (adds_used is None or adds_used < W["matchup"]["adds_limit"]):
        _on = best_add.get("on")
        _when = "today" if (not _on or _on <= TODAY.isoformat()) else datetime.fromisoformat(_on).strftime("%a")
        _dr = (best_add.get("drop") or {}).get("name")
        advice.append(f"Next planned add: {best_add['add']['name']} ({_when}" + (f", drop {_dr}" if _dr else "") + f", about +{best_add['gain']:.0f}).")
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "season": YEAR, "matchup": mp, "date": TODAY.isoformat(), "plan_used": bool(plan_match),
           "me": {"abbrev": teams[my_id], "name": names[my_id], "pts": round(pts[my_id], 1), "exp_rem": round(exp_rem[my_id], 1), "exp_final": round(pts[my_id] + exp_rem[my_id], 1), "starts": starts[my_id], "adds_used": adds_used, "left_today": sorted(left_today[my_id], key=lambda x: -x["ef"])[:8], "lineup": lineup[my_id], "starts_before_today": sb[my_id], "starts_end_today": round(cum_end[my_id], 1)},
           "opp": {"abbrev": teams[opp_id], "name": names[opp_id], "pts": round(pts[opp_id], 1), "exp_rem": round(exp_rem[opp_id], 1), "exp_final": round(pts[opp_id] + exp_rem[opp_id], 1), "starts": starts[opp_id], "left_today": sorted(left_today[opp_id], key=lambda x: -x["ef"])[:8], "lineup": lineup[opp_id], "starts_before_today": sb[opp_id], "starts_end_today": round(cum_end[opp_id], 1), "exp_stream": round(stream, 1)},
           "cap": cap, "days_left": round(rem_days, 2), "win_prob": round(wp, 3), "expected_margin": round(mu, 1), "advice": advice, "history": hist}
    print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=1)[:40000])
    if DRY:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    try:                                                     # ledger: one compact snapshot per run (win-probability calibration against the final result)
        led = ROOT / "ledger"
        led.mkdir(exist_ok=True)
        with open(led / f"live-{TODAY.strftime('%Y-%m')}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "live", "ts": out["generated"], "season": YEAR, "matchup": mp, "wp": out["win_prob"], "margin": round(margin, 1), "me": out["me"]["pts"], "opp": out["opp"]["pts"],
                                 "exp_me": out["me"]["exp_final"], "exp_opp": out["opp"]["exp_final"], "days_left": out["days_left"], "opp_abbrev": out["opp"]["abbrev"]}, separators=(",", ":")) + chr(10))
    except Exception:
        pass
    print("wrote", OUT)


if __name__ == "__main__":
    main()
