"""Phone alerts through ntfy (free push notifications; the topic name is the secret).

  python research/alerts.py check     every 15-20 min while games are near: status changes on your roster, new dynasty free agents, lineup problems before tip-off
  python research/alerts.py daily     ~8:30 ET: headline moves for today
  python research/alerts.py weekly    ~9:30 ET on the first day of a matchup (Monday): the week's plan
  python research/alerts.py lineup    every ~5 min on game days (cheap; exits at once if no game of yours tips within 4h): LATE-SCRATCH watch. When a player in
                                      your lineup is ruled out before his game, it names the swap (bench player or free agent) and re-sends a reminder 35 min before tip.
  python research/alerts.py test      one test notification
Flags: --dry-run (print, do not send), --force (ignore time windows and the offseason gate), --now 2026-10-20T09:30 (pretend it is this ET time),
       --pretend-out "Player Name" (test: treat him as ruled out)
Needs NTFY_TOPIC (env).  `check` also needs ESPN_S2 / SWID.  State lives in alerts/state.json so overlapping or duplicate runs never send the same alert twice.
Design rules: alert only on CHANGES; never alert on a suggested add unless the weekly plan actually recommends it; quiet hours 11pm-8am ET (early-tip lineup checks excepted).
"""
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from team_abbr import canon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent.parent
DASH = ROOT / "dashboard"
STATE_PATH = ROOT / "alerts" / "state.json"
SITE = "https://pitchafwa.github.io/delegation-hub/"
try:
    import season_calendar as _SCAL
    ACTIVE_FROM = _SCAL.opener() - timedelta(days=1)   # the day before the opener (derived): nothing is sent before this except `test` and --force
except Exception:
    ACTIVE_FROM = date(2026, 10, 19)
QUIET = (23, 8)                           # no status/FA alerts from 11pm to 8am ET
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY = "--dry-run" in sys.argv
FORCE = "--force" in sys.argv
NOW = datetime.now(ET)
for i, a in enumerate(sys.argv):
    if a == "--now":
        NOW = datetime.fromisoformat(sys.argv[i + 1]).replace(tzinfo=ET)
TODAY = NOW.date()
PRETEND_OUT = set()
for i, a in enumerate(sys.argv):
    if a == "--pretend-out":
        PRETEND_OUT.add(sys.argv[i + 1].lower())
TOPIC = os.environ.get("NTFY_TOPIC")
P_STATUS = {"ACTIVE": 0.94, "DAY_TO_DAY": 0.55, "Available": 0.94, "Probable": 0.9, "Questionable": 0.55, "Doubtful": 0.03, "OUT": 0.0, "Out": 0.0,
            "INJURY_RESERVE": 0.0, "SUSPENSION": 0.0}


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def load(name):
    p = DASH / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def state():
    return json.load(open(STATE_PATH, encoding="utf-8")) if STATE_PATH.exists() else {}


def save_state(s):
    if DRY:
        return
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(s, indent=1, sort_keys=True), encoding="utf-8")


def _ledger(kind, **kw):
    """append-only record of what we sent (ledger/alerts-YYYY-MM.jsonl), graded later"""
    if DRY:
        return
    try:
        d = ROOT / "ledger"
        d.mkdir(exist_ok=True)
        with open(d / f"alerts-{TODAY.strftime('%Y-%m')}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, "ts": datetime.now(ET).isoformat(timespec="seconds"), **kw}, ensure_ascii=False, separators=(",", ":")) + chr(10))
    except Exception:
        pass


def push(title, body, priority=3, tags=None):
    print(f"--- {title} (priority {priority})\n{body}\n")
    if title != "Fantasy Hub test":
        _ledger("alert", title=title, body=body[:600], priority=priority)
    if DRY:
        return True
    if not TOPIC:
        print("NTFY_TOPIC is not set: cannot send")
        return False
    r = requests.post("https://ntfy.sh/", json={"topic": TOPIC, "title": title, "message": body[:3800], "priority": priority, "tags": tags or [], "click": SITE}, timeout=30)
    return r.status_code == 200


def et_hm(dt):
    return dt.astimezone(ET).strftime("%-I:%M%p").lower() if os.name != "nt" else dt.astimezone(ET).strftime("%I:%M%p").lstrip("0").lower()


def my_team(W):
    return next((t for t in W["teams"] if t["abbrev"] == W["my_abbrev"]), W["teams"][0])


def in_quiet():
    return NOW.hour >= QUIET[0] or NOW.hour < QUIET[1]


def active():
    return FORCE or TODAY >= ACTIVE_FROM


# ============================================================ digests
def best_moves(T, n=3):
    out = []
    for m in (T.get("sequence") or [])[:n]:
        by = m.get("by_day") or []
        g = [x["gain"] for x in by]
        timing = ""
        if m.get("on"):                                           # stream manager step: it names the day itself
            timing = " (add now)" if m["on"] <= TODAY.isoformat() else f" (add on {datetime.fromisoformat(m['on']).strftime('%a')})"
            if m.get("release"):
                timing += f", hold until {datetime.fromisoformat(m['release']).strftime('%a')}"
            if m.get("games_next"):
                timing += " (also lines up next week's first days)"
        elif len(g) > 1 and g[0] >= max(g) - 5:
            timing = " (add now" + (f"; waiting a day costs {g[0] - g[1]:.0f})" if g[0] - g[1] >= 3 else ")")
        elif g:
            timing = f" (best on {datetime.fromisoformat(by[g.index(max(g))]['date']).strftime('%a')})"
        drop = f", drop {m['drop']['name']}" if m.get("drop") else " (open roster spot)"
        out.append(f"Add {m['add']['name']} ({m['add']['team']}){drop}: +{m['gain']:.0f}{timing}")
    return out


def last_week_review(W):
    """Monday brief: how last matchup went against the plan. Expected total at its first plan run vs the final score (live ledger), and the moves you made (players that came onto / left
    your roster across the plan runs), including whether players you dropped came back (the re-add rate the streaming math assumes is about 85%)."""
    led = ROOT / "ledger"
    prev = W["matchup"]["id"] - 1
    if prev < 1 or not led.exists():
        return []
    plans, lives = [], []
    for f in sorted(led.glob("plan-*.jsonl")):
        for l in open(f, encoding="utf-8"):
            try:
                j = json.loads(l)
            except Exception:
                continue
            if j.get("matchup") == prev and j.get("team") == W["my_abbrev"]:
                plans.append(j)
    for f in sorted(led.glob("live-*.jsonl")):
        for l in open(f, encoding="utf-8"):
            try:
                j = json.loads(l)
            except Exception:
                continue
            if j.get("matchup") == prev:
                lives.append(j)
    out = []
    if plans and lives:
        p0, lv = plans[0], lives[-1]
        res = "won" if lv["me"] > lv["opp"] else ("lost" if lv["me"] < lv["opp"] else "tied")
        out.append(f"Last week (matchup {prev} vs {lv.get('opp_abbrev', '?')}): the plan expected {p0['expected_total']:.0f} at the start; you scored {lv['me']:.0f} and {res} {lv['me']:.0f}-{lv['opp']:.0f}"
                   + (f" (it gave you {p0['win_prob'] * 100:.0f}%)" if p0.get("win_prob") is not None else "") + ".")
    if len(plans) >= 2:
        names = {}
        rosters = []
        for j in plans:
            ids = set()
            for pl in j.get("players", []):
                ids.add(pl["id"])
                names[pl["id"]] = pl["name"]
            rosters.append(ids)
        added = set().union(*rosters[1:]) - rosters[0]
        dropped = rosters[0] - rosters[-1]
        gone_then_back = {i for i in rosters[0] if any(i not in r for r in rosters[1:]) and i in rosters[-1]}
        if added or dropped or gone_then_back:
            out.append(f"Moves: {len(added)} add(s)" + (f" ({', '.join(names[i] for i in list(added)[:5])})" if added else "")
                       + (f"; dropped and not back: {', '.join(names[i] for i in list(dropped)[:5])}" if dropped else "")
                       + (f"; dropped and re-added: {', '.join(names[i] for i in list(gone_then_back)[:5])}" if gone_then_back else "") + ".")
        else:
            out.append("Moves: none.")
    return out


def digest(kind):
    W = load("week_plan.json")
    if not W:
        return None
    T = my_team(W)
    M = W["matchup"]
    O = next((t for t in W["teams"] if t["id"] == T.get("opp")), None)
    gen = datetime.fromisoformat(W["generated"].replace("Z", "+00:00")).astimezone(ET)
    lines = []
    wp = T.get("win_prob")
    lines.append(f"Matchup {M['id']} vs {O['abbrev'] if O else '?'}: {T['expected_total']:.0f} expected" + (f" vs {O['expected_total']:.0f}" if O else "") + (f", {wp * 100:.0f}% to win" if wp is not None else ""))
    if T.get("ir_moves"):
        for m in T["ir_moves"][:3]:
            lines.append({"to_ir": f"IR: move {m['name']} ({m['status']}) to an open IR slot", "activate": f"IR: {m['name']} is healthy but stuck in IR, activate him",
                          "activate_swap": f"IR: activate {m['name']} and drop {m.get('drop', {}).get('name', '?')}"}.get(m["action"], f"IR: {m['name']}: {m['action']}"))
    mv = best_moves(T, 5 if kind == "weekly" else 3)
    adds_left = M["adds_limit"] - T.get("adds_used", 0)
    lines.append(f"Moves ({adds_left} of {M['adds_limit']} adds left):" if mv else "No add clears the bar this week: hold.")
    lines += ["  " + x for x in mv]
    d = next((x for x in T["days"] if x["date"] == TODAY.isoformat()), None)
    if d:
        if d.get("locked"):
            lines.append("Today: cap reached, nothing counts.")
        elif d.get("sat"):
            lines.append(f"Today: sit {d['sat']} playing player(s) to protect the cap.")
        else:
            lines.append(f"Today: start {len(d['start'])} ({', '.join(x['name'].split()[-1] for x in d['start'][:6])}{'...' if len(d['start']) > 6 else ''}).")
    if kind == "weekly":
        try:
            lines += last_week_review(W)
        except Exception as ex:
            print("last-week review skipped:", repr(ex)[:90])
        SP = load("schedule_plan.json")
        if SP:
            me = next((t for t in SP["teams"] if t["abbrev"] == W["my_abbrev"]), None)
            if me:
                flags = [f"{p['name']} ({p['games'][SP['current_week'] - 1]} games)" for p in me["players"] if "light week soon" in p["flags"] and not p["ir"]][:4]
                if flags:
                    lines.append("Light weeks ahead: " + ", ".join(flags))
    dy = [x for x in (T.get("dynasty_adds") or []) if x.get("new") and x["would_rank"] <= 5][:1]
    if dy:
        lines.append(f"Dynasty stash: {dy[0]['name']} ({dy[0]['age']:.0f}yo) would be your #{dy[0]['would_rank']} keeper asset.")
    lines.append(f"(plan built {gen.strftime('%a %I:%M%p').lstrip('0')} ET)")
    title = ("Week ahead: " if kind == "weekly" else "Today: ") + ("moves to make" if mv else "no moves needed")
    return title, "\n".join(lines)


def heartbeat_problem():
    """the daily model refresh runs on Tommy's PC (it failed silently for 12 days in October 2026): a one-line problem, or None when it is fine"""
    hb = load("local_refresh_status.json")
    try:
        hb_t = datetime.fromisoformat(hb["finished"]) if hb else None
    except Exception:
        hb_t = None
    if hb_t is None:
        rd = load("redraft_data.json") or {}
        try:
            hb_t = datetime.fromisoformat(rd["generated"]).astimezone(ET)    # before the first heartbeat file: the redraft build is part of the same refresh
        except Exception:
            return None
    if (NOW - hb_t).total_seconds() > 36 * 3600:
        return ("The daily model refresh on your PC has not finished since " + hb_t.strftime("%a %b %d") +
                ", so the redraft list, player news and season model are going stale. Check that the PC is on and the FantasyHubDailyRefresh task is running.")
    if hb and hb.get("failed"):
        return "This morning's model refresh on your PC had failed steps: " + ", ".join(hb["failed"][:4]) + ". The rest of the site still updates."
    return None


def run_heartbeat():
    """runs in the daily-brief slot (8:30am ET): an alert only when the PC refresh is stale or failed, at most once a day"""
    s = state()
    if not (active() or FORCE):
        return
    if not FORCE and s.get("sent", {}).get("heartbeat") == TODAY.isoformat():
        return
    msg = heartbeat_problem()
    if msg and push("PC refresh needs attention", msg, priority=3, tags=["warning"]) and not DRY:
        s.setdefault("sent", {})["heartbeat"] = TODAY.isoformat()
        save_state(s)


def run_digest(kind):
    s = state()
    sent = s.setdefault("sent", {})
    W = load("week_plan.json")
    if not FORCE:
        if not active():
            print("before the season: nothing sent")
            return
        if kind == "daily" and not (8 * 60 + 25 <= NOW.hour * 60 + NOW.minute < 12 * 60):
            print("outside the daily-brief window")
            return
        if kind == "weekly":
            if not (9 * 60 + 25 <= NOW.hour * 60 + NOW.minute < 13 * 60):
                print("outside the weekly-brief window")
                return
            if not (NOW.weekday() == 0 or (W and W["matchup"]["start"] == TODAY.isoformat())):
                print("not the first day of a matchup")
                return
        key = TODAY.isoformat() if kind == "daily" else str(W["matchup"]["id"] if W else TODAY)
        if sent.get(kind) == key:
            print(f"{kind} brief already sent ({key})")
            return
    r = digest(kind)
    if not r:
        print("no plan available")
        return
    if push(r[0], r[1], priority=3 if kind == "weekly" else 2, tags=["basketball"]) and not DRY:
        sent[kind] = TODAY.isoformat() if kind == "daily" else str(W["matchup"]["id"])
        save_state(s)


# ============================================================ live checks
def official_today(keys):
    try:
        import nba_injury_reports as NIR
        st, at = NIR.latest_statuses(NOW)
        return {k: v for (g, k), v in st.items() if g == TODAY.isoformat() and k in keys}
    except Exception as ex:
        print("official report unavailable:", ex)
        return {}


def first_tip(W, T):
    sched = load("nba_schedule.json") or {}
    teams = {p["team"] for p in T["roster"] if not p["ir"]}
    tips = []
    for a, h, t in sched.get("games", {}).get(TODAY.isoformat(), []):
        if a in teams or h in teams:
            hh, mm = int(t[:2]), int(t[3:5])
            dt = datetime(TODAY.year, TODAY.month, TODAY.day, hh, mm, tzinfo=ZoneInfo("UTC"))
            if hh < 10:
                dt += timedelta(days=1)                        # 00:00-09:59Z is the evening of the same ET date
            tips.append(dt.astimezone(ET))
    return min(tips) if tips else None


PLAN_RUNS_UTC = [(1, 7), (9, 37), (19, 7), (22, 7)]   # scheduled plan builds: refresh-gameday x3 and the morning refresh-league


def plan_rebuild_check(W, T, mine, s):
    """Ask for a fresh plan (the workflow step after this script dispatches refresh-gameday when rebuild_request.txt exists) when
    (a) Tommy's ESPN roster no longer matches the roster the plan was built from (he made a move), or (b) a scheduled plan build was skipped.
    At most one request every 10 minutes; a skipped run is retried once."""
    now = datetime.now(ZoneInfo("UTC")) if "--now" not in sys.argv else NOW.astimezone(ZoneInfo("UTC"))
    last = s.get("plan_rebuild") or {}
    last_at = datetime.fromisoformat(last["at"]) if last.get("at") else None
    if last_at and (now - last_at) < timedelta(minutes=10):
        return
    gen = datetime.fromisoformat(W["generated"]).astimezone(ZoneInfo("UTC"))
    plan_ids = sorted(int(p["id"]) for p in T["roster"])
    espn_ids = sorted(int(p.playerId) for p in mine.roster)
    why = None
    if plan_ids != espn_ids:
        added = [p.name for p in mine.roster if int(p.playerId) not in set(plan_ids)]
        gone = [p["name"] for p in T["roster"] if int(p["id"]) not in set(espn_ids)]
        sig = ",".join(map(str, espn_ids))
        if last.get("sig") == sig and last_at and gen > last_at:
            return                                             # already rebuilt for this roster (e.g. ESPN's roster read lags); do not loop
        why = "roster changed: " + ", ".join([f"+{n}" for n in added] + [f"-{n}" for n in gone])
    else:
        sig = last.get("sig")
        # most recent scheduled build that should have finished by now (30 min grace for GitHub's delays)
        cands = []
        for d in (0, 1):
            day = (now - timedelta(days=d)).date()
            for hh, mm in PLAN_RUNS_UTC:
                t = datetime(day.year, day.month, day.day, hh, mm, tzinfo=ZoneInfo("UTC"))
                if t <= now - timedelta(minutes=30):
                    cands.append(t)
        exp = max(cands) if cands else None
        if exp and gen < exp - timedelta(minutes=10) and not (last_at and last_at >= exp):
            why = f"scheduled build at {exp.strftime('%H:%MZ')} did not run (plan from {gen.strftime('%H:%MZ')})"
    if not why:
        return
    request_rebuild(s, why, sig, now)


def request_rebuild(s, why, sig=None, now=None):
    now = now or datetime.now(ZoneInfo("UTC"))
    print("plan rebuild requested:", why)
    s["plan_rebuild"] = {"at": now.isoformat(timespec="seconds"), "sig": sig if sig is not None else (s.get("plan_rebuild") or {}).get("sig"), "why": why}
    if not DRY:
        (Path(__file__).resolve().parent.parent / "rebuild_request.txt").write_text(why, encoding="utf-8")


def run_check():
    from espn_api.basketball import League
    import config
    s = state()
    W = load("week_plan.json")
    if not W:
        print("no plan file")
        return
    T = my_team(W)
    lg = League(league_id=config.LEAGUE_ID, year=W["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    mine = next(t for t in lg.teams if t.team_abbrev == W["my_abbrev"])
    plan_by_id = {p["id"]: p for p in T["roster"]}
    keys = {norm(p.name).replace(" ", ""): p for p in mine.roster}
    off = official_today(set(keys))
    can_send = active() or FORCE
    first_run = "last_status" not in s
    prev = s.get("last_status", {})
    cur, msgs = {}, []
    for p in mine.roster:
        pk = str(p.playerId)
        e = (p.injuryStatus or "ACTIVE")
        o = off.get(norm(p.name).replace(" ", ""))
        cur[pk] = {"espn": e, "official": o, "date": TODAY.isoformat()}
        old = prev.get(pk)
        pl = plan_by_id.get(p.playerId, {})
        relevant = (pl.get("level", 0) >= 20) or (p.lineupSlot not in ("BE", "IR"))
        if old is None or not relevant or p.lineupSlot == "IR":
            continue
        changed_e = old["espn"] != e
        changed_o = old.get("official") != o and o is not None and old.get("date") == TODAY.isoformat()
        if not (changed_e or changed_o):
            continue
        p_before = P_STATUS.get(old.get("official") if old.get("date") == TODAY.isoformat() and old.get("official") else old["espn"], 0.94)
        p_after = P_STATUS.get(o or e, 0.94)
        left = len([g for g in pl.get("games", []) if g >= TODAY.isoformat()])
        impact = pl.get("level", 0) * left * (p_after - p_before)
        what = f"{old['espn']} to {e}" if changed_e else f"official report now {o}"
        adv = next((a for a in (T.get("injury_advice") or []) if a["id"] == p.playerId), None)
        if adv and e in ("OUT", "INJURY_RESERVE"):
            what += f". Expected out about {adv['plan_games']:.0f} more games; {adv['verdict'].lower()}"
        msgs.append((abs(impact), f"{p.name}: {what}" + (f" (official: {o})" if o and changed_e else "") + (f". About {impact:+.0f} pts over {left} game(s) left this week." if left and abs(impact) >= 5 else ".")
                     + (" He is in your lineup." if p.lineupSlot not in ("BE", "IR") else "")))
    s["last_status"] = cur
    drops = s.setdefault("my_drops", {})
    for pk in prev:
        if pk not in cur:
            drops[pk] = TODAY.isoformat()
    for k in [k for k, d in drops.items() if (TODAY - date.fromisoformat(d)).days > 4 or k in cur]:
        drops.pop(k)
    if active() or FORCE:
        try:
            plan_rebuild_check(W, T, mine, s)
        except Exception as ex:
            print("plan rebuild check failed:", repr(ex)[:120])
    # ---- status alerts
    if msgs:
        print("status changes (no alert; late-scratch and the plan handle them):", "; ".join(m for _, m in msgs)[:300])
    # ---- pickup alert: a player who should almost certainly be picked up just became available (free agent now, or dropped onto waivers)
    try:
        pickup_alert(lg, W, T, s, mine, can_send)
    except Exception as ex:
        print("pickup check failed:", ex)
    try:
        roster_fix_alert(T, s, mine, can_send)
    except Exception as ex:
        print("roster fix check skipped:", repr(ex)[:90])
    try:
        plan_move_alert(lg, T, s, mine, can_send)
        plan_target_watch(lg, T, s, mine, can_send)
    except Exception as ex:
        print("plan target watch skipped:", repr(ex)[:90])
    # (injury-opportunity adds are announced by plan_move_alert, which names the injury news as the reason: one message per move)
    # ---- lineup checks: planned starter benched / OUT player in the lineup before the first tip, and the per-game late-scratch watch with a named swap
    first_tip_check(W, T, mine, off, s, can_send)
    late_scratch_watch(W, T, mine, off, s, can_send, plan_by_id)
    save_state(s)


def p_taken(ll):
    """chance another team takes a just-available player within about two days (2024-26, players dropped after 4+ recent games, by his points per game before the drop): 34+ = 25%, 27-34 = 16%, lower = 12%"""
    return 0.25 if (ll or 0) >= 33 else (0.16 if (ll or 0) >= 27 else 0.12)


def _backup_text(bk, n=2):
    return ", then ".join(f"{b['name']} ({b['net']:+.0f})" for b in bk[:n])


def plan_guidance(T, pid):
    """what the weekly plan says about this player, so the alert and the plan agree. Returns (text, short tag) or ('', '').
    The plan values a free agent day by day against the lineup you will actually have, so a great player can still be worth little today (slots are full, or the player you would drop still has games).
    Add now vs wait: waiting keeps the dropped player's games but risks losing him to another team; the expected value of waiting counts the plan's backup if he is gone."""
    sq = next((m for m in (T.get("sequence") or []) if str(m["add"]["id"]) == str(pid)), None)
    pc = (T.get("pickup_checks") or {}).get(str(pid))
    now = (pc or {}).get("now")
    if sq:
        day = datetime.fromisoformat(sq["on"]).strftime("%a")
        drop = (sq.get("drop") or {}).get("name") or "an open spot"
        P = p_taken(sq["add"].get("ll") or sq["add"].get("level"))
        bk = sq.get("backups") or []
        alt = f" If he is gone: {_backup_text(bk)}." if bk else ""
        what = (f"PLAN: he is a KEEP, and the best timing is {day} (drop {drop} then, after his games)" if sq.get("keep") else f"PLAN: use him as a one-day rental on {day} (drop {drop} then)") + f", about {sq['gain']:+.0f} net."
        if now:
            ev_wait = (1 - P) * sq["gain"] + P * max(0.0, bk[0]["net"] if bk else 0.0)
            if now["net"] > ev_wait:
                return what + f" But adding him NOW (drop {now['drop']}) is worth more on average ({now['net']:+.0f} now vs {ev_wait:+.0f} waiting, counting a {P:.0%} chance he is taken first): add him now.{alt}", "plan: add now"
            return what + f" Adding him now (drop {now['drop']}) would be {now['net']:+.0f}; waiting averages {ev_wait:+.0f} even after a {P:.0%} chance he is taken first. Wait.{alt}", f"plan: wait for {day}"
        return what + alt, f"plan: {day}"
    if pc:
        # not in the plan. The plan's search already tried him (as a keep and as a rental, against every possible drop) alongside every other free agent, so his
        # one-at-a-time value below is information, not an instruction: the planned moves beat him once they are combined (they share days, open slots and adds).
        keep = {"kind": "keep", **{k: pc.get(k) for k in ("net", "adj", "wk", "fut", "drop", "on")}} if pc.get("drop") and pc.get("net", -1e9) > -1e8 else None
        rent = dict(pc["rent"], kind="rent") if pc.get("rent") else None
        best = rent if rent and (not keep or (rent.get("adj", rent["net"]) >= (keep.get("adj") if keep.get("adj") is not None else keep["net"]))) else keep
        if not best:
            return "", ""
        d0 = datetime.fromisoformat(best["on"]).strftime("%a")
        how = (f"keeping him from {d0}" if best["kind"] == "keep"
               else (f"a one-day rental on {d0}" if best.get("until") in (None, best["on"]) else f"a rental {d0}-{datetime.fromisoformat(best['until']).strftime('%a')}"))
        left = (T["_adds_limit"] - (T.get("adds_used") or 0)) if T.get("_adds_limit") is not None else None
        sq_n = len(T.get("sequence") or [])
        adds_note = f" The plan's {sq_n} move{'s' if sq_n != 1 else ''} use all of your remaining adds." if left is not None and left - sq_n <= 0 and sq_n else ""
        if best["net"] >= 6:
            return (f"PLAN CHECK: on his own he is worth about {best['net']:+.0f} as {how} (drop {best['drop']}{', no long-run cost' if best['kind'] == 'rent' else ', counting the rest of the season'}), "
                    f"but he is not in the plan: the planned moves are worth more once they are combined (they compete for the same days, lineup slots and adds).{adds_note} "
                    "Nothing to do; he is a fallback if a planned target is taken."), "plan: fallback only"
        fut_txt = f" and the rest of the season by {-best['fut']:+.0f}" if best.get("fut") else ""
        return f"PLAN CHECK: no help. His best use, {how} for {best['drop']}, changes this week by {best['wk']:+.0f}{fut_txt}. Skip.", "plan: not needed"
    return "", ""


def _mark_announced(T, pid, s):
    """the pickup alert already told you about this plan move (today's or tomorrow's): do not send the plan-move alert for it as well"""
    ann = s.setdefault("plan_announced", {})
    for m in (T.get("sequence") or []):
        if str(m["add"]["id"]) == str(pid) and 0 <= (date.fromisoformat(m["on"]) - TODAY).days <= 1:
            base = f"{m['add']['id']}:{m['on']}:{(m.get('drop') or {}).get('id')}"
            ann[base + (":today" if m["on"] == TODAY.isoformat() else ":eve")] = TODAY.isoformat()
            ann.setdefault(base, TODAY.isoformat())


def roster_fix_alert(T, s, mine, can_send):
    """free IR moves the plan wants (move an injured player into an open IR slot, activate a healthy one): checked against ESPN right now so a move you already made is not repeated"""
    if not can_send or (in_quiet() and not FORCE):
        return
    sent = s.setdefault("rosterfix_sent", {})
    for k in [k for k, d in sent.items() if (TODAY - date.fromisoformat(d)).days > 3]:
        sent.pop(k)
    slot = {int(p.playerId): p.lineupSlot for p in mine.roster}
    for m in (T.get("ir_moves") or []):
        pid = int(m["id"]) if m.get("id") is not None else None
        if pid is None or pid not in slot:
            continue
        in_ir = slot[pid] == "IR"
        if (m["action"] == "to_ir" and in_ir) or (m["action"] != "to_ir" and not in_ir):
            continue                                             # already done in ESPN
        key = f"{pid}:{m['action']}"
        if key in sent:
            continue
        if m["action"] == "to_ir":
            title, body = f"Roster fix: move {m['name']} to IR", f"{m['name']} ({m.get('status', 'out')}) can go into an open IR slot. It is free and opens a roster spot for the plan's adds."
        elif m["action"] == "activate_swap":
            title, body = f"Roster fix: activate {m['name']}", f"{m['name']} is healthy but stuck in an IR slot, so he cannot play. Activate him and drop {(m.get('drop') or {}).get('name', 'your cheapest player')}."
        else:
            title, body = f"Roster fix: activate {m['name']}", f"{m['name']} is healthy but stuck in an IR slot, so he cannot play. Move him to your bench or lineup."
        if m.get("unlocks") and m["unlocks"] > 0.5:
            body += f" Worth about +{m['unlocks']:.0f} points this week."
        if push(title, body, priority=4, tags=["hospital"]):
            sent[key] = TODAY.isoformat()


def plan_move_alert(lg, T, s, mine, can_send):
    """Tells you when the weekly plan wants a pickup, so you never have to open the page to find out:
      * TODAY'S move, the first time the plan has it (one message per player, day and drop; if the plan changes the drop you get the new version);
      * TOMORROW'S move, as an evening heads-up (6pm ET on) so the add can be made first thing, or a waiver claim placed in time;
      * a REMINDER about two hours before his game if today's add still has not been made;
      * a player on waivers: the claim has to be in before the 3am run, and the message says so.
    plan_target_watch only reports a player as 'taken' after he was announced here."""
    import waivers as WV
    if not can_send:
        return
    pool = {int(p["id"]) for p in WV.available_pool(lg, 300)}
    mine_ids = {int(p.playerId) for p in mine.roster}
    ann = s.setdefault("plan_announced", {})
    for k in [k for k, d in ann.items() if (TODAY - date.fromisoformat(d)).days > 3]:
        ann.pop(k)
    tips = team_tips()
    for m in (T.get("sequence") or []):
        on = date.fromisoformat(m["on"])
        days_to = (on - TODAY).days
        pid = int(m["add"]["id"])
        if days_to not in (0, 1) or pid in mine_ids:
            continue
        if days_to == 1 and NOW.hour < 18:
            continue
        w = m["add"].get("waiver")
        if pid not in pool and not w:
            continue
        drop = (m.get("drop") or {})
        base = f"{pid}:{m['on']}:{drop.get('id')}"
        stage = "today" if days_to == 0 else "eve"
        kind = "KEEP him" if m.get("keep") else ("rental through " + datetime.fromisoformat(m["hold_through"]).strftime("%a") if m.get("hold_through") else "rental")
        bk = m.get("backups") or []
        tail = ((" Reclaim the dropped player once he clears waivers (about 40 hours)." if m.get("reclaim") else "")
                + (f" If he is taken first: {_backup_text(bk)}." if bk else "") + " Details on the Today tab.")
        who = f"{m['add']['name']} ({m['add']['team']})"
        if m["add"].get("boost", 0) >= 2.5 and m["add"].get("boost_why"):
            tail = f" Why now: {' and '.join(m['add']['boost_why'][:2])} out, so he gains about +{m['add']['boost']:.0f} pts/g." + tail
        dr = drop.get("name") or "nobody (open spot)"
        wv = f" He is on WAIVERS ({WV.when(w)}): place the claim before that run." if w and pid not in pool else ""
        if (FORCE or not in_quiet()) and base + ":" + stage not in ann and not (days_to == 1 and base + ":today" in ann) and not (days_to == 0 and base + ":eve" in ann):   # told last night: the morning reminder below covers it
            lead = "Plan move for today" if days_to == 0 else "Plan move for tomorrow (" + on.strftime("%a") + ")"
            body = f"{lead}: add {who}, drop {dr} ({kind}), about {m['gain']:+.0f} net.{wv}{tail}"
            if push(("Plan move today: add " if days_to == 0 else "Plan move tomorrow: add ") + m["add"]["name"], body, priority=4, tags=["arrows_counterclockwise"]):
                ann[base + ":" + stage] = TODAY.isoformat()
                ann.setdefault(base, TODAY.isoformat())
        # reminder: today's add still not made and his game is about two hours away
        tip = tips.get(canon(m["add"]["team"]))
        if days_to == 0 and pid in pool and tip is not None and base + ":rem" not in ann:
            mins = (tip - NOW).total_seconds() / 60
            if 0 < mins <= 120:
                if push(f"Reminder: add {m['add']['name']} before {tip.strftime('%I:%M%p').lstrip('0').lower()}", f"The plan's move for today is still open: add {who}, drop {dr} ({kind}). His game tips off in about {int(mins)} minutes.{tail}", priority=5, tags=["alarm_clock"]):
                    ann[base + ":rem"] = TODAY.isoformat()


def plan_target_watch(lg, T, s, mine, can_send):
    """a player the weekly plan is counting on (an add in the next two days) has been taken by another team: say so right away with the plan's backups. The plan itself re-optimizes at its next
    refresh (5:37am, 3:07pm, 6:07pm ET), or when you run it by hand."""
    import waivers as WV
    if not can_send:
        return
    pool = {int(p["id"]) for p in WV.available_pool(lg, 300)}
    mine_ids = {int(p.playerId) for p in mine.roster}
    seen = s.setdefault("target_gone", {})
    for k in [k for k, d in seen.items() if (TODAY - date.fromisoformat(d)).days > 3]:
        seen.pop(k)
    for m in (T.get("sequence") or []):
        on = date.fromisoformat(m["on"])
        pid = int(m["add"]["id"])
        if not (0 <= (on - TODAY).days <= 2) or pid in pool or pid in mine_ids:
            continue
        key = f"{pid}:{m['on']}"
        if key in seen:
            continue
        request_rebuild(s, f"plan target taken: {m['add']['name']}")     # no message: the rebuilt plan's next move comes as a plan-move alert
        seen[key] = TODAY.isoformat()


def pickup_alert(lg, W, T, s, mine, can_send):
    """Phone alert when a player you should almost certainly pick up becomes available, and whether that means ADD NOW (free agent) or CLAIM BEFORE ~3AM ET (just dropped onto
    waivers, 24h+). "Almost certainly" = either a keeper asset that would crack your top 5 (the old dynasty alert), or a player ranked in the top 150 for the rest of this season
    who projects at least 3 points per game above your cheapest droppable player. Each player alerts once per availability (a second, shorter alert only if a player we flagged on
    waivers clears with nobody claiming him). State in alerts/state.json: avail_prev (who was available last run) and pickup_sent."""
    import waivers as WV
    from espn_api.basketball.constant import PRO_TEAM_MAP
    pool = WV.available_pool(lg, 300)
    wm = WV.waiver_map(lg)
    T["_adds_limit"] = (W.get("matchup") or {}).get("adds_limit")
    hub = load("hub_data.json")
    by_hub = {p["id"]: p for p in hub["players"]}
    e2n = {int(k): int(v) for k, v in json.load(open(Path(__file__).resolve().parent / "espn_id_map.json")).items()}
    rd = {str(p["id"]): p for p in (load("redraft_data.json") or {"players": []})["players"]}

    def a5(espn_id, name):
        nba = e2n.get(int(espn_id))
        hp = by_hub.get(f"c{nba}") or by_hub.get(f"p{nba}") if nba else None
        if hp is None:
            hp = next((q for q in hub["players"] if norm(q["player"]) == norm(name)), None)
        return (hp["asset_k"][5] if hp and hp.get("asset_k") and len(hp["asset_k"]) > 5 else 0.0) or 0.0
    my_assets = sorted([a5(p.playerId, p.name) for p in mine.roster], reverse=True)
    bar = my_assets[4] if len(my_assets) >= 5 else 25.0
    droppable = sorted((p["level"], p["name"]) for p in T["roster"] if not p.get("protected") and not p.get("ir"))
    worst = droppable[0] if droppable else None
    cur = {}
    for p in pool:
        pid = str(p["id"])
        v = a5(p["id"], p["name"])
        r = rd.get(pid)
        dyn = v >= max(bar, 10)
        good = bool(r and r.get("rank") and r["rank"] <= 150 and worst and r["ppg"] >= worst[0] + 3)
        if dyn or good:
            cur[pid] = {"name": p["name"], "status": p["status"], "asset": round(v, 1), "ppg": (r or {}).get("ppg"), "rank": (r or {}).get("rank"),
                        "team": canon(PRO_TEAM_MAP.get(p["pro_team_id"]) or ""), "dyn": dyn, "good": good}
    prev = s.get("avail_prev")
    sent = s.setdefault("pickup_sent", {})
    for k in [k for k, d in sent.items() if (TODAY - date.fromisoformat(d)).days > 7]:
        sent.pop(k)
    new_prev = {k: v["status"] for k, v in cur.items()}
    if prev is None:
        s["avail_prev"] = new_prev                               # first run: remember who is available, alert on nothing
        return
    can = can_send and (FORCE or not in_quiet())
    events = []
    my_drops = s.get("my_drops") or {}
    unseen = [v["name"] for pid, v in cur.items() if prev.get(pid) is None and pid not in my_drops and str(pid) not in (T.get("pickup_checks") or {})
              and str(pid) not in {str(m["add"]["id"]) for m in (T.get("sequence") or [])}]
    if unseen and can_send:
        request_rebuild(s, "new free agent(s) to evaluate: " + ", ".join(unseen[:4]))       # the plan decides; a worthwhile one comes back as a plan-move alert
    in_plan = {str(m["add"]["id"]) for m in (T.get("sequence") or [])}
    for pid, v in cur.items():
        was = prev.get(pid)
        if pid in my_drops and pid not in in_plan:
            continue                                             # the player you just dropped: the plan decides if and when he comes back (plan move alert)
        if not v["dyn"]:
            continue                                             # only keeper-level players (would be one of your top 5 keepers) get their own message
        if was is None:
            events.append((pid, v, "new"))
        elif was == "WAIVERS" and v["status"] == "FREEAGENT" and f"{pid}:WAIVERS" in sent:
            events.append((pid, v, "cleared"))
    if events and not can:
        for pid, v, _ in events:
            new_prev.pop(pid, None)                              # keep them 'new' so they alert when alerts are allowed again
    s["avail_prev"] = new_prev
    if not events or not can:
        return
    events.sort(key=lambda e: -((e[1]["ppg"] or 0) + e[1]["asset"]))
    pri, n = WV.my_priority(lg, T["id"])
    lines, recheck = [], []
    for pid, v, kind in events[:3]:
        val = f"about {v['ppg']:.0f} pts/g, redraft rank #{v['rank']}" if v["ppg"] else "no rest-of-season projection yet"
        extra = f"; keeper asset {v['asset']:.0f} vs your #5 at {bar:.0f}" if v["dyn"] else ""
        drop = f" Your cheapest drop is {worst[1]} ({worst[0]:.0f} pts/g)." if worst else ""
        if v["status"] == "FREEAGENT":
            head = f"{v['name']} ({v['team']}) cleared waivers and nobody claimed him." if kind == "cleared" else f"{v['name']} ({v['team']}) just became a FREE AGENT."
            pg, ptag = plan_guidance(T, pid)
            _mark_announced(T, pid, s)
            if not pg:
                recheck.append(v["name"])
                pg = "The plan has not seen him yet and is re-checking now: if the new plan wants him you get a Plan move alert (on the day of the move, or the evening before); otherwise skip him."
            lines.append((ptag, v["name"], f"{head} {val}{extra}. {pg}"))
        else:
            w = wm.get(int(pid))
            when = WV.when(w) if w else "the next 3am ET run"
            prio = f" Your waiver priority is {pri} of {n} (1 = first), so a team ahead of you could win him." if pri else " A team ahead of you in waiver order could win him."
            try:
                intel = load("waiver_intel.json")
                ranks = WV.all_priorities(lg)
                pw, ahead = WV.win_chance(intel, v["ppg"] or (35.0 if v["dyn"] else 28.0), T["id"], ranks)
                if pw is not None:
                    names = {t.team_id: t.team_abbrev for t in lg.teams}
                    top = ", ".join(f"{names.get(t, t)} {q:.0%}" for t, q in ahead[:2] if q >= 0.03)
                    prio = f" Odds you win the claim: about {pw:.0%} (priority {pri} of {n}" + (f"; likeliest rivals ahead of you: {top}" if top else "; no team ahead of you has claimed players like him often") + ")."
            except Exception as ex:
                print("claim-odds estimate failed:", repr(ex)[:80])
            pg, ptag = plan_guidance(T, pid)
            _mark_announced(T, pid, s)
            if not pg:
                recheck.append(v["name"])
                pg = "The plan has not seen him yet and is re-checking now: if the new plan wants him you get a Plan move alert (on the day of the move, or the evening before); otherwise skip him."
            lines.append((ptag, v["name"], f"{v['name']} ({v['team']}) was just dropped and is ON WAIVERS. {val}{extra}. A claim is processed at {when}.{prio} {pg}"))
        sent[f"{pid}:{v['status']}"] = TODAY.isoformat()
    if recheck and not (s.get("plan_rebuild") or {}).get("why", "").startswith("new free agent"):
        request_rebuild(s, "new free agent(s) to evaluate: " + ", ".join(recheck))
    act = [x for x in lines if x[0] and x[0] not in ("plan: not needed", "plan: fallback only")]
    lines.sort(key=lambda x: (0 if x in act else (1 if not x[0] else 2)))
    first = lines[0]
    title = f"Keeper-level free agent: {first[1]}" + (f" ({first[0].replace('plan: ', '')})" if first[0] else "")
    push(title, "\n".join(x[2] for x in lines) + "\nHe would be one of your top 5 keepers. This-season value is in the Today tab; the keeper call is yours.", priority=4, tags=["star"])
    _ledger("pickup", names=[e[1]["name"] for e in events[:3]], kinds=[e[2] for e in events[:3]])


def first_tip_check(W, T, mine, off, s, can_send):
    """lineup check in the 90 minutes before the first tip of the day: an OUT player in the lineup, or a planned starter left on the bench"""
    tip = first_tip(W, T)
    if tip and can_send and tip - timedelta(minutes=90) <= NOW < tip:
        made = {int(m["add"]["id"]) for m in (T.get("sequence") or [])} & {int(p.playerId) for p in mine.roster}
        days_src = T.get("days_after") if (made and T.get("days_after")) else T["days"]      # you have made the plan's moves: check against the lineup that assumes them
        today_plan = next((d for d in days_src if d["date"] == TODAY.isoformat()), None)
        probs = []
        for p in mine.roster:
            slot = p.lineupSlot
            starting = slot not in ("BE", "IR")
            is_out = (p.injuryStatus in ("OUT", "INJURY_RESERVE", "SUSPENSION")) or off.get(norm(p.name).replace(" ", "")) == "Out"
            if starting and is_out:
                probs.append(f"{p.name} is OUT but in your lineup ({slot}).")
        if today_plan and not today_plan.get("locked"):
            benched = {p.playerId: p for p in mine.roster if p.lineupSlot in ("BE", "IR")}
            for x in today_plan["start"]:
                q = benched.get(x["id"])
                if q is not None and q.injuryStatus not in ("OUT", "INJURY_RESERVE") and off.get(norm(q.name).replace(" ", "")) != "Out":
                    probs.append(f"The plan starts {x['name']} today but he is on your {'IR' if q.lineupSlot == 'IR' else 'bench'}.")
        if probs:
            h = hashlib.md5("|".join(sorted(probs)).encode()).hexdigest()[:10]
            key = f"{TODAY.isoformat()}:{h}"
            if s.get("sent", {}).get("lineup") != key:
                push(f"Lineup check: tip-off at {tip.strftime('%I:%M%p').lstrip('0').lower()} ET", "\n".join(probs[:6]) + "\nFix in ESPN before the first game locks.", priority=4, tags=["rotating_light"])
                s.setdefault("sent", {})["lineup"] = key


# ============================================================ late-scratch watch (per game, names the swap)
TIPS_CACHE = {}


def team_tips():
    """canon NBA team -> tip-off datetime (ET) of today's game"""
    sched = load("nba_schedule.json") or {}
    out = {}
    for a, h, t in sched.get("games", {}).get(TODAY.isoformat(), []):
        try:
            hh, mm = int(t[:2]), int(t[3:5])
        except (TypeError, ValueError):
            continue
        dt = datetime(TODAY.year, TODAY.month, TODAY.day, hh, mm, tzinfo=ZoneInfo("UTC"))
        if hh < 10:
            dt += timedelta(days=1)                            # 00:00-09:59Z is the evening of the same ET date
        dt = dt.astimezone(ET)
        out[canon(a)] = dt
        out[canon(h)] = dt
    return out


def swap_options(out_p, roster, tips, now):
    """ways to cover a starter who is out, best first. roster: dicts with id, name, slot, elig (slots he may fill), team, out, p (chance he plays), level.
    A bench player must have a game that has not tipped off yet. Direct: he is eligible for the open slot. Chain: another unlocked starter slides into the
    open slot and the bench player takes that starter's slot. Returns [(expected pts, bench player, slot he takes, starter who slides or None)]."""
    S = out_p["slot"]
    starters = [q for q in roster if q["slot"] not in ("BE", "IR")]
    bench = [q for q in roster if q["slot"] == "BE" and not q["out"] and tips.get(q["team"]) is not None and tips[q["team"]] > now]
    opts = []
    for b in bench:
        val = b["level"] * b["p"]
        if S in b["elig"]:
            opts.append((val, b, S, None))
            continue
        for y in starters:
            if y is out_p or y["out"]:
                continue
            ty = tips.get(y["team"])
            if ty is not None and ty <= now:                      # already locked: cannot be moved
                continue
            if S in y["elig"] and y["slot"] in b["elig"]:
                opts.append((val - 0.01, b, y["slot"], y))
                break
    opts.sort(key=lambda o: -o[0])
    return opts


def fa_options(out_p, fa_pool, tips, now, n=2):
    """best free agents who can fill the slot and whose game has not started (from the weekly plan's free-agent pool)"""
    S = out_p["slot"]
    res = []
    for f in fa_pool or []:
        t = tips.get(canon(f.get("team") or ""))
        if f.get("waiver"):
            continue                                              # on waivers: can only be claimed (about 3am ET), so he cannot fill a slot tonight
        if t is None or t <= now or f.get("status") in ("OUT", "INJURY_RESERVE", "SUSPENSION"):
            continue
        sl = f.get("slots") or []
        if S in sl:
            res.append((f.get("tonight") or f.get("level", 0), f, t))
    res.sort(key=lambda r: -r[0])
    return res[:n]


def _fmt_t(dt):
    return dt.strftime("%I:%M%p").lstrip("0").lower()


def scratch_message(out_p, tip, now, opts, fas, reason, tips):
    mins = int((tip - now).total_seconds() // 60)
    lines = [f"{out_p['name']} ({out_p['slot']} slot) is {reason}. His game tips at {_fmt_t(tip)} ET ({mins} min from now)."]
    if opts:
        val, b, slot, y = opts[0]
        bt = _fmt_t(tips[b["team"]])
        if y is None:
            lines.append(f"BEST SWAP: start {b['name']} in the {slot} slot (about {val:.0f} expected pts, game at {bt}).")
        else:
            lines.append(f"BEST SWAP: move {y['name']} into the {out_p['slot']} slot and start {b['name']} at {slot} (about {val:.0f} expected pts, game at {bt}).")
        for v2, b2, s2, y2 in opts[1:3]:
            lines.append(f"Next best: {b2['name']} ({v2:.0f} pts)" + (f", via {y2['name']} sliding over" if y2 else "") + ".")
    else:
        lines.append("No bench player who plays tonight can fill that slot.")
        if fas:
            f = fas[0]
            lines.append(f"Free-agent option: {f[1]['name']} ({f[1]['team']}, about {(f[1].get('tonight') or f[1]['level']):.0f} pts tonight, tips {_fmt_t(f[2])}). He is a true free agent (nobody dropped him recently), so the add takes effect immediately; it costs one of your adds and a drop.")
        else:
            lines.append("No free agent fits either. Leaving the slot empty costs only that player's game.")
    lines.append("Only matters if you are under your games cap.")
    return "\n".join(lines)


def late_scratch_watch(W, T, mine, off, s, can_send, plan_by_id):
    """for every starter whose game has not tipped off and who is ruled out (official report Out/Doubtful or ESPN OUT/IR/suspended): send the swap now, and once
    more 35 minutes before his tip if the lineup still has him in. One alert per (player, day, stage); state in alerts/state.json under `lu`."""
    if not can_send:
        return
    tips = team_tips()
    roster = []
    for p in mine.roster:
        k = norm(p.name).replace(" ", "")
        o = off.get(k)
        status = o or p.injuryStatus or "ACTIVE"
        pretend = p.name.lower() in PRETEND_OUT
        out = pretend or (p.injuryStatus in ("OUT", "INJURY_RESERVE", "SUSPENSION")) or o in ("Out", "Doubtful")
        reason = "(test) pretend-out" if pretend else ("ruled out in the official NBA report" if o in ("Out", "Doubtful") else f"listed {p.injuryStatus} by ESPN")
        roster.append({"id": p.playerId, "name": p.name, "slot": p.lineupSlot, "elig": list(getattr(p, "eligibleSlots", []) or []), "team": canon(p.proTeam or ""),
                       "out": out, "reason": reason if out else "", "p": 0.0 if out else P_STATUS.get(status, 0.94), "level": plan_by_id.get(p.playerId, {}).get("tonight") or plan_by_id.get(p.playerId, {}).get("level", 20.0)})
    lu = s.setdefault("lu", {})
    for k in [k for k in lu if not k.startswith(TODAY.isoformat())]:
        lu.pop(k)
    for q in roster:
        if q["slot"] in ("BE", "IR") or not q["out"]:
            continue
        tip = tips.get(q["team"])
        if tip is None or tip <= NOW:
            continue                                              # no game today, or his game has started (locked)
        mins = (tip - NOW).total_seconds() / 60
        if in_quiet() and not FORCE and mins > 120:
            continue
        key = f"{TODAY.isoformat()}:{q['id']}"
        stage = lu.get(key, 0)
        want = 1 if stage < 1 else (2 if (stage < 2 and mins <= 35) else 0)
        if not want:
            continue
        opts = swap_options(q, roster, tips, NOW)
        fas = [] if opts else fa_options(q, W.get("fa_pool"), tips, NOW)
        body = scratch_message(q, tip, NOW, opts, fas, q["reason"], tips)
        title = ("REMINDER: " if want == 2 else "Late scratch: ") + q["name"] + " is out" + (" (still in your lineup)" if want == 2 else "")
        if push(title, body, priority=5 if mins <= 60 else 4, tags=["rotating_light"]):
            lu[key] = want
            _ledger("late_scratch", player=q["name"], stage=want, mins_to_tip=round(mins), swap=(opts[0][1]["name"] if opts else None))


def run_lineup():
    """fast path for the 5-minute timer: exits at once unless one of your teams tips off within 4 hours"""
    W = load("week_plan.json")
    if not W:
        print("no plan file")
        return
    T = my_team(W)
    tips = team_tips()
    mine_teams = {canon(p["team"]) for p in T["roster"] if not p["ir"]}
    soon = [t for tm, t in tips.items() if tm in mine_teams and NOW < t <= NOW + timedelta(hours=4)]
    from espn_api.basketball import League
    import config
    s = state()
    lg = League(league_id=config.LEAGUE_ID, year=W["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    mine = next(t for t in lg.teams if t.team_abbrev == W["my_abbrev"])
    if active() or FORCE:                     # every 5 minutes: a roster move (or a skipped scheduled build) starts a plan rebuild
        before_rb = json.dumps(s.get("plan_rebuild"), sort_keys=True)
        try:
            plan_rebuild_check(W, T, mine, s)
        except Exception as ex:
            print("plan rebuild check failed:", repr(ex)[:120])
        if json.dumps(s.get("plan_rebuild"), sort_keys=True) != before_rb:
            save_state(s)
    if not soon and not FORCE:
        print("no game of yours tips within 4 hours")
        return
    plan_by_id = {p["id"]: p for p in T["roster"]}
    off = official_today({norm(p.name).replace(" ", "") for p in mine.roster})
    before = json.dumps(s, sort_keys=True)
    can_send = active() or FORCE
    first_tip_check(W, T, mine, off, s, can_send)
    late_scratch_watch(W, T, mine, off, s, can_send, plan_by_id)
    if json.dumps(s, sort_keys=True) != before:
        save_state(s)


if __name__ == "__main__":
    mode = ARGS[0] if ARGS else "check"
    if mode == "test":
        push("Fantasy Hub test", "If you can read this on your phone, alerts are working.", priority=3, tags=["white_check_mark"])
    elif mode == "lineup":
        run_lineup()
    elif mode == "daily":
        run_heartbeat()                                     # the daily brief is retired (only actionable alerts); this slot keeps the PC-refresh check
    elif mode == "weekly":
        run_digest(mode)
    else:
        run_check()
