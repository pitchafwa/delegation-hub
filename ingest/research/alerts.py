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
        elif len(g) > 1 and g[0] >= max(g) - 5:
            timing = " (add now" + (f"; waiting a day costs {g[0] - g[1]:.0f})" if g[0] - g[1] >= 3 else ")")
        elif g:
            timing = f" (best on {datetime.fromisoformat(by[g.index(max(g))]['date']).strftime('%a')})"
        drop = f", drop {m['drop']['name']}" if m.get("drop") else " (open roster spot)"
        out.append(f"Add {m['add']['name']} ({m['add']['team']}){drop}: +{m['gain']:.0f}{timing}")
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
        SP = load("schedule_plan.json")
        if SP:
            me = next((t for t in SP["teams"] if t["abbrev"] == W["my_abbrev"]), None)
            if me:
                flags = [f"{p['name']} ({p['games'][SP['current_week'] - 1]} games)" for p in me["players"] if "light week soon" in p["flags"] and not p["ir"]][:4]
                if flags:
                    lines.append("Light weeks ahead: " + ", ".join(flags))
            st = SP["playoffs"]["streamers_next_week"][:3]
            if st:
                lines.append("Streamers this week: " + ", ".join(f"{s['name']} ({s['games']}g)" for s in st))
    dy = [x for x in (T.get("dynasty_adds") or []) if x.get("new") and x["would_rank"] <= 5][:1]
    if dy:
        lines.append(f"Dynasty stash: {dy[0]['name']} ({dy[0]['age']:.0f}yo) would be your #{dy[0]['would_rank']} keeper asset.")
    lines.append(f"(plan built {gen.strftime('%a %I:%M%p').lstrip('0')} ET)")
    title = ("Week ahead: " if kind == "weekly" else "Today: ") + ("moves to make" if mv else "no moves needed")
    return title, "\n".join(lines)


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
    # ---- status alerts
    if msgs and can_send and not first_run and (FORCE or not in_quiet()):
        msgs.sort(reverse=True)
        worst = msgs[0][0]
        push("Injury update: " + msgs[0][1].split(":")[0], "\n".join(m for _, m in msgs) + "\nOpen This week for replacements.", priority=4 if worst >= 20 else 3, tags=["warning"])
    # ---- pickup alert: a player who should almost certainly be picked up just became available (free agent now, or dropped onto waivers)
    try:
        pickup_alert(lg, W, T, s, mine, can_send)
    except Exception as ex:
        print("pickup check failed:", ex)
    # ---- injury beneficiaries: only when the weekly plan itself recommends the add (it names the drop and the net gain)
    if can_send and (FORCE or not in_quiet()):
        sent_b = s.setdefault("benef_sent", {})
        for m in (T.get("sequence") or []):
            add = m["add"]
            if m.get("on") and m["on"] > TODAY.isoformat():
                continue                                           # a later-day step of the stream plan: not an injury-opportunity add for today
            fresh = [n for n in (add.get("boost_why") or []) if next((o["absent"]["streak"] for o in (W.get("opportunities") or []) if o["absent"]["name"] == n), 99) < 8]
            boost_pts = add.get("boost", 0) * len(add.get("games", []))
            # only NEWS (someone recently ruled out) and only when the boost is a real part of why the plan wants him
            if add.get("boost", 0) < 2.5 or not fresh or boost_pts < 0.25 * m["week_gain"] or m["gain"] < 15 or not m.get("drop"):
                continue
            key = f"{TODAY.isoformat()}:{add['id']}"
            if key in sent_b:
                continue
            why = " and ".join(fresh[:2])
            by = m.get("by_day") or []
            g0 = by[0]["gain"] if by else m["gain"]
            wait = f" Waiting a day costs {by[0]['gain'] - by[1]['gain']:.0f}." if len(by) > 1 and by[0]["gain"] - by[1]["gain"] >= 3 else ""
            push("Injury opportunity: add " + add["name"], f"{why} out: {add['name']} ({add['team']}) gains about +{add['boost']:.0f} pts/g. Plan: add him, drop {m['drop']['name']}, net +{g0:.0f} this week.{wait}" + chr(10) + "The other injuries are in the This week tab.", priority=4, tags=["chart_with_upwards_trend"])
            sent_b[key] = 1
        for k in [k for k in sent_b if not k.startswith(TODAY.isoformat())]:
            sent_b.pop(k)
    # ---- lineup checks: planned starter benched / OUT player in the lineup before the first tip, and the per-game late-scratch watch with a named swap
    first_tip_check(W, T, mine, off, s, can_send)
    late_scratch_watch(W, T, mine, off, s, can_send, plan_by_id)
    save_state(s)


def pickup_alert(lg, W, T, s, mine, can_send):
    """Phone alert when a player you should almost certainly pick up becomes available, and whether that means ADD NOW (free agent) or CLAIM BEFORE ~3AM ET (just dropped onto
    waivers, 24h+). "Almost certainly" = either a keeper asset that would crack your top 5 (the old dynasty alert), or a player ranked in the top 150 for the rest of this season
    who projects at least 3 points per game above your cheapest droppable player. Each player alerts once per availability (a second, shorter alert only if a player we flagged on
    waivers clears with nobody claiming him). State in alerts/state.json: avail_prev (who was available last run) and pickup_sent."""
    import waivers as WV
    from espn_api.basketball.constant import PRO_TEAM_MAP
    pool = WV.available_pool(lg, 300)
    wm = WV.waiver_map(lg)
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
    for pid, v in cur.items():
        was = prev.get(pid)
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
    lines, title = [], None
    for pid, v, kind in events[:3]:
        val = f"about {v['ppg']:.0f} pts/g, redraft rank #{v['rank']}" if v["ppg"] else "no rest-of-season projection yet"
        extra = f"; keeper asset {v['asset']:.0f} vs your #5 at {bar:.0f}" if v["dyn"] else ""
        drop = f" Your cheapest drop is {worst[1]} ({worst[0]:.0f} pts/g)." if worst else ""
        if v["status"] == "FREEAGENT":
            head = f"{v['name']} ({v['team']}) cleared waivers and nobody claimed him." if kind == "cleared" else f"{v['name']} ({v['team']}) just became a FREE AGENT."
            lines.append(f"{head} {val}{extra}. ADD HIM NOW: free agents are added instantly, any hour.{drop}")
            tag = "FREE AGENT, add now"
        else:
            w = wm.get(int(pid))
            when = WV.when(w) if w else "the next 3am ET run"
            prio = f" Your waiver priority is {pri} of {n} (1 = first), so a team ahead of you could win him." if pri else " A team ahead of you in waiver order could win him."
            lines.append(f"{v['name']} ({v['team']}) was just dropped and is ON WAIVERS. {val}{extra}. CLAIM HIM BEFORE the run at {when} (he cannot be added directly).{prio} If nobody claims him he becomes a free agent after that run.{drop}")
            tag = f"ON WAIVERS, claim before {when}"
        sent[f"{pid}:{v['status']}"] = TODAY.isoformat()
        title = title or f"Pickup alert: {v['name']} ({tag})"
    push(title, "\n".join(lines) + "\nSee Suggested moves on This week for who to drop.", priority=4, tags=["star"])
    _ledger("pickup", names=[e[1]["name"] for e in events[:3]], kinds=[e[2] for e in events[:3]])


def first_tip_check(W, T, mine, off, s, can_send):
    """lineup check in the 90 minutes before the first tip of the day: an OUT player in the lineup, or a planned starter left on the bench"""
    tip = first_tip(W, T)
    if tip and can_send and tip - timedelta(minutes=90) <= NOW < tip:
        today_plan = next((d for d in T["days"] if d["date"] == TODAY.isoformat()), None)
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
            res.append((f.get("level", 0), f, t))
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
            lines.append(f"Free-agent option: {f[1]['name']} ({f[1]['team']}, about {f[1]['level']:.0f} pts/g, tips {_fmt_t(f[2])}). He is a true free agent (nobody dropped him recently), so the add takes effect immediately; it costs one of your adds and a drop.")
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
                       "out": out, "reason": reason if out else "", "p": 0.0 if out else P_STATUS.get(status, 0.94), "level": plan_by_id.get(p.playerId, {}).get("level", 20.0)})
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
    if not soon and not FORCE:
        print("no game of yours tips within 4 hours")
        return
    from espn_api.basketball import League
    import config
    s = state()
    lg = League(league_id=config.LEAGUE_ID, year=W["season"], espn_s2=config.ESPN_S2, swid=config.SWID)
    mine = next(t for t in lg.teams if t.team_abbrev == W["my_abbrev"])
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
    elif mode in ("daily", "weekly"):
        run_digest(mode)
    else:
        run_check()
