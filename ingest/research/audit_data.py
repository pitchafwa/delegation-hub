"""Data sanity audit across the dashboard JSON outputs (2026-10-25 bug audit). Prints findings, changes nothing.
Run from ingest/:  .venv/Scripts/python research/audit_data.py"""
import json, math, sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
H = Path(__file__).resolve().parent.parent.parent / "dashboard"
L = lambda n: json.load(open(H / n, encoding="utf-8"))
issues = []
def flag(area, msg): issues.append((area, msg))
bad = lambda v: v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))

# ---------- freshness
for f in ["week_plan.json", "hub_data.json", "redraft_data.json", "league_rosters.json", "playoff_odds.json", "schedule_plan.json", "team_usage.json", "trade_market.json", "trade_context.json", "trade_ideas_v2.json", "player_news.json", "nba_schedule.json", "season_calendar.json", "lineup_preview.json", "odds_history.json", "waiver_intel.json", "team_profiles.json"]:
    try:
        d = L(f)
    except Exception as ex:
        flag("files", f"{f}: cannot load ({ex!r:.60})"); continue
    g = d.get("generated") if isinstance(d, dict) else None
    print(f"{f:24s} generated {g}")

W = L("week_plan.json"); HB = L("hub_data.json"); RD = L("redraft_data.json"); LR = L("league_rosters.json"); PO = L("playoff_odds.json"); SP = L("schedule_plan.json")
M = W["matchup"]; me = W["my_abbrev"]
# ---------- hub
names = Counter(p["player"] for p in HB["players"])
for n, c in names.items():
    if c > 1: flag("hub", f"duplicate player {n} x{c}")
for p in HB["players"]:
    for k in ("year0_ppg", "peak_ppg", "age"):
        if bad(p.get(k)) and p["kind"] == "current": flag("hub", f"{p['player']}: {k} missing")
    t = p.get("trajectory") or []
    if t and (min(t) < 0 or max(t) > 95): flag("hub", f"{p['player']}: trajectory out of range {t[:3]}")
    if p["kind"] == "current" and p.get("season_ppg") is not None and p.get("year0_ppg") and abs(p["season_ppg"] - p["year0_ppg"]) > 15:
        flag("hub", f"{p['player']}: season {p['season_ppg']} vs year0 {p['year0_ppg']}")
# ---------- redraft vs ESPN outliers
rd = {p["name"]: p for p in RD["players"]}
for p in RD["players"]:
    if p.get("espn_ppg") and p["espn_ppg"] > 15 and abs(p["ppg"] - p["espn_ppg"]) > 0.35 * p["espn_ppg"] and (p.get("n_season") or 0) < 15:
        flag("redraft-gap", f"{p['name']} ({p['team']}): ours {p['ppg']} vs ESPN {p['espn_ppg']} (n={p.get('n_season')}, l10 {p.get('l10')})")
    if p["exp_gp"] > p["g_rem"] + 0.01: flag("redraft", f"{p['name']}: exp_gp {p['exp_gp']} > games left {p['g_rem']}")
    if p["ppg"] < 0 or p["exp_gp"] < 0: flag("redraft", f"{p['name']}: negative ppg/gp")
    if p.get("status") in ("OUT", "INJURY_RESERVE") and not p.get("games_out") and p["exp_gp"] >= p["g_rem"] * 0.95:
        flag("redraft", f"{p['name']}: status {p['status']} but no games out")
# hub vs redraft season number
hbn = {p["player"]: p for p in HB["players"]}
for n, p in rd.items():
    h = hbn.get(n)
    if h and h.get("season_ppg") and p.get("our") and abs(h["season_ppg"] - p["ppg"]) > 3:
        flag("hub-vs-redraft", f"{n}: hub season {h['season_ppg']} vs redraft {p['ppg']}")
# ---------- league rosters vs plan rosters
lr = {t["abbrev"]: {x.get("espn_id") for x in t["roster"]} for t in LR["teams"]}
for t in W["teams"]:
    ids = {p["id"] for p in t["roster"]}
    a = lr.get(t["abbrev"], set())
    if ids != a:
        flag("rosters", f"{t['abbrev']}: plan has {len(ids - a)} not in league_rosters, missing {len(a - ids)} (league_rosters {LR.get('generated')} vs plan {W['generated']})")
# ---------- weekly plan sanity
nba = M.get("nba_games") or {}
slots_n = 10
for t in W["teams"]:
    ros = {p["id"]: p for p in t["roster"]}
    for days_key in ("days", "days_after"):
        for d in t.get(days_key) or []:
            st = d.get("start") or []
            if len(st) > slots_n: flag("plan", f"{t['abbrev']} {days_key} {d['date']}: {len(st)} starters")
            ids = [x["id"] for x in st]
            if len(ids) != len(set(ids)): flag("plan", f"{t['abbrev']} {days_key} {d['date']}: duplicate starter")
            playing = set(nba.get(d["date"], []))
            for x in st:
                p = ros.get(x["id"])
                if p and playing and p["team"] not in playing and x.get("p", 1) > 0:
                    flag("plan", f"{t['abbrev']} {days_key} {d['date']}: {x['name']} ({p['team']}) starts with no game")
                if p and p.get("ir") and days_key == "days": flag("plan", f"{t['abbrev']} {d['date']}: IR player {x['name']} starting")
                if p and p.get("status") in ("OUT",) and d["date"] == M["planned_days"][0] and x.get("p", 0) > 0.05:
                    flag("plan", f"{t['abbrev']} {d['date']}: OUT player {x['name']} starting p={x.get('p')}")
            if d.get("starts_after", 0) > M["cap"] + 10: flag("plan", f"{t['abbrev']} {d['date']}: starts_after {d.get('starts_after')} way over cap")
    au = t.get("adds_used") or 0
    if au > M["adds_limit"]: flag("plan", f"{t['abbrev']}: adds_used {au} > limit")
    if t["abbrev"] == me:
        seq = t.get("sequence") or []
        if len(seq) > max(0, M["adds_limit"] - au): flag("plan", f"{me}: {len(seq)} planned adds but {M['adds_limit'] - au} left")
        fa = {p["id"] for p in W.get("fa_pool", [])}
        rostered = set().union(*lr.values())
        for m in seq:
            if m["add"]["id"] in rostered: flag("plan", f"{me}: planned add {m['add']['name']} is on a roster")
            if m.get("drop") and m["drop"]["id"] not in ros and m["drop"]["id"] not in {x['add']['id'] for x in seq}: flag("plan", f"{me}: planned drop {m['drop']['name']} not on roster")
    for p in t["roster"]:
        if bad(p.get("level")) or p["level"] < 0 or p["level"] > 90: flag("plan", f"{t['abbrev']}: {p['name']} level {p.get('level')}")
        if p["level"] == 0 and p["status"] == "ACTIVE" and not p["ir"]: flag("plan-level0", f"{t['abbrev']}: {p['name']} healthy with level 0 (src {p.get('src')})")
        if p.get("espn_level") and p["level"] and p["status"] == "ACTIVE" and abs(p["level"] - p["espn_level"]) > max(12, 0.4 * p["espn_level"]):
            flag("plan-vs-espn", f"{t['abbrev']}: {p['name']} level {p['level']} vs ESPN {p['espn_level']} (model {p.get('model_level')}, src {p.get('src')})")
for p in W.get("fa_pool", [])[:60]:
    if p.get("espn_level") and p.get("level") and abs(p["level"] - p["espn_level"]) > max(12, 0.4 * p["espn_level"]):
        flag("fa-vs-espn", f"FA {p['name']} level {p['level']} vs ESPN {p['espn_level']} (model {p.get('model_level')})")
# ---------- playoff odds
tm = PO["teams"]
for k, tot in (("made", 6), ("bye", 2), ("title", 1)):
    s = sum(x.get(k, 0) for x in tm)
    if abs(s - tot) > 0.02: flag("odds", f"sum of {k} = {s:.3f} (expected {tot})")
for x in tm:
    if x.get("seed") and abs(sum(x["seed"]) + (1 - x["made"]) - 1) > 0.02: flag("odds", f"{x['abbrev']}: seeds+out = {sum(x['seed']) + 1 - x['made']:.3f}")
    if not (0 <= x["title"] <= x.get("made", 1) + 1e-9): flag("odds", f"{x['abbrev']}: title > made")
# ---------- team usage
TU = L("team_usage.json")
for k, v in (TU.get("eff") or {}).items():
    if not 0.6 <= v["e"] <= 1.0: flag("usage", f"team {k} eff {v['e']}")
# ---------- schedule plan
if SP.get("current_week") != M["id"]: flag("schedule", f"schedule_plan current_week {SP.get('current_week')} vs plan matchup {M['id']}")
# ---------- calendar
CAL = L("season_calendar.json")
ids = [p["id"] for p in CAL["periods"]]
if ids != sorted(ids): flag("calendar", "periods out of order")
for a, b in zip(CAL["periods"], CAL["periods"][1:]):
    if date.fromisoformat(a["end"]) >= date.fromisoformat(b["start"]): flag("calendar", f"period {a['id']} overlaps {b['id']}")
by = defaultdict(list)
for a, m in issues: by[a].append(m)
print(f"\n{len(issues)} findings")
for a, ms in by.items():
    print(f"\n[{a}] {len(ms)}")
    for m in ms[:25]: print("  -", m)
