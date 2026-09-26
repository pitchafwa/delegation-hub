"""Redraft board data: every fantasy-relevant player valued for the REST OF THIS SEASON (dashboard/redraft_data.json), for the Players > Redraft tab.

  per player   ESPN projected points/game (league scoring) and our own projection (the Kalman model's this-season line in hub_data.json)
               expected games left = team games remaining x availability, minus games he is expected to miss right now (injury advisor)
               proj points   = points/game x expected games          VOR = (points/game - replacement) x expected games
               market rank   = rank by ESPN projection x expected games (the redraft "market");  our rank = by our projection x expected games
               90th percentile points/game = projection + 1.28 sigma (sigma measured on 2010-2026: about 6.2 preseason falling to about 5.3 after 20 games)
               last 10 games (fantasy points, minutes, dates) for the trajectory line, the next 3 games with opponents, games next 7 days, games in the playoff weeks,
               minutes/points-per-minute trend, consistency (game-to-game spread), the form split from build_form_split.py when present
Runs in the daily local refresh (needs ESPN cookies and the local game logs); the page joins owners from league_rosters.json.
  uv run python research/build_redraft.py
"""
import json
import re
import statistics
import sys
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import form_common as F
from espn_api.basketball import League
from espn_api.basketball.constant import PRO_TEAM_MAP

sys.stdout.reconfigure(encoding="utf-8")
ET = ZoneInfo("America/New_York")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
SEASON_ID = 2027
FIX = {"PHL": "PHI", "NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
canon = lambda t: FIX.get(t, t)
today = datetime.now(ET).date()
SEASON_START = date(2026, 10, 20)
N_KEEP = 700
REPL_RANK = 165                       # 12 teams x ~14 usable players: the free-agent level a manager can actually pick up
IR_OK = ("OUT", "INJURY_RESERVE")
LAST_N = 10


def key(name):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(" jr.", "").replace(" jr", "").replace(" iii", "").replace(" ii", ""))


def sigma_ppg(games_played):
    return 5.2 + 1.0 * max(0.0, 1 - games_played / 20.0)


# ---------------- schedule
sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))
sp = json.load(open(HUB / "schedule_plan.json", encoding="utf-8"))
END = max(w["end"] for w in sp["calendar"])
PLAYOFF_START = next(w["start"] for w in sp["calendar"] if w["id"] == 20)
games_by_team = {}                    # team -> [(date, opp, home)]
for ds, gl in sorted(sched["games"].items()):
    for a, h, tip in gl:
        a, h = canon(a), canon(h)
        games_by_team.setdefault(a, []).append((ds, h, False))
        games_by_team.setdefault(h, []).append((ds, a, True))
t0 = today.isoformat()
first_day = max(t0, SEASON_START.isoformat())

# ---------------- ESPN projections (league scoring), ownership
lg = League(league_id=config.LEAGUE_ID, year=SEASON_ID, espn_s2=config.ESPN_S2, swid=config.SWID)
flt = {"players": {"limit": N_KEEP, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}, "filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}}}
raw = lg.espn_request.league_get(params={"view": "kona_player_info"}, headers={"x-fantasy-filter": json.dumps(flt)})["players"]
print("ESPN players:", len(raw))

# ---------------- our model, injuries, logs
hub = json.load(open(HUB / "hub_data.json", encoding="utf-8"))
hub_by = {}
for p in hub["players"]:
    if p["kind"] == "current" or key(p["player"]) not in hub_by:
        hub_by[key(p["player"])] = p
try:
    import injury_advisor as IA
    ESPN_INJ = IA.espn_injuries()
except Exception as ex:
    IA, ESPN_INJ = None, {}
    print("injury advisor unavailable:", ex)
try:
    import availability_state as AV
except Exception:
    AV = None
form_p = {}
fp_ = HUB / "form_split.json"
if fp_.exists():
    try:
        fj = json.load(open(fp_, encoding="utf-8"))
        if (today - date.fromisoformat(fj["as_of"])).days <= 4:
            form_p = fj["players"]
    except Exception:
        pass
logs = F.load_all(cache=False)
season_now = logs.loc[logs["date"].idxmax(), "season"]
by_name = {}
for name, pl in logs.groupby("name", sort=False):
    by_name.setdefault(key(name), []).append(pl)
print("game logs through", str(logs["date"].max())[:10], "season", season_now, "; form split for", len(form_p), "players")


def team_games(team, d0, d1):
    return sum(1 for ds, _, _ in games_by_team.get(team, ()) if d0 <= ds <= d1)


rows = []
for e in raw:
    p = e["player"]
    name = p["fullName"]
    team = canon(PRO_TEAM_MAP.get(p.get("proTeamId")) or "")
    if team not in games_by_team:
        continue
    st = {s["id"]: s for s in p.get("stats", [])}
    proj = st.get(f"10{SEASON_ID}") or {}
    espn_ppg = proj.get("appliedAverage") or 0.0
    if espn_ppg < 8:
        continue
    own = p.get("ownership") or {}
    hp = hub_by.get(key(name))
    ours = hp.get("year0_ppg") if hp else None
    ppg = ours if ours else espn_ppg
    status = p.get("injuryStatus") or "ACTIVE"
    info = ESPN_INJ.get(int(p["id"]))
    g_rem = team_games(team, first_day, END)
    g7 = team_games(team, t0, (today + timedelta(days=6)).isoformat())
    g_po = team_games(team, PLAYOFF_START, END)
    out_now = status in IR_OK or (info and info.get("status") == "Out")
    games_out, back = 0.0, None
    if out_now and IA is not None:
        try:
            group, tier = IA.classify(info) if info else ("other", "moderate")
            streak = 1 if today < SEASON_START else max(1, AV.out_streak(key(name).replace(" ", ""), today) if AV else 1)
            curve, _, _ = IA.out_curve(group, tier, streak)
            med = IA.median_games(curve)
            g_espn = None
            if info and info.get("return_date"):
                rd = info["return_date"][:10]
                g_espn = team_games(team, t0, (date.fromisoformat(rd) - timedelta(days=1)).isoformat()) if rd > t0 else 0
            ours_g = med if med is not None else IA.KS[-1] + 10
            games_out = g_espn if (g_espn is not None and today < SEASON_START) else (max(ours_g, g_espn) if g_espn is not None else ours_g)
            games_out = min(games_out, g_rem)
            ds_ = [ds for ds, _, _ in games_by_team[team] if ds >= first_day]
            i_ = int(round(games_out))
            back = ds_[i_] if i_ < len(ds_) else None
        except Exception as ex:
            games_out = 0.0
    miss_share = (hp.get("injury_missed") / 82.0) if hp and hp.get("injury_missed") is not None else 0.13
    miss_share = min(max(miss_share, 0.03), 0.6)
    exp_gp = max(0.0, g_rem - games_out) * (1 - miss_share)
    # recent games
    rec = {}
    pls = by_name.get(key(name))
    if pls:
        pl = max(pls, key=lambda x: x["date"].iloc[-1]).reset_index(drop=True)
        last = pl.iloc[-LAST_N:]
        cur = pl[pl["season"] == season_now]
        rec = {"g": [round(float(v), 1) for v in last.fp], "gm": [round(float(v)) for v in last["min"]], "gd": [str(v)[5:10] for v in last["date"]], "gs": last["season"].iloc[-1],
               "l10": round(float(last.fp.mean()), 1), "m10": round(float(last["min"].mean()), 1), "ppm": round(float(last.fp.sum() / last["min"].sum()), 2),
               "sd": round(float(pl.fp.iloc[-25:].std()), 1) if len(pl) >= 8 else None, "n_season": int(len(cur)),
               "s_avg": round(float(cur.fp.mean()), 1) if len(cur) else None, "s_min": round(float(cur["min"].mean()), 1) if len(cur) else None}
    n_played = rec.get("n_season", 0) if rec.get("gs") == season_now and season_now.startswith(str(SEASON_ID - 1)) else 0
    nxt = [(ds[5:], ("vs " if home else "@") + opp) for ds, opp, home in games_by_team[team] if ds >= t0][:3]
    fr = form_p.get(key(name))
    s = sigma_ppg(n_played)
    rows.append({"id": p["id"], "name": name, "team": team, "pos": None,
                 "pos_ids": p.get("eligibleSlots"), "age": hp.get("age") if hp else None, "status": status, "espn_ppg": round(espn_ppg, 1), "ppg": round(ppg, 1), "our": bool(ours),
                 "g_rem": g_rem, "exp_gp": round(exp_gp, 1), "games_out": round(games_out, 1) if games_out else 0, "back": back, "g7": g7, "gpo": g_po,
                 "p90": round(ppg + 1.28 * s, 1), "p10": round(ppg - 1.28 * s, 1), "own": own.get("percentOwned"), "adp": (own.get("averageDraftPosition") if (own.get("averageDraftPosition") or 0) < 139 else None),
                 "inj_tier": hp.get("injury_tier") if hp else None, "inj_missed": hp.get("injury_missed") if hp else None, "inj_hmissed": hp.get("injury_health_missed") if hp else None, "traj": (hp.get("trajectory") or [None])[:3] if hp else None,
                 "next": nxt, **rec,
                 "form": ({"d": fr["d"], "keep": fr["keep"], "chips": fr.get("chips", [])} if fr and "d" in fr else None)})

# ESPN slot ids -> positions (0 PG, 1 SG, 2 SF, 3 PF, 4 C, 5 G, 6 F)
POS = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}
for r in rows:
    r["pos"] = "/".join(POS[i] for i in (r.pop("pos_ids") or []) if i in POS) or "—"

# ---------------- valuation
rows.sort(key=lambda r: -r["ppg"] * r["exp_gp"])
pool = [r for r in rows if r["g_rem"] and r["exp_gp"] >= 0.5 * r["g_rem"]]
repl = sorted((r["ppg"] for r in pool), reverse=True)[REPL_RANK - 1] if len(pool) >= REPL_RANK else 20.0
for r in rows:
    r["pts"] = round(r["ppg"] * r["exp_gp"])
    r["vor"] = round((r["ppg"] - repl) * r["exp_gp"])
    r["pts90"] = round(r["p90"] * r["exp_gp"])
    r["espn_pts"] = round(r["espn_ppg"] * r["exp_gp"])
for i, r in enumerate(sorted(rows, key=lambda r: -r["pts"]), 1):
    r["rank"] = i
for i, r in enumerate(sorted(rows, key=lambda r: -r["espn_pts"]), 1):
    r["mkt"] = i
rows.sort(key=lambda r: -r["vor"])
out = {"generated": datetime.now(timezone.utc).isoformat(), "asof": t0, "season_now": season_now, "preseason": today < SEASON_START, "repl_ppg": round(repl, 1), "repl_rank": REPL_RANK,
       "season_end": END, "playoff_start": PLAYOFF_START, "log_through": str(logs["date"].max())[:10], "sigma": {"pre": sigma_ppg(0), "late": sigma_ppg(99)}, "players": rows}
(HUB / "redraft_data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"wrote redraft_data.json: {len(rows)} players, replacement {repl:.1f} ppg (rank {REPL_RANK}); {round((HUB / 'redraft_data.json').stat().st_size / 1024)} KB")
for r in rows[:12]:
    print(f"{r['name']:24s} {r['team']} {r['pos']:8s} ppg {r['ppg']:5.1f} espn {r['espn_ppg']:5.1f} gp {r['exp_gp']:5.1f}/{r['g_rem']} pts {r['pts']:5d} vor {r['vor']:5d} rank {r['rank']:3d} mkt {r['mkt']:3d} l10 {r.get('l10')} {r['status']}")
