"""Redraft board data: every fantasy-relevant player valued for the REST OF THIS SEASON (dashboard/redraft_data.json), for the Players > Redraft tab.

  per player   ESPN projected points/game (league scoring) and our own projection (the Kalman model's this-season line in hub_data.json)
               expected games left = team games remaining x availability, minus games he is expected to miss right now (injury advisor)
               proj points   = points/game x expected games          VOR = (points/game - replacement) x expected games
               market rank   = ESPN's points-league rankings article averaged with ESPN ADP (FantasyPros' consensus is the fallback for players ESPN does not rank; switch MARKET_PRIMARY when its
                               injury/rookie rankings look right); the old rule below is unused: where FantasyPros and the ESPN blend
                               are surprisingly different (15+ places and 35%+) the market is their average and the row is flagged.  Fallback (a player in none of them): rank by ESPN projection x expected games.  Our rank = by our projection x expected games
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
from season import current_season_id
SEASON_ID = current_season_id()
from team_abbr import canon
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



# ---------------- market: FantasyPros consensus + ESPN rankings/ADP (public pages; only the resulting rank is published, not the source lists)
MARKET_DIR = Path(__file__).resolve().parent / "data" / "market"
MARKET_DIR.mkdir(parents=True, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0"}
FP_URL = "https://www.fantasypros.com/nba/rankings/overall-points-espn.php"
ESPN_RANK_URL = "https://www.espn.com/fantasy/basketball/story/_/id/49960162/fantasy-basketball-points-league-rankings-2026-27-nba-season"
DISAGREE_ABS, DISAGREE_REL = 15, 0.35          # 'surprisingly different': 15+ places AND 35%+ of the better rank (a 20-place gap means more at #20 than at #200)


def _cached(name, fetch):
    path = MARKET_DIR / name
    try:
        txt = fetch()
        path.write_text(json.dumps(txt), encoding="utf-8")
        return txt, "live"
    except Exception as ex:
        print(f"market source {name} failed ({ex}); using the last saved copy" if path.exists() else f"market source {name} failed ({ex}); not available")
        return (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}), "cached"


def _fetch_fp():
    import requests
    h = requests.get(FP_URL, headers=HDR, timeout=40).text
    i = h.index("ecrData = ") + len("ecrData = ")
    data, _ = json.JSONDecoder().raw_decode(h[i:])
    return {key(pl["player_name"]): int(float(pl["rank_ecr"])) for pl in data["players"] if pl.get("rank_ecr")}


def _fetch_espn_article():
    import requests
    h = requests.get(ESPN_RANK_URL, headers=HDR, timeout=40).text
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    t = re.sub(r"<[^>]+>", chr(10), t)
    pat = re.compile(r"(\d+)\.\s+([^,]+?)\s*,\s*([A-Z]+)")
    t = re.sub(r"\s+", " ", t)
    rows = pat.findall(t)
    if len(rows) < 50:
        raise ValueError("ESPN rankings article parsed too few rows")
    return {key(nm.strip()): int(n) for n, nm, _ in rows}


FP_RANK, fp_src = _cached("fantasypros.json", _fetch_fp)
ESPN_RANK, es_src = _cached("espn_article.json", _fetch_espn_article)
print(f"market sources: FantasyPros {len(FP_RANK)} players ({fp_src}), ESPN rankings article {len(ESPN_RANK)} ({es_src})")

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
try:
    import matchup_context as MC
    import usage_flow as UF
    team_matchup = json.load(open(HUB / "team_matchup.json", encoding="utf-8"))["teams"]
    dates_by_team = {t: set(ds for ds, _, _ in v) for t, v in games_by_team.items()}
except Exception as ex:
    MC, team_matchup, dates_by_team = None, {}, {}
    print("matchup context unavailable for next-3-games chart:", ex)
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
    nxt_games = [(ds, opp, home) for ds, opp, home in games_by_team[team] if ds >= t0][:3]
    nxt = [(ds[5:], ("vs " if home else "@") + opp) for ds, opp, home in nxt_games]
    # per-game adjusted projection for the next-3-games chart: same matchup adjustment (opponent defense, positional defense, back-to-back,
    # expected pace) build_week_plan.py applies to the daily plan -- "opponent missing production" is skipped here since it needs that script's
    # live league-wide injury/usage-flow build, not worth duplicating for this chart. Falls back to the flat ppg when matchup data is unavailable.
    next_proj = None
    if MC and team_matchup:
        own_tm = team_matchup.get(team)
        nsp = (hp.get("next_season_proj") if hp and hp.get("kind") == "current" else (hp.get("rookie_proj") if hp else None)) if hp else None
        pos = UF.pos_probs(nsp["REB"] * 36, nsp["AST"] * 36, nsp["BLK"] * 36, nsp["STL"] * 36, nsp.get("FG3M", 0) * 36) if nsp and (nsp.get("MIN") or 0) > 0 else (1 / 3, 1 / 3, 1 / 3)
        next_proj = []
        for ds, opp, home in nxt_games:
            tm = team_matchup.get(opp)
            if not (own_tm and tm):
                next_proj.append(round(ppg, 1))
                continue
            posdef = pos[0] * tm["fpC"] + pos[1] * tm["fpF"] + pos[2] * tm["fpG"]
            day_before = (date.fromisoformat(ds) - timedelta(days=1)).isoformat()
            b2b_opp = day_before in dates_by_team.get(opp, set())
            exp_pace = MC.expected_pace(own_tm["pace"], tm["pace"])
            adj = MC.adjustment(drtg_opp=tm["drtg"], posdef_opp=posdef, b2b_opp=b2b_opp, exp_pace=exp_pace)
            next_proj.append(round(ppg + adj, 1))
    fr = form_p.get(key(name))
    s = sigma_ppg(n_played)
    rows.append({"id": p["id"], "name": name, "team": team, "pos": None,
                 "pos_ids": p.get("eligibleSlots"), "age": hp.get("age") if hp else None, "status": status, "espn_ppg": round(espn_ppg, 1), "ppg": round(ppg, 1), "our": bool(ours),
                 "g_rem": g_rem, "exp_gp": round(exp_gp, 1), "games_out": round(games_out, 1) if games_out else 0, "back": back, "g7": g7, "gpo": g_po,
                 "p90": round(ppg + 1.28 * s, 1), "p10": round(ppg - 1.28 * s, 1), "own": own.get("percentOwned"), "adp": (own.get("averageDraftPosition") if (own.get("averageDraftPosition") or 0) < 139 else None),
                 "inj_tier": hp.get("injury_tier") if hp else None, "inj_missed": hp.get("injury_missed") if hp else None, "inj_hmissed": hp.get("injury_health_missed") if hp else None, "traj": (hp.get("trajectory") or [None])[:3] if hp else None,
                 "next": nxt, "next_proj": next_proj, **rec,
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
    r["mkt_proj"] = i                                        # fallback market: ESPN's projection x our expected games
MARKET_PRIMARY = "espn"        # "espn": ESPN's points rankings article averaged with ESPN ADP; FantasyPros only for players ESPN does not rank.  Flip to "fp" once FantasyPros' injury/rookie rankings catch up (they had not by 9/27: Tatum #118 vs ESPN #9).
for r in rows:
    k = key(r["name"])
    fp, er, adp = FP_RANK.get(k), ESPN_RANK.get(k), r.get("adp")
    espn_parts = [x for x in (er, adp) if x]
    espn_blend = round(sum(espn_parts) / len(espn_parts)) if espn_parts else None
    order = (("espn", espn_blend), ("fp", fp)) if MARKET_PRIMARY == "espn" else (("fp", fp), ("espn", espn_blend))
    r["mkt"], r["mkt_src"] = next(((v, src) for src, v in order if v), (None, "none"))
print(f"market ranks: {sum(1 for r in rows if r['mkt_src']=='espn')} ESPN (rankings+ADP), {sum(1 for r in rows if r['mkt_src']=='fp')} FantasyPros (fallback), {sum(1 for r in rows if r['mkt_src']=='none')} none")
rows.sort(key=lambda r: -r["vor"])
out = {"generated": datetime.now(timezone.utc).isoformat(), "asof": t0, "season_now": season_now, "preseason": today < SEASON_START, "repl_ppg": round(repl, 1), "repl_rank": REPL_RANK,
       "season_end": END, "playoff_start": PLAYOFF_START, "log_through": str(logs["date"].max())[:10], "sigma": {"pre": sigma_ppg(0), "late": sigma_ppg(99)}, "players": rows}
(HUB / "redraft_data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"wrote redraft_data.json: {len(rows)} players, replacement {repl:.1f} ppg (rank {REPL_RANK}); {round((HUB / 'redraft_data.json').stat().st_size / 1024)} KB")
for r in rows[:12]:
    print(f"{r['name']:24s} {r['team']} {r['pos']:8s} ppg {r['ppg']:5.1f} espn {r['espn_ppg']:5.1f} gp {r['exp_gp']:5.1f}/{r['g_rem']} pts {r['pts']:5d} vor {r['vor']:5d} rank {r['rank']:3d} mkt {r['mkt'] or 0:3d} l10 {r.get('l10')} {r['status']}")
