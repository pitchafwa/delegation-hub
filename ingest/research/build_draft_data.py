"""Draft assistant data -> dashboard/draft_data.json (ESPN only, so it can also run in the GitHub Action).

Contents:
  draft   : the league's real draft board for the upcoming draft (ESPN pre-creates every pick slot: 12 teams x 15 rounds, snake; rounds that are
            reserved for keepers are flagged; picks fill in as keepers are entered and as the draft runs)
  teams   : team ids / abbreviations / current rosters (ESPN player ids).  After the keeper deadline, rosters ARE the keepers.
  players : per ESPN player id: league-scored projection (ESPN's own points/game, projected games, season total), eligible lineup slots, NBA team,
            injury status, and ESPN's redraft rank = rank by projected league-scored SEASON TOTAL (the redraft value in this scoring system).
The page joins this to hub_data.json (our VOR / asset / market / model projection) by name and ESPN id.
Run from ingest/:  uv run python research/build_draft_data.py
"""
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
SEASON_ID = 2027
MY_ABBREV = "DRNK"
LINEUP = {"PG", "SG", "SF", "PF", "C", "G", "F", "UT"}


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


lg = League(league_id=config.LEAGUE_ID, year=SEASON_ID, espn_s2=config.ESPN_S2, swid=config.SWID)
raw = lg.espn_request.league_get(params={"view": ["mDraftDetail", "mSettings"]})
dd = raw["draftDetail"]
ds = raw["settings"]["draftSettings"]
picks = [{"n": p["overallPickNumber"], "r": p["roundId"], "team": p["teamId"], "keeperSlot": bool(p.get("reservedForKeeper")), "keeper": bool(p.get("keeper")),
          "player": p["playerId"] if p.get("playerId", -1) > 0 else None} for p in dd["picks"]]
teams = []
for t in lg.teams:
    teams.append({"id": t.team_id, "abbrev": t.team_abbrev, "name": t.team_name.strip(),
                  "roster": [{"id": p.playerId, "name": p.name, "slot": p.lineupSlot} for p in t.roster]})
_tp = Path(__file__).resolve().parent / "draft_trades.json"
trades = json.load(open(_tp, encoding="utf-8")).get("picks", []) if _tp.exists() else []
my = next((t for t in teams if t["abbrev"] == MY_ABBREV), teams[0])


def info(p):
    st = p.stats.get(f"{SEASON_ID}_projected") or {}
    avg = st.get("avg") or {}
    return {"name": p.name, "espn_id": p.playerId, "avg": st.get("applied_avg"), "total": st.get("applied_total"), "gp": avg.get("GP"),
            "slots": [s for s in p.eligibleSlots if s in LINEUP], "team": p.proTeam, "status": p.injuryStatus or "ACTIVE"}


players = {}
for t in lg.teams:
    for p in t.roster:
        players[p.playerId] = info(p)
for p in lg.free_agents(size=600):
    players.setdefault(p.playerId, info(p))
ranked = sorted([v for v in players.values() if v["total"]], key=lambda v: -v["total"])
for i, v in enumerate(ranked, 1):
    v["redraft_rank"] = i
n_kslots = sum(1 for p in picks if p["keeperSlot"])
filled_k = sum(1 for p in picks if p["keeperSlot"] and p["player"])
out = {"generated": datetime.now(timezone.utc).isoformat(), "season": SEASON_ID, "my_id": my["id"], "my_abbrev": my["abbrev"],
       "draft": {"date_ms": ds.get("date"), "keeper_deadline_ms": ds.get("keeperDeadlineDate"), "keepers_now": ds.get("keeperCount"), "keepers_future": ds.get("keeperCountFuture"),
                 "type": ds.get("type"), "seconds_per_pick": ds.get("timePerSelection"), "in_progress": bool(dd.get("inProgress")), "drafted": bool(dd.get("drafted")),
                 "keeper_slots": n_kslots, "keeper_slots_filled": filled_k, "trades": trades, "picks": picks},
       "teams": teams, "players": {str(k): v for k, v in players.items()}}
(HUB / "draft_data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
sizes = sorted(len(t["roster"]) for t in teams)
print(f"{len(picks)} pick slots ({n_kslots} reserved for keepers, {filled_k} filled), {len(players)} players with ESPN projections, roster sizes {sizes}, my team {my['abbrev']} id {my['id']}")
print("wrote draft_data.json", round((HUB / "draft_data.json").stat().st_size / 1024), "KB")
