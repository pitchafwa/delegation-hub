"""Pull the CURRENT fantasy rosters of every team in Tommy's ESPN league (2026-27 = seasonId 2027)
and write dashboard/league_rosters.json for the site's league-context features (Owner column,
free-agent filter, Teams view).

Player ids are mapped to the hub's ids ('c'/'p' + NBA person id) via the ESPN->NBA crosswalk, with a
name fallback (same scheme as pull_espn_adp.py); anything that can't be mapped keeps its ESPN name
and shows on the team page but has no valuation.

Published data is deliberately minimal: team name/abbrev/logo and each player's slot, injury status
and how he was acquired. No owner names, no cookies, nothing from .env.
Run:  uv run python research/pull_league_rosters.py   (from ingest/)
"""
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from espn_api.basketball import League

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
HUB = ROOT.parent.parent / "dashboard"
SEASON_ID = 2027  # 2026-27


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


# ESPN id -> NBA id. The big crosswalk CSV is gitignored (not ours to republish), so a small committed map
# (espn_id_map.json, only ids that matter here) is what CI uses; it is refreshed from the CSV whenever the CSV exists.
MAP_PATH = ROOT / "espn_id_map.json"
CW_PATH = ROOT / "data" / "espn_nba_id_crosswalk.csv"
e2n = {int(k): int(v) for k, v in json.load(open(MAP_PATH)).items()} if MAP_PATH.exists() else {}
if CW_PATH.exists():
    _cw = pd.read_csv(CW_PATH, encoding="latin1").dropna(subset=["ESPNID", "NBAID"])
    e2n.update(dict(zip(_cw["ESPNID"].astype(int), _cw["NBAID"].astype(int))))
hub = json.load(open(HUB / "hub_data.json", encoding="utf-8"))
hub_ids = {p["id"] for p in hub["players"]}
hub_by_name = {norm(p["player"]): p["id"] for p in hub["players"]}


def hub_id(espn_id, name):
    nba = e2n.get(int(espn_id))
    if nba is not None:
        for pre in ("c", "p"):
            if f"{pre}{nba}" in hub_ids:
                return f"{pre}{nba}"
    return hub_by_name.get(norm(name))


lg = League(league_id=config.LEAGUE_ID, year=SEASON_ID, espn_s2=config.ESPN_S2, swid=config.SWID)
teams, unmapped = [], []
for t in lg.teams:
    roster = []
    for pl in t.roster:
        hid = hub_id(pl.playerId, pl.name)
        if hid is None:
            unmapped.append((t.team_abbrev, pl.name))
        roster.append({"espn_id": pl.playerId, "name": pl.name, "id": hid, "slot": pl.lineupSlot,
                       "injury": pl.injuryStatus, "acq": getattr(pl, "acquisitionType", None)})
    teams.append({"id": t.team_id, "name": t.team_name.strip(), "abbrev": t.team_abbrev, "logo": t.logo_url,
                  "roster": roster})

if len(teams) < 12 or sum(len(t["roster"]) for t in teams) < 150:
    sys.exit(f"refusing to write: only {len(teams)} teams / {sum(len(t['roster']) for t in teams)} players came back from ESPN")
if CW_PATH.exists():  # keep the committed map current (only hub players' ids)
    nba_hub = {int(i[1:]) for i in hub_ids}
    MAP_PATH.write_text(json.dumps({str(k): v for k, v in sorted(e2n.items()) if v in nba_hub}, separators=(",", ":")))
out = {"generated": datetime.now(timezone.utc).isoformat(), "season": SEASON_ID, "league": lg.settings.name,
       "roster_size": max(len(t["roster"]) for t in teams), "teams": teams}
(HUB / "league_rosters.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
n = sum(len(t["roster"]) for t in teams)
print(f"{len(teams)} teams, {n} rostered players, {n - len(unmapped)} mapped to hub valuations")
for u in unmapped:
    print("  unmapped:", u)
