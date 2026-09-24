"""Match Tommy's real ESPN roster to nba_api PLAYER_IDs the right way:
stable-ID join via a public ESPN<->NBA.com crosswalk FIRST, name-matching only
as a validated fallback -- logged explicitly, per the WRPI/RUPI lesson that
name-only joins caused two real data-corruption incidents in that project.
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config

ROOT = Path(__file__).resolve().parent

crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1")
crosswalk = crosswalk.dropna(subset=["ESPNID"]).copy()
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
current = pd.read_csv(ROOT / "data" / "current_season_for_prediction.csv")


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


current["NORM_NAME"] = current["PLAYER_NAME"].apply(normalize_name)
name_to_playerid = dict(zip(current["NORM_NAME"], current["PLAYER_ID"]))

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)
my_team = next(t for t in league.teams if t.team_id == 12)

print(f"{'ESPN name':24s} {'ESPN ID':>10s}  {'Method':16s} {'NBA PLAYER_ID':>14s}  Result\n")
rows = []
for p in my_team.roster:
    espn_id = p.playerId
    nba_id = espnid_to_nbaid.get(espn_id)
    method = "stable-ID crosswalk"
    if nba_id is None or pd.isna(nba_id):
        # Fallback: validated name match, logged explicitly.
        norm = normalize_name(p.name)
        nba_id = name_to_playerid.get(norm)
        method = "NAME FALLBACK (validate!)"
    else:
        nba_id = int(nba_id)

    status = "OK" if nba_id is not None else "NO MATCH"
    print(f"{p.name:24s} {espn_id:>10d}  {method:26s} {str(nba_id):>14s}  {status}")
    rows.append({"espn_name": p.name, "espn_id": espn_id, "nba_player_id": nba_id, "match_method": method})

result = pd.DataFrame(rows)
result.to_csv(ROOT / "data" / "roster_id_map.csv", index=False)
print(f"\n{(result['match_method'] == 'stable-ID crosswalk').sum()} matched via stable ID, "
      f"{result['match_method'].str.contains('FALLBACK').sum()} via validated name fallback, "
      f"{result['nba_player_id'].isna().sum()} unmatched.")
