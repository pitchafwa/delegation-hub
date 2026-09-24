"""Real, read-only pull of ESPN's actual fantasy position eligibility
(multi-position, e.g. Quentin Grimes = SG/SF) for the full relevant player
pool -- free agents + every rostered player across the whole league, deduped
by ESPN playerId, then mapped to our NBA PLAYER_ID via the existing real
espn_nba_id_crosswalk.csv (no fuzzy name-matching needed).
"""
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

ROOT = Path(__file__).resolve().parent

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)
fa = league.free_agents(size=2000)
rostered = []
for t in league.teams:
    rostered.extend(t.roster)
all_players = {p.playerId: p for p in fa + rostered}
print(f"Real ESPN player pool pulled: {len(all_players)} unique players "
      f"({len(fa)} free agents + {len(rostered)} rostered across {len(league.teams)} teams)", flush=True)

TRUE_POS = ["PG", "SG", "SF", "PF", "C"]

# ESPN's own internal team codes differ from the standard NBA abbreviation
# used everywhere else in this pipeline (player_bio.csv, logo URLs)
ESPN_TEAM_FIX = {"PHL": "PHI", "PHO": "PHX"}


def clean_position(eligible_slots, default_position):
    """Real ESPN compound slots (e.g. "SG/SF", "PF/C") are what ESPN's own
    site displays as a player's position -- exploding EVERY compound into
    its parts and unioning them (the first version of this) overstates
    flexibility for players with broad early-season rookie eligibility
    (e.g. Cooper Flagg showed PG/SG/SF/PF/C, all 5 spots, because his real
    eligibleSlots technically include both an SG/SF and a PF/C compound).
    Instead: prefer the single two-position compound that contains the
    player's own default position: if there's exactly one, that IS the
    real "eligible at these 2" answer. If eligibility is broader than a
    single clean pair (multiple different compounds, like Flagg), that's
    real but ambiguous -- fall back to just the simple default position
    rather than manufacturing a 3+ position combo ESPN's own UI wouldn't
    show as a single tag."""
    compounds = [s for s in eligible_slots if "/" in s and all(p in TRUE_POS for p in s.split("/"))]
    matching = [c for c in compounds if default_position in c.split("/")]
    if len(matching) == 1:
        parts = matching[0].split("/")
        return "/".join(p for p in TRUE_POS if p in parts)
    return default_position


crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

# fallback for players the (somewhat dated) crosswalk misses: real name match
# against player_bio.csv, same normalization scheme used throughout this project
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
bio["_norm"] = (bio["PLAYER_FIRST_NAME"] + " " + bio["PLAYER_LAST_NAME"]).apply(normalize_name)
norm_to_nbaid = dict(zip(bio["_norm"], bio["PERSON_ID"]))

rows = []
via_crosswalk, via_name = 0, 0
for espn_id, p in all_players.items():
    nba_id = espnid_to_nbaid.get(espn_id)
    if nba_id is not None and not pd.isna(nba_id):
        via_crosswalk += 1
    else:
        nba_id = norm_to_nbaid.get(normalize_name(p.name))
        if nba_id is not None:
            via_name += 1
    if nba_id is None or (isinstance(nba_id, float) and pd.isna(nba_id)):
        continue
    rows.append({
        "PLAYER_ID": int(nba_id), "espn_name": p.name,
        "espn_position": clean_position(p.eligibleSlots, p.position),
        "injury_status": p.injuryStatus,
        "pro_team": ESPN_TEAM_FIX.get(p.proTeam, p.proTeam) if p.proTeam not in (None, "FA") else None,
    })
print(f"Matched to real NBA PLAYER_ID: {via_crosswalk} via crosswalk + {via_name} via name = "
      f"{via_crosswalk+via_name} of {len(all_players)}", flush=True)

out = pd.DataFrame(rows).drop_duplicates(subset=["PLAYER_ID"])
out.to_csv(ROOT / "data" / "espn_positions.csv", index=False)
print(f"Saved {len(out)} rows to espn_positions.csv")

grimes = out[out["espn_name"].str.contains("Grimes", case=False, na=False)]
print(grimes)
