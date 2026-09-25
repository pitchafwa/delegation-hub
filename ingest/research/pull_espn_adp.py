"""Real ESPN average draft position (ADP) by season, pulled from ESPN's public
player pool (ownership.averageDraftPosition -- the actual ESPN.com draft
data). ESPN seasonId = the year the season ENDS (2025 = 2024-25); the
upcoming 2026-27 season is seasonId 2027 and has live ADP right now.

seasonId 2026 (2025-26) has no recorded ADP (ESPN returns the 140.0
placeholder for everyone) -- that season simply has no ADP benchmark.

Writes data/espn_adp.csv keyed to NBA PLAYER_ID (crosswalk + name fallback,
same scheme as pull_espn_positions.py). ADP >= 139 = ESPN's "undrafted"
placeholder, stored as NaN.
"""
import json, re, sys, time, unicodedata
from pathlib import Path
import pandas as pd, requests
from espn_api.basketball.constant import PRO_TEAM_MAP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
cookies = {"espn_s2": config.ESPN_S2, "SWID": config.SWID}


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


cw = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
e2n = dict(zip(cw["ESPNID"].astype(int), cw["NBAID"]))
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
n2n = dict(zip((bio["PLAYER_FIRST_NAME"] + " " + bio["PLAYER_LAST_NAME"]).apply(norm), bio["PERSON_ID"]))

flt = {"players": {"limit": 600, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
rows = []
SEASONS = [int(x) for x in sys.argv[1:]] or [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2027]   # pass e.g. `2027` to refresh only the live season
for season in SEASONS:
    r = requests.get(f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/players",
                     params={"view": "kona_player_info"}, headers={"x-fantasy-filter": json.dumps(flt)},
                     cookies=cookies, timeout=60)
    r.raise_for_status()
    pl = r.json()
    pl = pl.get("players", pl) if isinstance(pl, dict) else pl
    n = 0
    for e in pl:
        p = e.get("player", e)
        own = p.get("ownership", {}) or {}
        adp = own.get("averageDraftPosition")
        pid = e.get("id", p.get("id"))
        nba = e2n.get(pid)
        if nba is None or pd.isna(nba):
            nba = n2n.get(norm(p["fullName"]))
        rows.append(dict(season_id=season, espn_id=pid, name=p["fullName"], PLAYER_ID=nba,
                         adp=(adp if adp and adp < 139 else None), pct_owned=own.get("percentOwned"),
                         pro_team=PRO_TEAM_MAP.get(p.get('proTeamId')), espn_rank=(p.get("draftRanksByRankType", {}).get("STANDARD", {}) or {}).get("rank")))
        n += 1
    print(season, "players:", n, flush=True)
    time.sleep(1)
out = pd.DataFrame(rows)
_path = ROOT / "data" / "espn_adp.csv"
if _path.exists() and len(sys.argv) > 1:      # incremental: keep the seasons we did not refresh
    _old = pd.read_csv(_path)
    out = pd.concat([_old[~_old["season_id"].isin(SEASONS)], out], ignore_index=True)
out.to_csv(_path, index=False)
g = out.groupby("season_id").agg(n=("name", "count"), with_adp=("adp", lambda s: s.notna().sum()),
                                  mapped=("PLAYER_ID", lambda s: s.notna().sum()))
print(g)
