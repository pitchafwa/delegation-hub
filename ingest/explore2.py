import json

from espn_api.basketball import League
from espn_api.basketball.constant import STATS_MAP

import config

league = League(
    league_id=config.LEAGUE_ID,
    year=config.SEASON,
    espn_s2=config.ESPN_S2,
    swid=config.SWID,
)

raw = league.settings._raw_scoring_settings
print("=== RAW SCORING SETTINGS ===")
print(json.dumps(raw, indent=2)[:3000])

print()
print("=== ROSTER SETTINGS (via cached league request) ===")
data = league.espn_request.league_get(params={"view": "mSettings"})
roster_settings = data.get("settings", {}).get("rosterSettings", {})
print(json.dumps(roster_settings, indent=2))
