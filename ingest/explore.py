"""One-off exploration script: connect to the real league and dump raw shapes.
Not part of the eventual pipeline — throwaway, for Phase 0 verification only.
"""
import json

from espn_api.basketball import League

import config

league = League(
    league_id=config.LEAGUE_ID,
    year=config.SEASON,
    espn_s2=config.ESPN_S2,
    swid=config.SWID,
)

print("=== LEAGUE SETTINGS ===")
s = league.settings
print("name:", s.name)
print("team_count:", s.team_count)
print("playoff_team_count:", s.playoff_team_count)

print()
print("=== SCORING ===")
# espn-api basketball settings expose scoring settings differently per version;
# dump whatever's on the object so we can see the real shape.
for attr in dir(s):
    if attr.startswith("_"):
        continue
    val = getattr(s, attr)
    if callable(val):
        continue
    print(f"{attr}: {val!r}"[:300])

print()
print("=== TEAMS ===")
for t in league.teams:
    print(t.team_id, t.team_name, "owners:", t.owners)

print()
print("=== MY TEAM ROSTER (first team as sample) ===")
team = league.teams[0]
for p in team.roster[:5]:
    print(json.dumps({
        "name": p.name,
        "playerId": p.playerId,
        "position": p.position,
        "proTeam": p.proTeam,
        "injuryStatus": getattr(p, "injuryStatus", None),
        "total_points": getattr(p, "total_points", None),
        "avg_points": getattr(p, "avg_points", None),
        "stats_keys": list(p.stats.keys())[:10] if hasattr(p, "stats") else None,
    }, indent=2, default=str))
