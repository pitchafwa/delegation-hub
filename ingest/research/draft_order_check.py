"""Does last season's finish (playoff finish for playoff teams, regular-season standing for the rest) determine next draft's order (reverse)? Prints per year."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import config
from espn_api.basketball import League
for Y in (2021, 2022, 2023, 2024, 2025, 2026):
    try:
        a = League(league_id=config.LEAGUE_ID, year=Y, espn_s2=config.ESPN_S2, swid=config.SWID)
        b = League(league_id=config.LEAGUE_ID, year=Y + 1, espn_s2=config.ESPN_S2, swid=config.SWID)
    except Exception as e:
        print(Y, "skip", repr(e)[:80]); continue
    fin = {t.team_id: (t.final_standing, t.standing, t.team_abbrev) for t in a.teams}
    order = []
    for p in b.draft:
        if p.round_num == 1:
            order.append(p.team.team_id)
    print(f"\nseason {Y} -> draft {Y+1}: round-1 order (earliest first) with (final_standing, reg-season standing):")
    print("  ", [(fin[t][2], fin[t][0], fin[t][1]) for t in order if t in fin])
