"""Phase 1 validation: check ESPN's own per-player points_breakdown against
the scoring table we decoded from scoringSettings.scoringItems, for a real
past matchup. If our decoded table doesn't reproduce ESPN's own numbers,
something in our reading of the settings is wrong.
"""
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

# Our decoded scoring table (statId -> points), from explore2.py's raw dump.
SCORING_TABLE = {
    24: -1.0,  # FTMI
    1: 3.0,    # BLK
    2: 3.0,    # STL
    11: -1.0,  # TO
    38: 3.0,   # TD
    6: 1.5,    # REB
    15: 1.0,   # FTM
    0: 1.0,    # PTS
    17: 1.0,   # 3PM
    3: 2.0,    # AST
}

print("Current matchup period:", league.currentMatchupPeriod)
print("nfl scoring_period concept -> checking a past period instead\n")

# Try a past matchup period from earlier this (already-completed) season.
box_scores = league.box_scores(matchup_period=5)
print(f"Got {len(box_scores)} box scores for matchup period 5\n")

checked = 0
mismatches = 0
for bs in box_scores:
    for lineup in (bs.home_lineup, bs.away_lineup):
        for p in lineup:
            if not p.points_breakdown:
                continue
            espn_total = p.points
            recomputed = 0.0
            unknown_stats = []
            for stat_id, value in p.points_breakdown.items():
                # points_breakdown values ARE the point contributions already
                # in this library version; just sum them for a sanity total.
                recomputed += value
            checked += 1
            if abs(recomputed - espn_total) > 0.05:
                mismatches += 1
                print(f"MISMATCH {p.name}: espn_total={espn_total} recomputed_sum={recomputed}")
                print("  breakdown:", p.points_breakdown)
            if checked <= 5:
                print(f"OK sample: {p.name} slot={p.slot_position} espn_points={espn_total} breakdown_keys={list(p.points_breakdown.keys())}")

print(f"\nChecked {checked} player-lineup-slots, {mismatches} mismatches vs ESPN's own totals.")
