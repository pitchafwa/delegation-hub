"""Independent validation: take RAW counting stats (not ESPN's precomputed
appliedStats) for real players in a real past matchup period, multiply by
our own hand-decoded scoring table, and check the result equals ESPN's
own appliedTotal. This is the real test — the earlier check only confirmed
appliedStats sums to appliedTotal, which is circular (both are ESPN's own
numbers). This one recomputes from raw box-score counting stats independently.
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

SCORING_TABLE = {
    "24": -1.0,  # FTMI
    "1": 3.0,    # BLK
    "2": 3.0,    # STL
    "11": -1.0,  # TO
    "38": 3.0,   # TD
    "6": 1.5,    # REB
    "15": 1.0,   # FTM
    "0": 1.0,    # PTS
    "17": 1.0,   # 3PM
    "3": 2.0,    # AST
}

SCORING_PERIOD = 30  # a day within matchup period 5 of last completed season

data = league.espn_request.league_get(
    params={"view": "mBoxscore", "scoringPeriodId": SCORING_PERIOD},
)

checked = 0
matches = 0
mismatches = 0

teams = data.get("teams") or data.get("schedule", [])
# mBoxscore returns schedule with per-matchup rosters; walk it defensively.
schedule = data.get("schedule", [])
print(f"Found {len(schedule)} matchups for scoringPeriodId={SCORING_PERIOD}\n")

for matchup in schedule:
    for side in ("home", "away"):
        team_data = matchup.get(side, {})
        roster = team_data.get("rosterForCurrentScoringPeriod") or team_data.get("rosterForMatchupPeriod")
        if not roster:
            continue
        for entry in roster.get("entries", []):
            player = entry.get("playerPoolEntry", {}).get("player", {})
            name = player.get("fullName", "?")
            for stat_line in player.get("stats", []):
                if stat_line.get("scoringPeriodId") != SCORING_PERIOD:
                    continue
                if stat_line.get("statSourceId") != 0:  # 0 = actual, 1 = projected
                    continue
                raw_stats = stat_line.get("stats", {})
                applied_total = stat_line.get("appliedTotal")
                if applied_total is None or not raw_stats:
                    continue
                recomputed = 0.0
                for stat_id, coeff in SCORING_TABLE.items():
                    recomputed += coeff * raw_stats.get(stat_id, 0)
                checked += 1
                ok = abs(recomputed - applied_total) <= 0.05
                matches += ok
                mismatches += not ok
                if checked <= 8 or not ok:
                    print(f"{'OK ' if ok else 'MISMATCH'} {name}: recomputed={recomputed:.2f} espn_applied={applied_total:.2f} raw={raw_stats}")

print(f"\nChecked {checked} real player-days. Matches={matches} Mismatches={mismatches}")
