"""Single source of truth for NBA team abbreviation normalization, used across every script that joins data between sources.

Different data sources spell the same team differently -- ESPN's fantasy API uses its own quirky short codes for some teams (PHO, BRK,
CHO, NY, SA, GS, UTAH, WSH, PHL, NOR/NO), while historical box-score data (nba_api) still carries old franchise codes for teams that moved
or renamed (NOH -> New Orleans Pelicans, NJN -> Brooklyn Nets, CHO -> Charlotte [also happens to map to CHA, no conflict], SEA -> OKC).
Before 2026-09-29, over a dozen scripts each defined their own partial copy of this dict (some missing 6-8 of the 14 real mappings), which
silently broke joins for whichever teams a given copy happened to omit -- e.g. asset_value_v2.py and breakout_model.py's team-change signal
came out NaN for any player on the Nets, Hornets, Knicks, Spurs, Warriors, Jazz, or Wizards, with no warning anywhere. This is now the only
place this mapping is defined; every script imports `FIX` or calls `canon()` from here instead of keeping its own copy.
"""

FIX = {
    # ESPN fantasy API quirks (current teams)
    "PHL": "PHI", "PHO": "PHX", "NY": "NYK", "SA": "SAS", "GS": "GSW", "UTAH": "UTA", "WSH": "WAS", "BRK": "BKN", "CHO": "CHA",
    "NOR": "NOP", "NO": "NOP",
    # historical franchise relocations/renames (nba_api box-score data)
    "NOH": "NOP", "NJN": "BKN", "SEA": "OKC",
}


def canon(abbr):
    """Normalize one team abbreviation to its current, canonical form. Passes through anything already canonical or unrecognized."""
    return FIX.get(abbr, abbr)
