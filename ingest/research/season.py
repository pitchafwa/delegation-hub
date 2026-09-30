"""Single source of truth for "what NBA/ESPN fantasy season is it right now" -- so nothing needs manual updating each year.

ESPN's fantasy API labels a season by its ENDING calendar year (the 2026-27 season is seasonId 2027). Cutover is set at July 1: free
agency, the draft, and keeper decisions for the upcoming season are already underway by early summer, well before opening night in
October, so any script run between July and the following June should resolve to the season people mean by "this season" in that window.

Before 2026-09-29, about half a dozen scripts each hardcoded `SEASON_ID = 2027` (or `SEASON = "2026-27"`) separately, all needing a manual
find-and-replace once a year. This is now the only place that number is decided; everything else imports it.

Note: this only derives the SEASON LABEL/ID from a date, not exact real-world dates like the actual opening night or playoff schedule --
those aren't computable from a formula (the NBA sets them) and still need a manual update each year wherever they're hardcoded (e.g.
build_week_plan.py's SEASON_START, refresh_all.py's OPENER).
"""
from datetime import date


def current_season_id(today=None):
    """ESPN's seasonId for "the season we mean right now" -- e.g. 2027 for the 2026-27 season, from any date July 2026 through June 2027."""
    d = today or date.today()
    return d.year + 1 if d.month >= 7 else d.year


def current_season_str(today=None):
    """e.g. "2026-27" for the same window current_season_id() covers."""
    sid = current_season_id(today)
    return f"{sid - 1}-{str(sid)[2:]}"
