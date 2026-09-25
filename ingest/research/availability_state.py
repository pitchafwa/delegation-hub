"""Live side of the availability model: for a player listed on today's official injury report, work out the STATE that the table in availability_table.json
is conditioned on:  first (not listed in the previous 7 days), played (listed before and played his last game), missed (listed before and missed it).
Sources: the previous days' 5:30 PM official reports (nba_injury_reports.py) and ESPN's public per-player game log.  Every step fails soft: unknown -> None,
and the plan falls back to the pooled rate for that status.
"""
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nba_injury_reports as NIR

ET = ZoneInfo("America/New_York")
_T = json.load(open(Path(__file__).resolve().parent / "availability_table.json"))["table"]
HDR = {"User-Agent": "Mozilla/5.0"}
_prior_cache = {}


def p_table(status, tier, state):
    """P(play) for an official status; tier 'rotation'|'bench'; state 'first'|'played'|'missed'|None (None = pooled). Returns None if the status is not in the table."""
    if status == "Out":
        return _T["Out|any|all"]["p"]
    for st in ([state] if state else []) + ["all"]:
        v = _T.get(f"{status}|{tier}|{st}")
        if v:
            return v["p"]
    return None


def _report(d):
    """parsed rows (game_date, key, status) of the 5:30 PM report on date d, cached; tries the 15-minute file name, then the hourly one"""
    if d in _prior_cache:
        return _prior_cache[d]
    rows = []
    c = NIR.fetch(d, 5, 30, "PM")
    if not c:
        for s in ("05PM",):
            try:
                r = requests.get(f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d.isoformat()}_{s}.pdf", headers=HDR, timeout=30)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    c = r.content
            except Exception:
                pass
    if c:
        try:
            rows = [(g, k, s) for g, k, s in NIR.parse_pdf(c) if g == d.isoformat()]
        except Exception:
            rows = []
    _prior_cache[d] = rows
    return rows


def prior_listings(today, days=7):
    """{player key: (game_date, status)} = his most recent listing on a report in the previous `days` days (before today)"""
    out = {}
    for k in range(days, 0, -1):
        d = today - timedelta(days=k)
        for g, key, status in _report(d):
            out[key] = (g, status)
    return out


_gl_cache = {}


def game_dates(espn_id):
    """{ET date iso: minutes} for the games the player has actually played, from ESPN's public game log; None if unavailable"""
    if espn_id in _gl_cache:
        return _gl_cache[espn_id]
    res = None
    try:
        j = requests.get(f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}/gamelog", headers=HDR, timeout=30).json()
        ev = j.get("events", {})
        res = {}
        for st in j.get("seasonTypes", []):
            for cat in st.get("categories", []):
                for e in cat.get("events", []):
                    meta = ev.get(e["eventId"])
                    if not meta:
                        continue
                    d = datetime.fromisoformat(meta["gameDate"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()
                    try:
                        mins = float(e["stats"][0])
                    except Exception:
                        mins = 1.0
                    res[d] = mins
    except Exception:
        res = None
    _gl_cache[espn_id] = res
    return res


def state_for(key, espn_id, today, prior):
    """'first' | 'played' | 'missed' | None"""
    p = prior.get(key)
    if p is None:
        return "first"
    g, status = p
    if status == "Out":
        return "missed"
    dates = game_dates(espn_id)
    if dates is None:
        return None
    return "played" if dates.get(g, 0) > 0 else "missed"
