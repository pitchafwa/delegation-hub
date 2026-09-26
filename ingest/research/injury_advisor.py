"""Injured-player advisor: how long will he be out, how does he look when he returns, and is he worth the roster spot?

Model (injury_return_study.py, official reports 2021-26 + box scores 2010-24):
  * out_curve(group, tier, streak): P(back within k more games | out `streak` games so far), by body-part group and injury tier (checked out of sample: within about 3 points)
  * the return ramp: first game back about 70% of his usual minutes and points, games 2-3 about 85%, later about 90%, and he plays only ~3 of every 4 of his next 10 games
ESPN's public injury feed supplies the injury type, side, detail, its own estimated return date and the news text.
"""
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import injury_common as IC

_M = json.load(open(Path(__file__).resolve().parent / "injury_return_model.json"))
KS = _M["ks"]
SBINS = _M["sbins"]
CELLS = _M["cells"]
RAMP = _M["ramp"]
MAJOR_TXT = re.compile(r"torn|acl|achilles|fractur|surgery|season|rupture|broken|out for the year|indefinite", re.I)
PLAYER_ID = re.compile(r"/id/(\d+)/")
HDR = {"User-Agent": "Mozilla/5.0"}


def _sbin(s):
    for i, (a, b) in enumerate(SBINS):
        if a <= s <= b:
            return i
    return len(SBINS) - 1


def out_curve(group, tier, streak, min_n=40):
    si = _sbin(max(1, streak))
    for key in (f"{group}|{tier}|{si}", f"all|{tier}|{si}", f"all|all|{si}"):
        c = CELLS.get(key)
        if c and c["n"] >= min_n:
            return c["p"], c["n"], key
    c = CELLS[f"all|all|{si}"]
    return c["p"], c["n"], "all"


def median_games(curve):
    """games until he is 50% likely back (linear between the grid points); None if not within the largest k"""
    prev_k, prev_p = 0, 0.0
    for k, p in zip(KS, curve):
        if p >= 0.5:
            return prev_k + (k - prev_k) * (0.5 - prev_p) / max(p - prev_p, 1e-9)
        prev_k, prev_p = k, p
    return None


def quantile_games(curve, q):
    prev_k, prev_p = 0, 0.0
    for k, p in zip(KS, curve):
        if p >= q:
            return prev_k + (k - prev_k) * (q - prev_p) / max(p - prev_p, 1e-9)
        prev_k, prev_p = k, p
    return None


def p_back_within(curve, k):
    """interpolated P(back within k more games); beyond the grid the last value (a lower bound)"""
    if k <= 0:
        return 0.0
    prev_k, prev_p = 0, 0.0
    for kk, p in zip(KS, curve):
        if k <= kk:
            return prev_p + (p - prev_p) * (k - prev_k) / (kk - prev_k)
        prev_k, prev_p = kk, p
    return curve[-1]


def espn_injuries():
    """{espn athlete id: dict(status, type, side, detail, return_date, short, updated)} for every NBA player on ESPN's injury list; {} if the feed is down"""
    try:
        j = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries", headers=HDR, timeout=30).json()
    except Exception:
        return {}
    out = {}
    for t in j.get("injuries", []):
        for x in t.get("injuries", []):
            href = next((l["href"] for l in x.get("athlete", {}).get("links", []) if "playercard" in l.get("rel", [])), "")
            m = PLAYER_ID.search(href)
            if not m:
                continue
            d = x.get("details") or {}
            out[int(m.group(1))] = dict(status=x.get("status"), type=d.get("type"), side=d.get("side"), detail=d.get("detail"), return_date=d.get("returnDate"),
                                        short=x.get("shortComment") or "", updated=x.get("date"))
    return out


def classify(info):
    """(group, tier) for an ESPN injury record using the same parser as the report-based studies"""
    txt = f"Injury/Illness - {info.get('side') or ''} {info.get('type') or ''}; {info.get('detail') or ''}".replace("  ", " ")
    r = IC.parse_reason(txt)
    group, tier = r["group"], r["tier"]
    if MAJOR_TXT.search((info.get("detail") or "") + " " + info.get("short", "")):
        tier = "major"
    tier = {"recovery": "major", "management": "moderate", "other": "moderate"}.get(tier, tier)
    if group in ("other", "illness"):
        group = "other" if group == "other" else "other"
    return group, tier


def ramp_summary(expected_absence):
    """text-ready ramp factors for an absence of about this many games"""
    key = "5-9" if expected_absence < 10 else "10-24"
    r = RAMP.get(key) or next(iter(RAMP.values()))
    g = lambda k: r[str(k)] if str(k) in r else r[k]
    return dict(first=g(1)["minutes"], early=g(3)["minutes"], later=g(8)["minutes"], play_rate=g(8)["played"], fp_later=g(8)["fp"])
