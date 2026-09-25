"""Sportsbook player props -> this league's fantasy-point projection.

A prop line is the market's estimate of a player's stat for tonight, and it already reflects injuries to teammates, minutes, matchup and
pace (it is conditional on the player playing: books void the bet if he doesn't). We convert each line + its over/under prices into a
MEAN for that stat, then score the means with the league formula:
    PTS + 1.5 REB + 2 AST + 3 STL + 3 BLK + 3PM + 2 FTM - FTA - TOV (+ 3 TD3)
No prop market exists for FTM, FTA or TD3 (and turnovers/steals/blocks are not always offered), so those come from ESPN's projected
stat line (free-throw volume is scaled by props points / ESPN points); anything the book doesn't list falls back to ESPN's number.

Line -> mean: remove the book's margin (fair P(over) from the two prices, averaged over books), then find the mean mu of a
negative-binomial (Poisson when variance is not above the mean) whose P(stat > line) equals that fair probability. Game-to-game
variance vs mean was fitted on 2023-26 NBA game logs (VAR below), e.g. points var = 4.13 mu - 0.052 mu^2.

Data source: The Odds API (https://the-odds-api.com), needs an API key in ODDS_API_KEY; without one the module is inert and the
weekly plan uses ESPN projections. Cost: 1 credit per market per event per bookmaker-group, so ~7 credits per game per day.
"""
import math
import os
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from scipy.optimize import brentq
from scipy.stats import nbinom, poisson

ET = ZoneInfo("America/New_York")
VAR = {"PTS": (4.13, -0.052), "REB": (1.51, -0.004), "AST": (1.39, -0.012), "STL": (1.12, -0.036),
       "BLK": (1.05, 0.087), "TOV": (1.13, -0.039), "FG3M": (1.30, -0.054)}
MARKETS = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_threes": "FG3M",
           "player_blocks": "BLK", "player_steals": "STL", "player_turnovers": "TOV"}
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def implied(american):
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)


def fair_over(price_over, price_under):
    a, b = implied(price_over), implied(price_under)
    return a / (a + b)


def _var(stat, mu):
    a, b = VAR[stat]
    return max(a * mu + b * mu * mu, 1.02 * mu)


def _sf_ge(stat, mu, k):
    """P(X >= k)"""
    v = _var(stat, mu)
    if v <= mu * 1.001:
        return float(poisson.sf(k - 1, mu))
    r = mu * mu / (v - mu)
    return float(nbinom.sf(k - 1, r, r / (r + mu)))


def mean_from_line(stat, line, p_over):
    """mean of the stat implied by 'over `line` has fair probability p_over'"""
    p_over = min(max(p_over, 0.03), 0.97)
    k = math.floor(line) + 1                       # over 24.5 means at least 25
    f = lambda mu: _sf_ge(stat, mu, k) - p_over
    return brentq(f, 0.02, 120.0)


def fantasy(means, espn_avg, espn_applied):
    """league-scored points from stat means (props where present, ESPN's projected stat line otherwise).
    espn_avg: ESPN projected per-game stats {'PTS','REB','AST','STL','BLK','3PM','TO','FTM','FTA'}; espn_applied: ESPN's own points projection."""
    e = espn_avg or {}
    g = lambda k, alt=None: e.get(k, e.get(alt, 0.0) if alt else 0.0) or 0.0
    espn_pts = g("PTS")
    comp = {"PTS": g("PTS"), "REB": g("REB"), "AST": g("AST"), "STL": g("STL"), "BLK": g("BLK"), "FG3M": g("3PM"), "TOV": g("TO")}
    ftm, fta = g("FTM"), g("FTA")
    resid = 0.0
    if espn_applied:
        base = comp["PTS"] + 1.5 * comp["REB"] + 2 * comp["AST"] + 3 * comp["STL"] + 3 * comp["BLK"] + comp["FG3M"] + 2 * ftm - fta - comp["TOV"]
        resid = espn_applied - base                 # triple-double bonus and anything else ESPN's line carries
    used = {}
    for st, mu in means.items():
        if st in comp:
            comp[st] = mu
            used[st] = round(mu, 2)
    if "PTS" in means and espn_pts > 0:
        ratio = min(max(comp["PTS"] / espn_pts, 0.6), 1.6)
        ftm, fta = ftm * ratio, fta * ratio
    pts = comp["PTS"] + 1.5 * comp["REB"] + 2 * comp["AST"] + 3 * comp["STL"] + 3 * comp["BLK"] + comp["FG3M"] + 2 * ftm - fta - comp["TOV"] + resid
    return pts, used


def fetch_player_props(api_key, dates, bookmakers="draftkings,fanduel"):
    """{normalized player name: {iso date: {stat: mean}}} for NBA games whose ET date is in `dates`, plus a small meta dict."""
    want = {d.isoformat() for d in dates}
    ev = requests.get(f"{BASE}/events", params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    ev.raise_for_status()
    out, meta = {}, {"events": 0, "players": 0, "remaining": None, "used": None}
    for e in ev.json():
        d = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(ET).date().isoformat()
        if d not in want:
            continue
        r = requests.get(f"{BASE}/events/{e['id']}/odds", params={"apiKey": api_key, "regions": "us", "bookmakers": bookmakers, "oddsFormat": "american",
                                                                   "markets": ",".join(MARKETS)}, timeout=30)
        if r.status_code != 200:
            continue
        meta["events"] += 1
        meta["remaining"], meta["used"] = r.headers.get("x-requests-remaining"), r.headers.get("x-requests-used")
        lines = {}      # (player, stat, line) -> list of fair over-probabilities from each book
        for bm in r.json().get("bookmakers", []):
            for mk in bm.get("markets", []):
                stat = MARKETS.get(mk["key"])
                if not stat:
                    continue
                by = {}
                for o in mk.get("outcomes", []):
                    by.setdefault((o.get("description"), o.get("point")), {})[o["name"]] = o["price"]
                for (player, line), px in by.items():
                    if player and line is not None and "Over" in px and "Under" in px:
                        lines.setdefault((player, stat, float(line)), []).append(fair_over(px["Over"], px["Under"]))
        best = {}       # (player, stat) -> (line, mean fair over prob, books)
        for (player, stat, line), ps in lines.items():
            cand = (len(ps), -abs(sum(ps) / len(ps) - 0.5))       # prefer the main line: most books, then closest to 50/50
            if (player, stat) not in best or cand > best[(player, stat)][0]:
                best[(player, stat)] = (cand, line, sum(ps) / len(ps))
        for (player, stat), (_, line, p) in best.items():
            try:
                mu = mean_from_line(stat, line, p)
            except Exception:
                continue
            out.setdefault(norm(player), {}).setdefault(d, {})[stat] = mu
    meta["players"] = len(out)
    return out, meta


if __name__ == "__main__":
    # self-check of the conversion math
    for stat, line, p in [("PTS", 24.5, 0.5), ("PTS", 24.5, 0.6), ("REB", 9.5, 0.5), ("AST", 6.5, 0.45), ("FG3M", 2.5, 0.5), ("BLK", 1.5, 0.35), ("STL", 1.5, 0.5)]:
        mu = mean_from_line(stat, line, p)
        print(f"{stat:5s} line {line:5.1f}  fair P(over) {p:.2f}  -> mean {mu:5.2f}   (check P(over) = {_sf_ge(stat, mu, math.floor(line) + 1):.3f})")
    print("fair_over(-115, -105) =", round(fair_over(-115, -105), 3))
    sample = {"PTS": mean_from_line("PTS", 24.5, .5), "REB": mean_from_line("REB", 9.5, .5), "AST": mean_from_line("AST", 6.5, .5)}
    print("example fantasy points:", round(fantasy(sample, {"PTS": 24, "REB": 9, "AST": 6, "STL": 1.2, "BLK": 1, "3PM": 2, "TO": 3, "FTM": 5, "FTA": 6}, 47.0)[0], 1))
