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


# MINUTES FROM PROPS (2026-11-13, Tommy's method). A book rarely offers every stat: tonight's test had points for all 95 players but steals for 36, blocks for 22,
# and no turnover or free-throw market at all, and in this league steals and blocks are worth 3 each. So: each prop that exists, divided by his DELCO per-minute
# rate for that stat, says how many minutes the market expects; average those (points weigh most, his DELCO minutes count as a prior), then every stat with no
# prop = those minutes x his DELCO rate. Free throws and turnovers follow scoring volume as well as minutes, so they are scaled halfway toward the points prop's
# usage (points prop / (minutes x his scoring rate)).
MIN_W = {"PTS": 4.0, "REB": 2.0, "AST": 2.0, "FG3M": 1.0, "STL": 0.5, "BLK": 0.5}
MIN_PRIOR_W = 1.5
STATS6 = ("PTS", "REB", "AST", "FG3M", "STL", "BLK")


def fantasy_from_rates(means, pm):
    """league points from prop means + DELCO per-minute rates pm {'MIN','PTS','REB','AST','STL','BLK','FG3M','FTM','FTA','TOV'}; returns (points, used, detail) or None"""
    if not pm or not pm.get("MIN") or not pm.get("PTS"):
        return None
    m0 = float(pm["MIN"])
    num, den = MIN_PRIOR_W * m0, MIN_PRIOR_W
    for st, mu in means.items():
        r = pm.get(st) or 0.0
        if st in MIN_W and r > 1e-4:
            w = MIN_W[st] * min(1.0, (r * m0) / 2.0) if st != "PTS" else MIN_W[st]      # a stat he rarely records says little about minutes
            num += w * (mu / r)
            den += w
    m = min(max(num / den, 0.4 * m0), min(44.0, 1.6 * m0 + 4.0))
    u = min(max(means["PTS"] / (m * pm["PTS"]), 0.6), 1.6) if "PTS" in means else 1.0
    vol = 0.5 + 0.5 * u
    line = {}
    for st in STATS6:
        line[st] = means[st] if st in means else m * (pm.get(st) or 0.0) * (vol if st == "FG3M" else 1.0)
    ftm, fta, tov = m * (pm.get("FTM") or 0.0) * vol, m * (pm.get("FTA") or 0.0) * vol, m * (pm.get("TOV") or 0.0) * vol
    pts = line["PTS"] + 1.5 * line["REB"] + 2 * line["AST"] + 3 * line["STL"] + 3 * line["BLK"] + line["FG3M"] + 2 * ftm - fta - tov
    used = {k: round(v, 2) for k, v in means.items()}
    detail = {"min": round(m, 1), "usage": round(u, 2), "filled": {k: round(v, 2) for k, v in line.items() if k not in means}, "ftm": round(ftm, 2), "fta": round(fta, 2), "tov": round(tov, 2)}
    return pts, used, detail


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


# ---------------------------------------------------------------------------------------------------------------------------------------------
# FREE SOURCE (2026-11-13, Tommy; DailyFantasyFuel cut its free table to the top 25 on 11-12): ESPN's public odds feed carries DraftKings player props
# for every game (sports.core.api.espn.com .../odds/100/propBets; ~15 players a game with a points line). Each 'Total <stat>' line comes as two entries
# with no side label: the FIRST listed is the over (checked against DraftKings' own ladder lines: 49 of 50). Turnovers, free throws and triple-doubles
# have no prop here, so fantasy() fills them from ESPN's projected stat line as before. Keyed by ESPN athlete id (= the fantasy playerId).
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_TYPES = {"Total Points": "PTS", "Total Rebounds": "REB", "Total Assists": "AST", "Total 3-Point Field Goals": "FG3M",
              "Total Steals": "STL", "Total Blocks": "BLK"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _american(v):
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def fetch_espn_props(dates, provider=100, sess=None):
    """{espn athlete id (str): {iso date: {stat: mean}}} for NBA games on `dates` that have not tipped yet, plus meta. Free, no key."""
    S = sess or requests.Session()
    out, meta = {}, {"source": "ESPN (DraftKings lines)", "events": 0, "players": 0, "requests": 0}
    for d in dates:
        sb = S.get(ESPN_SITE, params={"dates": d.strftime("%Y%m%d")}, headers=UA, timeout=30)
        meta["requests"] += 1
        if sb.status_code != 200:
            continue
        for e in sb.json().get("events", []):
            if (e.get("competitions") or [{}])[0].get("status", {}).get("type", {}).get("state") != "pre":
                continue                                            # tipped or final: in-game lines are not a pregame projection
            eid, items, page = e["id"], [], 1
            while True:
                r = S.get(f"{ESPN_CORE}/events/{eid}/competitions/{eid}/odds/{provider}/propBets", params={"limit": 1000, "page": page}, headers=UA, timeout=30)
                meta["requests"] += 1
                if r.status_code != 200:
                    break
                j = r.json()
                items += j.get("items", [])
                if page >= j.get("pageCount", 1):
                    break
                page += 1
            if not items:
                continue
            meta["events"] += 1
            seen = {}
            for x in items:
                stat = ESPN_TYPES.get((x.get("type") or {}).get("name"))
                if not stat:
                    continue
                aid = x["athlete"]["$ref"].split("/athletes/")[1].split("?")[0]
                line = ((x.get("current") or {}).get("target") or {}).get("value")
                px = _american(((x.get("odds") or {}).get("american") or {}).get("value"))
                if line is None or px is None:
                    continue
                seen.setdefault((aid, stat, float(line)), []).append(px)
            best = {}
            for (aid, stat, line), pxs in seen.items():
                if len(pxs) < 2:
                    continue
                p = fair_over(pxs[0], pxs[1])                       # first listed = over
                cand = -abs(p - 0.5)                                # the main line: closest to 50/50
                if (aid, stat) not in best or cand > best[(aid, stat)][0]:
                    best[(aid, stat)] = (cand, line, p)
            for (aid, stat), (_, line, p) in best.items():
                try:
                    out.setdefault(aid, {}).setdefault(d.isoformat(), {})[stat] = mean_from_line(stat, line, p)
                except Exception:
                    continue
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
