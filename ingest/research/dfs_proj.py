"""Tonight's per-player projection from DailyFantasyFuel's free DraftKings table, converted to this league's points (x1.206). It is the plan's standard 'what will this player do on this date'
number for any day the table lists: it already folds in injuries and who else is out, minutes and rotation changes, coaching comments, matchup and pace, which our season-long level does not.
It never replaces the career trajectories or season-long projections; it only overrides the per-DAY level (`lvl_by_date`) for days it covers.

  fetch()          -> {normalized name: {date: row}} (one polite request per call; {} if the page is down or empty)
  apply(players, plan_days)  sets pl["lvl_by_date"][date] and pl["dfs_dates"] for matched players, pl["dfs_out"] for players the table lists as out; returns a summary dict

Rules (OPENING_NIGHT_CHECKLIST 5): normal browser user agent, no login, no premium pages, one fetch per plan run (a few a day), the raw table is not published, and the raw numbers are not written to
committed files (only the plan's own expected points are). Assumption to grade on real nights: the projection is conditional on playing (a Questionable player's number is what he does if he plays).
"""
import os
import re
import unicodedata

from pull_dfs_projections import parse, URL
import requests

BLEND_BASE = float(os.environ.get("DFS_BLEND", "1.0"))
UNLISTED_MIN_ROWS = 8                                   # a team counts as 'slate posted' for a date when the table lists at least this many of its players
UNLISTED_P = 0.1                                               # chance of playing for a player the posted slate leaves out
FA_CARRY = float(os.environ.get("DFS_FA_CARRY", "0.6"))      # free agents only: days the table does not list yet are valued 60% on what the table says he does NOW (role, minutes) and 40% on his season level          # share of the per-day level taken from the table (1.0 = fully replace the base level on listed days)


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def fetch():
    try:
        r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}, timeout=30)
        r.raise_for_status()
        rows = parse(r.text)
    except Exception as ex:
        print("dfs projections unavailable:", repr(ex)[:90])
        return {}
    out = {}
    for x in rows:
        out.setdefault(norm(x["name"]), {})[x["game_date"]] = x
    return out


def apply(players, plan_days, table=None, fa_ids=frozenset()):
    table = fetch() if table is None else table
    dates = {d.isoformat() for d in plan_days}
    posted = {}                                                    # (date, team) -> rows the table lists: a team whose slate is posted lists everyone expected to play
    for byd in table.values():
        for dt, x in byd.items():
            posted[(dt, (x["team"] or "").upper())] = posted.get((dt, (x["team"] or "").upper()), 0) + 1
    used, out_n, rows = 0, 0, sum(len(v) for v in table.values())
    for pl in players:
        byd = table.get(norm(pl["name"]))
        if not byd:
            continue
        for dt, x in byd.items():
            if dt not in dates:
                continue
            if x["team"] and pl.get("team") and x["team"].upper() != pl["team"].upper() and norm(pl["name"]) != norm(x["name"]):
                continue
            if x["dk_proj"] <= 0:
                # zero = not expected to play. Graded 2026-10-23: zeros WITHOUT an injury tag were mostly players ruled out (Porzingis, Monk, Stewart...), so only a team
                # whose slate is not really posted yet (few rows) keeps the benefit of the doubt
                if (x["inj"] or "").upper() in ("O", "OUT") or posted.get((dt, (x["team"] or "").upper()), 0) >= UNLISTED_MIN_ROWS:
                    pl.setdefault("dfs_out", set()).add(dt)
                    out_n += 1
                continue
            lvl = x["league_proj"]
            base = pl["level"]
            pl.setdefault("lvl_by_date", {})[dt] = round(BLEND_BASE * lvl + (1 - BLEND_BASE) * base, 1)
            pl.setdefault("dfs_dates", set()).add(dt)
            pl.setdefault("dfs_tag", {})[dt] = x["inj"]
            used += 1
    for pl in players:                                           # a player on a team whose slate is posted but missing from the table is not expected to play (Caleb Love, Cam Whitmore...)
        byd = table.get(norm(pl["name"])) or {}
        for dt in dates:
            if dt not in byd and posted.get((dt, (pl.get("team") or "").upper()), 0) >= UNLISTED_MIN_ROWS and not pl.get("ir") and pl.get("espn_id") in fa_ids:     # free agents only: for your own players a name mismatch must not bench them
                pl.setdefault("dfs_unlisted", set()).add(dt)
    for pl in players:                                           # free-agent streamers: carry what the table says about his current role to the days it has not posted yet
        if pl.get("espn_id") in fa_ids and pl.get("dfs_dates"):
            vals = [pl["lvl_by_date"][d] for d in pl["dfs_dates"] if d in pl.get("lvl_by_date", {})]
            if vals:
                pl["dfs_carry"] = round(sum(vals) / len(vals), 1)
    return {"enabled": rows > 0, "rows": rows, "player_days_used": used, "listed_out": out_n}
