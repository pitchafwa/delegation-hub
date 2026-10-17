"""Pull DailyFantasyFuel's free NBA DraftKings projections table (one fetch, polite) and keep it LOCALLY for grading and blending.

  uv run python research/pull_dfs_projections.py            (from ingest/)

Rules we set (OPENING_NIGHT_CHECKLIST item 5): robots.txt allows /nba/projections/draftkings; their terms do not address automated personal retrieval; so: one or two fetches a day, a normal browser
user agent, no login, no premium pages, personal use only, and the raw numbers are NOT published on the site (data/dfs/ is gitignored with the rest of research/data/).

The page lists one row per player per slate game with everything in data- attributes: data-ppg_proj = projected DraftKings points, data-start_date, data-team, data-opp, data-loc (@ = away),
data-inj (injury tag), data-salary, data-l5_avg / l10_avg / szn_avg = his DK points average recently, data-depth_rank. League points = 1.206 x DraftKings points (measured on 46,520 player-games:
research/probe_dfs_projections.py docstring).

Writes data/dfs/<start_date>.csv for each game date on the page (replaced on each fetch, with the fetch time) and appends every row to data/dfs/log.csv (fetch time + row) so a projection can be
graded against what he scored. Columns: fetched_utc, game_date, name, team, opp, away, pos, inj, salary, dk_proj, league_proj, l5, l10, season_avg, depth_rank, player_id.
"""
import csv
import html
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")
URL = "https://www.dailyfantasyfuel.com/nba/projections/draftkings"
OUT = Path(__file__).resolve().parent / "data" / "dfs"
DK_TO_LEAGUE = 1.206
COLS = ["fetched_utc", "game_date", "name", "team", "opp", "away", "pos", "inj", "salary", "dk_proj", "league_proj", "l5", "l10", "season_avg", "depth_rank", "player_id"]


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse(page):
    """rows from the projections table; every <tr class=...projections-listing...> carries its numbers in data- attributes"""
    rows = []
    for m in re.finditer(r"<tr\s+class=\"[^\"]*projections-listing[^\"]*\"([^>]*)>", page):
        a = dict(re.findall(r'data-([\w]+)\s*=\s*"([^"]*)"', m.group(1)))
        dk = num(a.get("ppg_proj"))
        if not a.get("name") or dk is None:
            continue
        rows.append({"game_date": a.get("start_date"), "name": html.unescape(a["name"]).strip(), "team": a.get("team"), "opp": a.get("opp"), "away": a.get("loc") == "@",
                     "pos": a.get("pos"), "inj": a.get("inj") or "", "salary": num(a.get("salary")), "dk_proj": dk, "league_proj": round(dk * DK_TO_LEAGUE, 1),
                     "l5": num(a.get("l5_avg")), "l10": num(a.get("l10_avg")), "season_avg": num(a.get("szn_avg")), "depth_rank": a.get("depth_rank"), "player_id": a.get("player_id")})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}, timeout=30)
    r.raise_for_status()
    rows = parse(r.text)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_date = {}
    for x in rows:
        x["fetched_utc"] = now
        by_date.setdefault(x["game_date"], []).append(x)
    for d, rs in by_date.items():
        with open(OUT / f"{d}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(rs)
    new = not (OUT / "log.csv").exists()
    with open(OUT / "log.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} projections for {sorted(by_date)}: " + ", ".join(f"{x['name']} {x['dk_proj']} DK = {x['league_proj']} league" for x in rows[:5]))


if __name__ == "__main__":
    main()
