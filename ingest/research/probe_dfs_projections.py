"""Probe the three DFS sites for per-game projections. Run on a game day once the season has started (before then every site shows no slate):

    uv run python research/probe_dfs_projections.py

For each site it fetches the DraftKings projection page, says whether a slate with projections is present, and saves the raw HTML to
data/dfs_probe/<date>_<site>.html so the exact table markup can be read and a real parser written (that is the missing step: with no slate
live in the offseason the markup cannot be verified).

WHICH SCORING FORMAT TO USE: DraftKings. Measured on 46,520 player-games (2024-25 and 2025-26): correlation with this league's points
DraftKings 0.993 (residual SD 2.02 pts) vs FanDuel/Yahoo 0.9915 (2.23; the two are the same formula). DK counts 3PM (+0.5), double/triple-doubles
and weights STL/BLK 2, all closer to our league. Convert:  league points = 1.206 x DraftKings points  (intercept ~0).
Robots: dailyfantasyfuel.com allows everything except /lineup/*; rotowire.com disallows generic crawlers (check /robots.txt and their terms before
automating; their optimizer is JavaScript-rendered); fantasydata.com's optimizer is also JavaScript-driven and may need a login for projections.
"""
import re
import sys
from datetime import date
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "data" / "dfs_probe"
OUT.mkdir(parents=True, exist_ok=True)
SITES = {"dailyfantasyfuel": "https://www.dailyfantasyfuel.com/nba/projections/draftkings",
         "rotowire": "https://www.rotowire.com/daily/nba/optimizer.php",
         "fantasydata": "https://fantasydata.com/nba/optimizer"}
NO_SLATE = ("No Contests Scheduled", "No slates found", "no games scheduled")
for name, url in SITES.items():
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        html = r.text
    except Exception as ex:
        print(f"{name:17s} error {ex}")
        continue
    empty = any(m.lower() in html.lower() for m in NO_SLATE)
    (OUT / f"{date.today()}_{name}.html").write_text(html, encoding="utf-8")
    rows = len(re.findall(r"data-(?:player|name|pid)", html))
    print(f"{name:17s} HTTP {r.status_code}  {len(html) // 1024} KB  no-slate marker: {empty}  player-like attributes: {rows}")
print("saved snapshots in", OUT)
