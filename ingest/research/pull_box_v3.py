"""Starters and did-not-play reasons for 2024-25 onward (the 2010-24 box-score files already have them): one NBA box score per game
(nba_api BoxScoreTraditionalV3), written to data/game_logs/box_v3_<season>.csv with the same columns as regular_season_box_scores_2010_2024_part_*.csv.
Resumable: games already in the file are skipped. Short-term projection program, Phase 1 (RESEARCH_short_term_projection_plan.md).
Run from ingest/:  .venv/Scripts/python research/pull_box_v3.py [season ...]     (default: 2024-25 2025-26 2026-27)
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data" / "game_logs"
SEASONS = [a for a in sys.argv[1:] if "-" in a] or ["2024-25", "2025-26", "2026-27"]
COLS = {"teamId": "teamId", "teamTricode": "teamTricode", "personId": "personId", "firstName": "firstName", "familyName": "familyName", "position": "position",
        "comment": "comment", "minutes": "minutes"}

for season in SEASONS:
    src = D / f"nba_api_{season}.csv"
    if not src.exists():
        continue
    games = pd.read_csv(src, usecols=["GAME_ID", "GAME_DATE"], dtype={"GAME_ID": str}).drop_duplicates().sort_values("GAME_DATE")
    out = D / f"box_v3_{season}.csv"
    done = set(pd.read_csv(out, usecols=["gameId"], dtype={"gameId": str})["gameId"]) if out.exists() else set()
    todo = [(g, d) for g, d in zip(games["GAME_ID"], games["GAME_DATE"]) if g.zfill(10) not in done]
    print(f"{season}: {len(games)} games, {len(todo)} to pull", flush=True)
    buf = []
    for i, (gid, gdate) in enumerate(todo):
        for attempt in range(3):
            try:
                df = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=gid.zfill(10), timeout=60).get_data_frames()[0]
                break
            except Exception as ex:
                time.sleep(30 * (attempt + 1))
                df = None
        if df is None or df.empty:
            print("  failed", gid, flush=True)
            continue
        df = df.assign(gameId=gid.zfill(10), game_date=gdate, season_year=season)
        buf.append(df)
        time.sleep(1.5)
        if len(buf) >= 25 or i == len(todo) - 1:
            chunk = pd.concat(buf)
            chunk.to_csv(out, mode="a", header=not out.exists(), index=False)
            buf = []
            print(f"  {season}: {i + 1}/{len(todo)}", flush=True)
print("done")
