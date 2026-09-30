"""Pull full per-player box scores (incl. start/bench position) for every real 2025-26 game, so start_bench_study.py can be re-run on real,
recent, recognizable players -- the regular_season_box_scores_2010_2024 dataset it originally used stops at 2023-24. Game ids come from the
existing dnp_cache (already a complete pull of the 2025-26 season, 1230 games) rather than re-deriving them.

Per-game caching (same discipline as pull_recent_dnp_reasons.py): an interrupted run doesn't waste completed work, and a non-empty check keeps
a failed fetch from being cached as "confirmed no data."
Usage: uv run python research/pull_start_position_2025_26.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DNP_CACHE = ROOT / "data" / "dnp_cache"
CACHE = ROOT / "data" / "start_position_2025_26_cache"
CACHE.mkdir(exist_ok=True)

game_ids = []
for f in DNP_CACHE.iterdir():
    d = pd.read_csv(f, nrows=1)
    if len(d) and d.season.iloc[0] == "2025-26":
        game_ids.append(f.stem)
game_ids = sorted(set(game_ids))
print(f"{len(game_ids)} real 2025-26 games to pull")

KEEP = ["gameId", "teamTricode", "personId", "firstName", "familyName", "position", "comment", "minutes", "points", "reboundsTotal",
        "assists", "steals", "blocks", "turnovers", "threePointersMade", "freeThrowsMade", "freeThrowsAttempted", "fieldGoalsAttempted"]


def fetch_game(game_id):
    cache_path = CACHE / f"{game_id}.csv"
    if cache_path.exists():
        return
    for attempt in range(3):
        try:
            resp = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=20)
            df = resp.get_data_frames()[0]
            df["playerName"] = df["firstName"] + " " + df["familyName"]
            df = df[KEEP].copy()
            if len(df) == 0:
                raise ValueError("empty response")
            df.to_csv(cache_path, index=False)
            time.sleep(0.6)
            return
        except Exception as e:
            print(f"  retry {attempt + 1} for {game_id}: {e}")
            time.sleep(2.0)
    print(f"  FAILED permanently: {game_id} -- not cached, will retry on next run")


if __name__ == "__main__":
    for i, gid in enumerate(game_ids):
        fetch_game(gid)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(game_ids)} done")
    print("done pulling; combining...")
    files = sorted(CACHE.glob("*.csv"))
    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_df.to_pickle(ROOT / "data" / "start_position_2025_26.pkl")
    print(f"wrote data/start_position_2025_26.pkl: {len(all_df):,} rows from {len(files)} games")
