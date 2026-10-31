"""Play-by-play for the live in-game projection backtest (2026-10-31, Tommy: project each player's final points from his in-game pace, accurately).
nba_api PlayByPlayV3 (stats.nba.com; works from the PC, blocked from GitHub's servers) for a random sample of a season's games, kept columns only.
Writes data/pbp/pbp_<season>.csv (appends; resumable).
Run from ingest/:  .venv/Scripts/python research/pull_pbp.py [season] [n_games]     (default 2025-26, 600)
"""
import random
import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
season = sys.argv[1] if len(sys.argv) > 1 else "2025-26"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 600
OUT = D / "pbp" / f"pbp_{season}.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
KEEP = ["gameId", "actionNumber", "clock", "period", "teamId", "personId", "playerNameI", "actionType", "subType", "description", "shotResult", "shotValue",
        "isFieldGoal", "scoreHome", "scoreAway"]

games = sorted(pd.read_csv(D / "game_logs" / f"nba_api_{season}.csv", usecols=["GAME_ID"], dtype={"GAME_ID": str})["GAME_ID"].str.zfill(10).unique())
random.Random(7).shuffle(games)
games = games[:N]
done = set(pd.read_csv(OUT, usecols=["gameId"], dtype={"gameId": str})["gameId"].str.zfill(10)) if OUT.exists() else set()
todo = [g for g in games if g not in done]
print(f"{season}: {len(games)} sampled, {len(todo)} to pull", flush=True)
buf = []
for i, gid in enumerate(todo):
    df = None
    for attempt in range(3):
        try:
            df = playbyplayv3.PlayByPlayV3(game_id=gid, timeout=60).get_data_frames()[0]
            break
        except Exception:
            time.sleep(20 * (attempt + 1))
    if df is None or df.empty:
        print("  failed", gid, flush=True)
        continue
    df["gameId"] = gid
    buf.append(df[[c for c in KEEP if c in df.columns]])
    time.sleep(1.2)
    if len(buf) >= 25 or i == len(todo) - 1:
        pd.concat(buf).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        buf = []
        print(f"  {i + 1}/{len(todo)}", flush=True)
print("done")
