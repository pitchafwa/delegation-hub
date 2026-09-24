"""Pull real DNP-reason data for the 2 most recent seasons (2024-25 via
BoxScoreTraditionalV2, 2025-26 via V3 -- V2 stopped publishing data for
2025-26) to close the gap Tommy flagged: recent-season injury history matters
most, and until now it fell back to an unadjusted proxy.

Per-game caching so an interrupted run doesn't waste completed work, and a
non-empty check so a failed fetch doesn't get cached as "confirmed no data"
(the exact cache-poisoning bug the WRPI/RUPI doc warned about).
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoretraditionalv3

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "dnp_cache"
CACHE.mkdir(exist_ok=True)


def fetch_game(game_id: str, season: str) -> pd.DataFrame:
    cache_path = CACHE / f"{game_id}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)
    for attempt in range(3):
        try:
            if season == "2024-25":
                resp = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, timeout=20)
                df = resp.get_data_frames()[0]
                df = df.rename(columns={"PLAYER_ID": "personId", "PLAYER_NAME": "playerName", "COMMENT": "comment"})
            else:
                resp = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=20)
                df = resp.get_data_frames()[0]
                df["playerName"] = df["firstName"] + " " + df["familyName"]
            df = df[["personId", "playerName", "comment"]].copy()
            df["game_id"] = game_id
            df["season"] = season
            if len(df) == 0:
                raise ValueError("empty response")
            df.to_csv(cache_path, index=False)
            time.sleep(0.6)
            return df
        except Exception as e:
            print(f"  retry {attempt+1} for {game_id}: {e}")
            time.sleep(2.0)
    print(f"  FAILED permanently: {game_id} -- not cached, will retry on next run")
    return pd.DataFrame()


if __name__ == "__main__":
    all_rows = []
    for season, path in [("2024-25", ROOT / "data" / "game_logs" / "nba_api_2024-25.csv"),
                          ("2025-26", ROOT / "data" / "game_logs" / "nba_api_2025-26.csv")]:
        games = pd.read_csv(path)["GAME_ID"].unique()
        print(f"{season}: {len(games)} games")
        for i, gid in enumerate(games):
            gid_str = str(gid).zfill(10)
            df = fetch_game(gid_str, season)
            if len(df):
                all_rows.append(df)
            if (i + 1) % 100 == 0:
                print(f"  {season}: {i+1}/{len(games)} done")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(ROOT / "data" / "recent_dnp_reasons.csv", index=False)
    print(f"\nTotal rows: {len(combined)}")
    print(f"DNP rows with a comment: {combined['comment'].notna().sum()}")
