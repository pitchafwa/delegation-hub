"""Pull real historical NBA player-season data from stats.nba.com (via nba_api),
for building and backtesting the existing-player dynasty valuation model.

Caches each season's raw response to research/data/raw/ so re-runs are free.
Polite pacing: NBA's stats endpoint is unauthenticated but rate-limit-sensitive.
"""
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, playerindex

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2010, 2026)]  # 2010-11 .. 2025-26


def season_csv(measure: str, season: str) -> Path:
    return RAW_DIR / f"{measure}_{season}.csv"


def pull_season(measure: str, season: str, per_mode: str) -> pd.DataFrame:
    path = season_csv(measure, season)
    if path.exists():
        return pd.read_csv(path)
    print(f"  fetching {measure} {season} ...")
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed=per_mode,
        measure_type_detailed_defense=measure,
    )
    df = resp.get_data_frames()[0]
    df.to_csv(path, index=False)
    time.sleep(1.0)
    return df


def main():
    print("=== Base (Totals) ===")
    base_frames = []
    for season in SEASONS:
        df = pull_season("Base", season, "Totals")
        df["SEASON"] = season
        base_frames.append(df)
    base_all = pd.concat(base_frames, ignore_index=True)
    base_all.to_csv(ROOT / "data" / "player_season_base.csv", index=False)
    print(f"Base: {len(base_all)} player-seasons across {len(SEASONS)} seasons")

    print("\n=== Advanced (PerGame) ===")
    adv_frames = []
    for season in SEASONS:
        df = pull_season("Advanced", season, "PerGame")
        df["SEASON"] = season
        adv_frames.append(df)
    adv_all = pd.concat(adv_frames, ignore_index=True)
    adv_all.to_csv(ROOT / "data" / "player_season_advanced.csv", index=False)
    print(f"Advanced: {len(adv_all)} player-seasons")

    print("\n=== Player bio index (historical) ===")
    bio_path = ROOT / "data" / "player_bio.csv"
    if bio_path.exists():
        bio = pd.read_csv(bio_path)
    else:
        resp = playerindex.PlayerIndex(season="2025-26", historical_nullable="1")
        bio = resp.get_data_frames()[0]
        bio.to_csv(bio_path, index=False)
    print(f"Bio: {len(bio)} players")


if __name__ == "__main__":
    main()
