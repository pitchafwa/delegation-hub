"""Prepare the per-player chronological game sequences the Kalman filter
needs: sorted by date, with days-since-last-game and real age-at-that-game
(computed from birthdate where available, per the WRPI/RUPI lesson that age
should come from birthdate, not a rounded integer column, wherever possible).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
games = pd.read_csv(ROOT / "data" / "game_logs_unified.csv")
games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
bio_small = bio[["PERSON_ID"]].drop_duplicates()
# player_bio.csv (PlayerIndex) doesn't expose raw birthdate directly in the
# columns we pulled -- fall back to the season-level AGE column (rounded
# integer, known limitation) interpolated to a real date within that season
# rather than treating every game in a season as the same exact age.
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")[["PLAYER_ID", "SEASON", "AGE"]]
games = games.merge(season_base, on=["PLAYER_ID", "SEASON"], how="left")

# Approximate a real per-game age: seasons run ~Oct-Jun: interpolate AGE
# linearly across the season's real date span instead of freezing it, so
# age drift within a season isn't completely flattened (still an
# approximation -- flagged, not hidden).
games = games.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
season_bounds = games.groupby(["PLAYER_ID", "SEASON"])["GAME_DATE"].agg(["min", "max"]).reset_index()
season_bounds.columns = ["PLAYER_ID", "SEASON", "season_min_date", "season_max_date"]
games = games.merge(season_bounds, on=["PLAYER_ID", "SEASON"], how="left")
span_days = (games["season_max_date"] - games["season_min_date"]).dt.days.replace(0, np.nan)
frac_through = (games["GAME_DATE"] - games["season_min_date"]).dt.days / span_days
games["AGE_AT_GAME"] = games["AGE"] - 0.5 + frac_through.fillna(0.5)  # AGE is age-during-season; spread +/-0.5yr across it

games["DAYS_SINCE_LAST"] = games.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
games["DAYS_SINCE_LAST"] = games["DAYS_SINCE_LAST"].fillna(180)  # first game of career: treat as a fresh start

games = games.dropna(subset=["AGE"])
games.to_csv(ROOT / "data" / "kalman_input.csv", index=False)
print(f"{len(games)} rows ready. Players: {games['PLAYER_ID'].nunique()}. "
      f"Median days between games: {games['DAYS_SINCE_LAST'].median():.1f}")
