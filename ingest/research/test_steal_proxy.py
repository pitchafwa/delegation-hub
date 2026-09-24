"""Real test: does DEFLECTIONS (a hustle-stat proxy) predict a player's
FUTURE steal rate better than their own PAST steal rate does? Hustle stats
only exist from 2015-16 onward (confirmed via nba_api), so this uses that
subset, not the full 2010-2024 range.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguehustlestatsplayer
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "hustle_cache"
CACHE.mkdir(exist_ok=True)

SEASONS = [f"{y}-{str(y+1)[2:]}" for y in range(2015, 2025)]  # 2015-16 .. 2024-25

frames = []
for season in SEASONS:
    path = CACHE / f"{season}.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        resp = leaguehustlestatsplayer.LeagueHustleStatsPlayer(season=season, season_type_all_star="Regular Season", per_mode_time="PerGame")
        df = resp.get_data_frames()[0]
        df.to_csv(path, index=False)
        time.sleep(0.5)
    df["SEASON"] = season
    frames.append(df)
hustle = pd.concat(frames, ignore_index=True)
hustle["SEASON_YEAR"] = hustle["SEASON"].apply(lambda s: int(s.split("-")[0]))

base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
base["SEASON_YEAR"] = base["SEASON"].apply(lambda s: int(s.split("-")[0]))
base["STL_PG"] = base["STL"] / base["GP"].replace(0, np.nan)
base_small = base[["PLAYER_ID", "SEASON_YEAR", "STL_PG", "GP", "MIN"]].copy()
base_small["MIN_PG"] = base_small["MIN"] / base_small["GP"].replace(0, np.nan)

merged = hustle[["PLAYER_ID", "SEASON_YEAR", "DEFLECTIONS", "CHARGES_DRAWN", "LOOSE_BALLS_RECOVERED", "G", "MIN"]].merge(
    base_small, on=["PLAYER_ID", "SEASON_YEAR"], how="inner"
)
merged = merged[(merged["GP"] >= 20) & (merged["MIN_PG"] >= 10)]

# Build the t -> t+1 STL_PG target
target = base_small[["PLAYER_ID", "SEASON_YEAR", "STL_PG", "GP"]].rename(
    columns={"SEASON_YEAR": "TARGET_YEAR", "STL_PG": "TARGET_STL_PG", "GP": "TARGET_GP"}
)
merged["TARGET_YEAR"] = merged["SEASON_YEAR"] + 1
merged = merged.merge(target, on=["PLAYER_ID", "TARGET_YEAR"], how="inner")
merged = merged[merged["TARGET_GP"] >= 20]

print(f"Usable player-season transitions with hustle data: {len(merged)}")

naive_rho, _ = spearmanr(merged["STL_PG"], merged["TARGET_STL_PG"])
deflect_rho, _ = spearmanr(merged["DEFLECTIONS"], merged["TARGET_STL_PG"])
print(f"Current STL/g -> next STL/g:        Spearman = {naive_rho:.4f}  (baseline)")
print(f"Current DEFLECTIONS/g -> next STL/g: Spearman = {deflect_rho:.4f}")

# Does adding deflections to current steals improve on steals alone? Simple OLS check.
import numpy as np
X1 = merged[["STL_PG"]].to_numpy()
X2 = merged[["STL_PG", "DEFLECTIONS"]].to_numpy()
y = merged["TARGET_STL_PG"].to_numpy()

from numpy.linalg import lstsq
def fit_and_corr(X, y):
    X_ = np.column_stack([np.ones(len(X)), X])
    coef, *_ = lstsq(X_, y, rcond=None)
    pred = X_ @ coef
    rho, _ = spearmanr(pred, y)
    return coef, rho

coef1, rho1 = fit_and_corr(X1, y)
coef2, rho2 = fit_and_corr(X2, y)
print(f"\nOLS: STL_PG alone ->            in-sample Spearman={rho1:.4f}  coef={coef1}")
print(f"OLS: STL_PG + DEFLECTIONS ->     in-sample Spearman={rho2:.4f}  coef={coef2}")
