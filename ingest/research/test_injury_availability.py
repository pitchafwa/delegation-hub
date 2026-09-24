"""Redo the injury-history test the right way: include players who had a
real washout season next year (missed most/all of it) as a real, low
outcome -- not filtered out. The first pass conditioned on "played 20+ games
next season," which structurally excludes exactly the downside case injury
history is supposed to predict.

Tests injury history against TWO real outcomes:
1. Did this player have a washout season next year (GP < 40)?
2. Total season fantasy value next year (0 if they didn't play a real season),
   vs. what the model would project ASSUMING normal availability.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr

ROOT = Path(__file__).resolve().parent

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_base["FANTASY_PTS"] = (
    season_base["PTS"] + 1.5 * season_base["REB"] + 2 * season_base["AST"] + 3 * season_base["STL"]
    + 3 * season_base["BLK"] + season_base["FG3M"] + 2 * season_base["FTM"] - season_base["FTA"]
    - season_base["TOV"] + 3 * season_base["TD3"]
)
season_length = {y: (72 if y in (2020, 2021) else 82) for y in range(2010, 2026)}

rows = []
for pred_year in range(2013, 2025):
    trailing_years = [pred_year - 2, pred_year - 1, pred_year]
    trailing = season_base[season_base["SEASON_YEAR"].isin(trailing_years)].copy()
    trailing["season_len"] = trailing["SEASON_YEAR"].map(season_length)
    trailing["missed_frac"] = 1 - (trailing["GP"] / trailing["season_len"]).clip(upper=1.0)
    inj = trailing.groupby("PLAYER_ID")["missed_frac"].mean().rename("injury_history").reset_index()

    # Base pool: real rotation players as of pred_year (GP>=20, meaningful minutes) --
    # NOT filtered by what happens next season.
    base_pool = season_base[(season_base["SEASON_YEAR"] == pred_year) & (season_base["GP"] >= 20)]
    base_pool = base_pool[["PLAYER_ID"]].merge(inj, on="PLAYER_ID", how="left")
    base_pool["pred_source_year"] = pred_year

    next_season = season_base[season_base["SEASON_YEAR"] == pred_year + 1].set_index("PLAYER_ID")
    base_pool["next_gp"] = base_pool["PLAYER_ID"].map(next_season["GP"]).fillna(0)
    base_pool["next_fantasy_total"] = base_pool["PLAYER_ID"].map(next_season["FANTASY_PTS"]).fillna(0.0)
    base_pool["washout_season"] = (base_pool["next_gp"] < 40).astype(int)
    rows.append(base_pool)

pooled = pd.concat(rows, ignore_index=True).dropna(subset=["injury_history"])
print(f"Pooled real rotation-player-seasons: {len(pooled)} across {pooled['pred_source_year'].nunique()} transitions\n")

print("=== Does injury history predict a real washout season next year (GP<40)? ===")
overall_rate = pooled["washout_season"].mean()
print(f"Overall washout rate: {overall_rate:.1%}")
r, p = pointbiserialr(pooled["washout_season"], pooled["injury_history"])
print(f"Point-biserial correlation: r={r:.4f} (p={p:.2e})")

pooled["injury_quartile"] = pd.qcut(pooled["injury_history"], 4, labels=["Q1 healthiest", "Q2", "Q3", "Q4 most missed"], duplicates="drop")
print("\nWashout rate by injury-history quartile:")
print(pooled.groupby("injury_quartile", observed=True)["washout_season"].agg(["mean", "count"]).to_string())

print("\n=== Total next-season fantasy value (including real zeros for washouts) by quartile ===")
print(pooled.groupby("injury_quartile", observed=True)["next_fantasy_total"].agg(["mean", "median", "count"]).to_string())

pooled.to_csv(ROOT / "data" / "injury_availability_test.csv", index=False)
