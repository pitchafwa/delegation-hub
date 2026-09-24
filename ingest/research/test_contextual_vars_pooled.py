"""Retest candidate contextual variables against the POOLED multi-season
residuals (2292 real predictions across 6 held-out seasons) -- a much more
robust test than a single season.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

ROOT = Path(__file__).resolve().parent
pooled = pd.read_csv(ROOT / "data" / "pooled_residuals.csv")

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_length = {2018: 82, 2019: 82, 2020: 72, 2021: 72, 2022: 82, 2023: 82}

# === Injury history: trailing 3-year games-missed fraction, computed
# relative to EACH prediction's own source year (not a single fixed year). ===
rows = []
for pred_year in pooled["pred_source_year"].unique():
    trailing_years = [pred_year - 2, pred_year - 1, pred_year]
    trailing = season_base[season_base["SEASON_YEAR"].isin(trailing_years)].copy()
    trailing["season_len"] = trailing["SEASON_YEAR"].map(lambda y: season_length.get(y, 82))
    trailing["missed_frac"] = 1 - (trailing["GP"] / trailing["season_len"]).clip(upper=1.0)
    inj = trailing.groupby("PLAYER_ID")["missed_frac"].mean().rename("injury_history_missed_frac").reset_index()
    inj["pred_source_year"] = pred_year
    rows.append(inj)
injury_all = pd.concat(rows, ignore_index=True)

pooled_inj = pooled.merge(injury_all, on=["PLAYER_ID", "pred_source_year"], how="left").dropna(subset=["injury_history_missed_frac"])
print(f"Pooled rows with injury history available: {len(pooled_inj)} of {len(pooled)}")

r, p = pearsonr(pooled_inj["injury_history_missed_frac"], pooled_inj["residual"])
rho, ps = spearmanr(pooled_inj["injury_history_missed_frac"], pooled_inj["residual"])
print(f"\n=== POOLED: injury history vs residual ===")
print(f"Pearson r={r:.4f} (p={p:.5f})   Spearman rho={rho:.4f} (p={ps:.5f})")

# Quartile breakdown for an intuitive read.
pooled_inj["injury_quartile"] = pd.qcut(pooled_inj["injury_history_missed_frac"], 4, labels=["Q1 (healthiest)", "Q2", "Q3", "Q4 (most missed)"], duplicates="drop")
print("\nMean residual by injury-history quartile:")
print(pooled_inj.groupby("injury_quartile", observed=True)["residual"].agg(["mean", "count"]).to_string())

# === Usage vacuum, redone across all 6 transitions ===
adv = pd.read_csv(ROOT / "data" / "player_season_advanced.csv")
adv["SEASON_YEAR"] = adv["SEASON"].apply(lambda s: int(s.split("-")[0]))
base_small = season_base[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION", "GP"]]
usg = adv[["PLAYER_ID", "SEASON_YEAR", "USG_PCT"]].merge(base_small, on=["PLAYER_ID", "SEASON_YEAR"])
usg = usg[usg["GP"] >= 20]

vacuum_rows = []
for pred_year in pooled["pred_source_year"].unique():
    prev_teams = usg[usg["SEASON_YEAR"] == pred_year][["PLAYER_ID", "TEAM_ABBREVIATION", "USG_PCT"]]
    next_team = season_base[season_base["SEASON_YEAR"] == pred_year + 1].set_index("PLAYER_ID")["TEAM_ABBREVIATION"]
    for team, grp in prev_teams.groupby("TEAM_ABBREVIATION"):
        if len(grp) < 2:
            continue
        top = grp.sort_values("USG_PCT", ascending=False).iloc[0]
        stayed = next_team.get(top["PLAYER_ID"]) == team
        for pid in grp["PLAYER_ID"]:
            if pid == top["PLAYER_ID"]:
                continue
            vacuum_rows.append({"PLAYER_ID": pid, "pred_source_year": pred_year, "usage_vacuum": 0 if stayed else 1})
vacuum_df = pd.DataFrame(vacuum_rows)

pooled_vac = pooled.merge(vacuum_df, on=["PLAYER_ID", "pred_source_year"], how="left")
pooled_vac["usage_vacuum"] = pooled_vac["usage_vacuum"].fillna(0)
print(f"\n=== POOLED: usage vacuum vs residual ===")
print(f"Flagged: {int(pooled_vac['usage_vacuum'].sum())} of {len(pooled_vac)}")
vac = pooled_vac[pooled_vac["usage_vacuum"] == 1]["residual"]
novac = pooled_vac[pooled_vac["usage_vacuum"] == 0]["residual"]
t, tp = ttest_ind(vac, novac, equal_var=False)
print(f"Mean residual WITH vacuum: {vac.mean():.2f} (n={len(vac)})")
print(f"Mean residual WITHOUT vacuum: {novac.mean():.2f} (n={len(novac)})")
print(f"t={t:.3f}, p={tp:.4f}")

pooled_inj.to_csv(ROOT / "data" / "pooled_injury_test.csv", index=False)
pooled_vac.to_csv(ROOT / "data" / "pooled_vacuum_test.csv", index=False)
