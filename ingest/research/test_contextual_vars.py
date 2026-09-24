"""Test candidate contextual variables against the v2.0.0 composite's real
residual error on the 2023-24 holdout: actual minus predicted. If a variable
explains real variance in that residual, it's genuine incremental signal the
current model is missing. If not, it's logged as tested-and-rejected, same
discipline as everything else in this project.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent

compare = pd.read_csv(ROOT / "data" / "kalman_composite_vs_layer_a.csv")
compare["residual"] = compare["actual_ppg"] - compare["kalman_composite"]

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))

# Prediction was made using data through end of 2022-23, predicting 2023-24.
# "Trailing" injury history = the 3 seasons ending 2022-23 (2020, 2021, 2022).
PRED_YEAR = 2022  # season_year whose end-of-season state produced the 2023-24 prediction
TRAILING_YEARS = [2020, 2021, 2022]

# === Candidate 1: injury history (trailing 3-year games-missed rate) ===
trailing = season_base[season_base["SEASON_YEAR"].isin(TRAILING_YEARS)]
# a full season is ~82 games; a lockout-shortened 2020 season was 72 -- normalize per-season.
season_length = {2020: 72, 2021: 72, 2022: 82}
trailing = trailing.copy()
trailing["season_len"] = trailing["SEASON_YEAR"].map(season_length)
trailing["missed_frac"] = 1 - (trailing["GP"] / trailing["season_len"]).clip(upper=1.0)
injury_hist = trailing.groupby("PLAYER_ID")["missed_frac"].mean().rename("injury_history_missed_frac")

compare = compare.merge(injury_hist, on="PLAYER_ID", how="left")
n_have_full_history = compare["injury_history_missed_frac"].notna().sum()
print(f"Players with full 3-year trailing history: {n_have_full_history} of {len(compare)}")

sub = compare.dropna(subset=["injury_history_missed_frac"])
r, p = pearsonr(sub["injury_history_missed_frac"], sub["residual"])
rho, ps = spearmanr(sub["injury_history_missed_frac"], sub["residual"])
print(f"\n=== Injury history (trailing 3yr missed-games fraction) vs residual ===")
print(f"Pearson r={r:.4f} (p={p:.4f})   Spearman rho={rho:.4f} (p={ps:.4f})")
print("Interpretation: negative correlation = higher injury history -> model OVER-predicts "
      "(real outcome falls short) -> real incremental signal worth adding.")

# === Candidate 2: usage vacuum (did a high-usage teammate leave before this season?) ===
adv = pd.read_csv(ROOT / "data" / "player_season_advanced.csv")
adv["SEASON_YEAR"] = adv["SEASON"].apply(lambda s: int(s.split("-")[0]))
base_small = season_base[["PLAYER_ID", "SEASON_YEAR", "TEAM_ABBREVIATION", "GP"]]
usg = adv[["PLAYER_ID", "SEASON_YEAR", "USG_PCT"]].merge(base_small, on=["PLAYER_ID", "SEASON_YEAR"])
usg = usg[usg["GP"] >= 20]

prev_season_teams = usg[usg["SEASON_YEAR"] == PRED_YEAR][["PLAYER_ID", "TEAM_ABBREVIATION", "USG_PCT"]]
next_season_players = set(season_base[season_base["SEASON_YEAR"] == PRED_YEAR + 1]["PLAYER_ID"])

# For each team, find their highest-usage player in PRED_YEAR, check if that player
# is STILL on a team-2023-24 roster with real GP -- if not (or moved teams), it's a vacuum event.
next_season_team = season_base[season_base["SEASON_YEAR"] == PRED_YEAR + 1].set_index("PLAYER_ID")["TEAM_ABBREVIATION"]

vacuum_flag = {}
for team, grp in prev_season_teams.groupby("TEAM_ABBREVIATION"):
    top_usage_player = grp.sort_values("USG_PCT", ascending=False).iloc[0]
    stayed = next_season_team.get(top_usage_player["PLAYER_ID"]) == team
    for pid in grp["PLAYER_ID"]:
        if pid == top_usage_player["PLAYER_ID"]:
            continue
        vacuum_flag[pid] = 0 if stayed else 1

compare["usage_vacuum"] = compare["PLAYER_ID"].map(vacuum_flag).fillna(0)
print(f"\n=== Usage vacuum (team's top-usage player left) vs residual ===")
print(f"Players flagged with a vacuum opportunity: {int(compare['usage_vacuum'].sum())} of {len(compare)}")
vac = compare[compare["usage_vacuum"] == 1]["residual"]
novac = compare[compare["usage_vacuum"] == 0]["residual"]
print(f"Mean residual WITH vacuum: {vac.mean():.2f} (n={len(vac)})")
print(f"Mean residual WITHOUT vacuum: {novac.mean():.2f} (n={len(novac)})")
from scipy.stats import ttest_ind
t, tp = ttest_ind(vac, novac, equal_var=False)
print(f"t-test: t={t:.3f}, p={tp:.4f}")

compare.to_csv(ROOT / "data" / "contextual_vars_test.csv", index=False)
