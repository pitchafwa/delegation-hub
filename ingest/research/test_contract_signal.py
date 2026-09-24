"""Test the contract-status hypothesis with real data (2000-2020 salary
history -- the only free, directly-downloadable source found; misses the
most recent seasons, so this tests on 2010-2019 transitions specifically).

Signal: did a player get a big raise this season (new contract kicking in)?
Real literature found earlier suggests a nuanced, even opposite-of-hypothesis
pattern (a small contract-YEAR boost, then a post-signing dip) -- testing
the real direction here rather than assuming either way.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

ROOT = Path(__file__).resolve().parent

salaries = pd.read_csv(ROOT / "data" / "nba_salaries.csv")
salaries.columns = [c.strip() for c in salaries.columns]
salaries["name"] = salaries["name"].str.strip()


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


salaries["norm_name"] = salaries["name"].apply(normalize_name)
# salaries.csv 'season' = the year the season STARTED + 1 convention check:
# row season=2000 for Shaq et al reflects the 1999-2000 season based on known
# real salaries -- i.e. season_year (our convention) = season - 1.
salaries["SEASON_YEAR"] = salaries["season"] - 1

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_base["norm_name"] = season_base["PLAYER_NAME"].apply(normalize_name)
season_base["FANTASY_PTS"] = (
    season_base["PTS"] + 1.5 * season_base["REB"] + 2 * season_base["AST"] + 3 * season_base["STL"]
    + 3 * season_base["BLK"] + season_base["FG3M"] + 2 * season_base["FTM"] - season_base["FTA"]
    - season_base["TOV"] + 3 * season_base["TD3"]
)
season_base["FANTASY_PPG"] = season_base["FANTASY_PTS"] / season_base["GP"].replace(0, np.nan)

name_to_pid = season_base.drop_duplicates("norm_name").set_index("norm_name")["PLAYER_ID"]
salaries["PLAYER_ID"] = salaries["norm_name"].map(name_to_pid)
matched = salaries.dropna(subset=["PLAYER_ID"]).copy()
matched["PLAYER_ID"] = matched["PLAYER_ID"].astype(int)
print(f"Matched {matched['PLAYER_ID'].nunique()} of {salaries['norm_name'].nunique()} salary-listed players to real PLAYER_IDs")

matched = matched.sort_values(["PLAYER_ID", "SEASON_YEAR"])
matched["prev_salary"] = matched.groupby("PLAYER_ID")["salary"].shift(1)
matched["salary_ratio"] = matched["salary"] / matched["prev_salary"]
matched["big_raise"] = (matched["salary_ratio"] >= 1.5).astype(int)  # new contract kicking in, roughly

raises = matched[matched["big_raise"] == 1][["PLAYER_ID", "SEASON_YEAR", "salary_ratio"]].copy()
print(f"Real 'big raise' seasons identified (salary >= 1.5x prior year): {len(raises)}")

# Test: for players who just got a big raise in season Y, how did their
# fantasy PPG change from season Y (pre-raise-effects-fully-showing, since
# raise often mid-year via extension) to season Y+1, vs. players who didn't?
ppg = season_base.set_index(["PLAYER_ID", "SEASON_YEAR"])["FANTASY_PPG"]
gp = season_base.set_index(["PLAYER_ID", "SEASON_YEAR"])["GP"]

rows = []
for _, r in raises.iterrows():
    pid, y = r["PLAYER_ID"], r["SEASON_YEAR"]
    if (pid, y) not in ppg.index or (pid, y + 1) not in ppg.index:
        continue
    if gp.get((pid, y), 0) < 20 or gp.get((pid, y + 1), 0) < 20:
        continue
    rows.append({"PLAYER_ID": pid, "raise_year": y, "ppg_raise_year": ppg[(pid, y)], "ppg_next_year": ppg[(pid, y + 1)]})
raise_group = pd.DataFrame(rows)
raise_group["pct_change"] = (raise_group["ppg_next_year"] - raise_group["ppg_raise_year"]) / raise_group["ppg_raise_year"]

# Comparison group: everyone else (no big raise that year), same real constraint (GP>=20 both years)
all_transitions = season_base[season_base["GP"] >= 20][["PLAYER_ID", "SEASON_YEAR"]].copy()
raise_keys = set(zip(raises["PLAYER_ID"], raises["SEASON_YEAR"]))
all_transitions["had_raise"] = list(zip(all_transitions["PLAYER_ID"], all_transitions["SEASON_YEAR"]))
all_transitions["had_raise"] = all_transitions["had_raise"].isin(raise_keys)
control_rows = []
for _, r in all_transitions[~all_transitions["had_raise"]].iterrows():
    pid, y = r["PLAYER_ID"], r["SEASON_YEAR"]
    if (pid, y + 1) not in ppg.index or gp.get((pid, y + 1), 0) < 20:
        continue
    control_rows.append({"PLAYER_ID": pid, "year": y, "ppg_year": ppg[(pid, y)], "ppg_next_year": ppg[(pid, y + 1)]})
control_group = pd.DataFrame(control_rows)
control_group["pct_change"] = (control_group["ppg_next_year"] - control_group["ppg_year"]) / control_group["ppg_year"]

print(f"\n=== Big-raise group (n={len(raise_group)}): mean next-year PPG %% change = {raise_group['pct_change'].mean():+.2%} (median {raise_group['pct_change'].median():+.2%}) ===")
print(f"=== Control group, no raise (n={len(control_group)}): mean next-year PPG %% change = {control_group['pct_change'].mean():+.2%} (median {control_group['pct_change'].median():+.2%}) ===")

t, p = ttest_ind(raise_group["pct_change"].dropna(), control_group["pct_change"].dropna(), equal_var=False)
print(f"\nt-test: t={t:.3f}, p={p:.4f}")

raise_group.to_csv(ROOT / "data" / "contract_raise_test.csv", index=False)
