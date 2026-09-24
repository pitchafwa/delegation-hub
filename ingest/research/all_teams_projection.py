"""Pull EVERY team's roster in the league (not just Tommy's) and score each
player with Layer A, so Layer B can compute team-relative context (contend
vs rebuild) for ANY team -- needed generically, not just for Tommy's roster,
per his correction that this has to work for trade/waiver evaluation of
players on other rosters too.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config
from fit_talent_model import unpack_and_score

ROOT = Path(__file__).resolve().parent

crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

current = pd.read_csv(ROOT / "data" / "current_season_for_prediction_v2.csv")
for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
    base_col = col.replace("_PREV", "")
    current[col] = current[col].fillna(current[base_col])


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


current["NORM_NAME"] = current["PLAYER_NAME"].apply(normalize_name)
name_to_row = {r.NORM_NAME: i for i, r in enumerate(current.itertuples())}

params = np.load(ROOT / "data" / "_quick_fit_params.npy")  # PROVISIONAL -- swap for frozen LOSO params

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)

rows = []
unmatched = []
for team in league.teams:
    for p in team.roster:
        nba_id = espnid_to_nbaid.get(p.playerId)
        method = "crosswalk"
        idx = None
        if nba_id is not None and not pd.isna(nba_id):
            match = current[current["PLAYER_ID"] == int(nba_id)]
            if not match.empty:
                idx = match.index[0]
        if idx is None:
            norm = normalize_name(p.name)
            idx = name_to_row.get(norm)
            method = "name_fallback"
        if idx is None:
            unmatched.append((team.team_name, p.name))
            continue
        row = current.loc[[idx]]
        score = unpack_and_score(params, row)[0]
        rows.append({
            "team_id": team.team_id, "team_name": team.team_name, "player": p.name,
            "age": row["AGE"].iloc[0], "current_ppg": row["FANTASY_PPG"].iloc[0],
            "layer_a_value": score, "match_method": method,
        })

df = pd.DataFrame(rows)
df.to_csv(ROOT / "data" / "all_teams_layer_a.csv", index=False)
print(f"Scored {len(df)} rostered players across {df['team_id'].nunique()} teams. Unmatched: {len(unmatched)}")
if unmatched:
    print("Unmatched:", unmatched)

print("\n=== Team roster strength (sum of Layer A value, top-10 by value = starters+key bench) ===")
team_strength = (
    df.sort_values(["team_id", "layer_a_value"], ascending=[True, False])
    .groupby("team_id")
    .apply(lambda d: d.head(10)["layer_a_value"].sum(), include_groups=False)
    .rename("top10_value_sum")
    .reset_index()
)
team_strength = team_strength.merge(df[["team_id", "team_name"]].drop_duplicates(), on="team_id")
team_strength = team_strength.merge(
    df.groupby("team_id")["age"].mean().rename("roster_avg_age").reset_index(), on="team_id"
)
league_mean = team_strength["top10_value_sum"].mean()
league_std = team_strength["top10_value_sum"].std()
team_strength["strength_z"] = (team_strength["top10_value_sum"] - league_mean) / league_std
team_strength = team_strength.sort_values("strength_z", ascending=False)
pd.set_option("display.width", 140)
print(team_strength.to_string(index=False))
team_strength.to_csv(ROOT / "data" / "team_strength.csv", index=False)
