"""Apply the FROZEN Layer A model to every real rostered player in the
league (not just Tommy's team, since it has to work generically), so he can
inspect the raw output before Layer B gets built on top of it.
"""
import json
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

with open(ROOT / "data" / "layer_a_model.frozen.json") as f:
    artifact = json.load(f)
params = np.array(artifact["params_raw"])

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
name_to_idx = {r.NORM_NAME: i for i, r in enumerate(current.itertuples())}

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)

rows = []
for team in league.teams:
    for p in team.roster:
        nba_id = espnid_to_nbaid.get(p.playerId)
        idx = None
        if nba_id is not None and not pd.isna(nba_id):
            match = current[current["PLAYER_ID"] == int(nba_id)]
            if not match.empty:
                idx = match.index[0]
        if idx is None:
            idx = name_to_idx.get(normalize_name(p.name))
        if idx is None:
            rows.append({"team_id": team.team_id, "team_name": team.team_name, "player": p.name,
                         "age": None, "current_ppg": None, "layer_a_value": None,
                         "note": "NO MATCH -- likely too few games this season (injury) to score"})
            continue
        row = current.loc[[idx]]
        score = float(unpack_and_score(params, row)[0])
        rows.append({
            "team_id": team.team_id, "team_name": team.team_name, "player": p.name,
            "age": float(row["AGE"].iloc[0]), "current_ppg": float(row["FANTASY_PPG"].iloc[0]),
            "layer_a_value": round(score, 1), "note": "",
        })

df = pd.DataFrame(rows)
df["percentile"] = df["layer_a_value"].rank(pct=True) * 100
df.to_csv(ROOT / "data" / "layer_a_all_players.csv", index=False)

pd.set_option("display.width", 140)
print("=== Tommy's roster (team_id 12), Layer A output, sorted by value ===\n")
mine = df[df["team_id"] == 12].sort_values("layer_a_value", ascending=False, na_position="last")
print(mine[["player", "age", "current_ppg", "layer_a_value", "percentile", "note"]].to_string(index=False))

print("\n=== Top 15 players league-wide, for sanity-check scale ===\n")
top = df.dropna(subset=["layer_a_value"]).sort_values("layer_a_value", ascending=False).head(15)
print(top[["player", "team_name", "age", "current_ppg", "layer_a_value"]].to_string(index=False))

print("\n=== Bottom 10 rostered players league-wide, for sanity-check scale ===\n")
bottom = df.dropna(subset=["layer_a_value"]).sort_values("layer_a_value").head(10)
print(bottom[["player", "team_name", "age", "current_ppg", "layer_a_value"]].to_string(index=False))

print(f"\nUnmatched (no score): {df['layer_a_value'].isna().sum()} of {len(df)} rostered players")
print(df[df['layer_a_value'].isna()][["player", "team_name"]].to_string(index=False))
