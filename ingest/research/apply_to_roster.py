"""Apply the trained valuation model to Tommy's REAL current roster to produce
projected 2026-27 fantasy value for each player -- the actual keeper decision input.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset.csv")
current = pd.read_csv(ROOT / "data" / "current_season_for_prediction.csv")

NUMERIC = [
    "AGE", "EXPERIENCE", "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV",
    "USG_PCT", "USG_PCT_PREV", "TS_PCT", "TS_PCT_PREV", "AST_PCT", "REB_PCT",
    "PACE", "PACE_PREV", "PIE", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "DRAFT_NUMBER_FILLED", "UNDRAFTED", "TEAM_CHANGED_OFFSEASON", "TRADED_MIDSEASON",
]
CATEGORICAL = ["POSITION"]
TARGET = "TARGET_FANTASY_PPG"

# Final production model: train on ALL real historical transitions (2010-2024),
# now that the approach has already been backtested honestly out-of-sample.
preprocess = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
pipe = Pipeline([("prep", preprocess), ("gbt", HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42))])
pipe.fit(df[NUMERIC + CATEGORICAL], df[TARGET])

current["PRED_FANTASY_PPG_2026_27"] = pipe.predict(current[NUMERIC + CATEGORICAL])


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


current["NORM_NAME"] = current["PLAYER_NAME"].apply(normalize_name)

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)
my_team = next(t for t in league.teams if t.team_id == 12)

print(f"=== {my_team.team_name} -- projected 2026-27 fantasy PPG per player ===\n")
rows = []
for p in my_team.roster:
    norm = normalize_name(p.name)
    match = current[current["NORM_NAME"] == norm]
    if match.empty:
        rows.append((p.name, p.position, None, "NO MATCH in NBA data"))
        continue
    if len(match) > 1:
        match = match.iloc[[0]]
    pred = match["PRED_FANTASY_PPG_2026_27"].iloc[0]
    cur_ppg = match["FANTASY_PPG"].iloc[0]
    age = match["AGE"].iloc[0]
    rows.append((p.name, p.position, pred, f"age {age:.0f}, this season {cur_ppg:.1f} PPG"))

result = pd.DataFrame(rows, columns=["Player", "ESPN Pos", "Projected 2026-27 PPG", "Context"])
result = result.sort_values("Projected 2026-27 PPG", ascending=False, na_position="last")
pd.set_option("display.width", 140)
print(result.to_string(index=False))
result.to_csv(ROOT / "data" / "tommy_roster_projections.csv", index=False)
