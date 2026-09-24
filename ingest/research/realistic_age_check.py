"""Tommy correctly called out that 'production peaks at 19 and declines from
there' is implausible. Re-check properly: predict on REAL rows at each age
(their actual feature combinations), not a partial-dependence plot that holds
every OTHER feature at the population median -- which for a 19-year-old is an
unrealistic combination (real 19-year-olds who play real minutes don't look
like a median 27-year-old rotation player in every other stat) and produces
an extrapolation artifact, not a real finding.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset.csv")

NUMERIC = [
    "AGE", "EXPERIENCE", "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV",
    "USG_PCT", "USG_PCT_PREV", "TS_PCT", "TS_PCT_PREV", "AST_PCT", "REB_PCT",
    "PACE", "PACE_PREV", "PIE", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "DRAFT_NUMBER_FILLED", "UNDRAFTED", "TEAM_CHANGED_OFFSEASON", "TRADED_MIDSEASON",
]
CATEGORICAL = ["POSITION"]
TARGET = "TARGET_FANTASY_PPG"

train = df[df["SEASON_YEAR"] <= 2021]
preprocess = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
pipe = Pipeline([("prep", preprocess), ("gbt", HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42))])
pipe.fit(train[NUMERIC + CATEGORICAL], train[TARGET])

df["PRED"] = pipe.predict(df[NUMERIC + CATEGORICAL])

print("=== REAL rows only: actual vs predicted TARGET by age (own feature combos) ===")
by_age = df.groupby("AGE").agg(
    n=("PRED", "size"),
    mean_actual_current=("FANTASY_PPG", "mean"),
    mean_actual_target=(TARGET, "mean"),
    mean_predicted=("PRED", "mean"),
).reset_index()
by_age = by_age[by_age["n"] >= 10]
print(by_age.to_string(index=False))

print("\n=== The age=19 rows specifically -- who are they, really? ===")
young = df[df["AGE"] == 19][["PLAYER_NAME", "SEASON", "EXPERIENCE", "GP", "MIN_PG", "USG_PCT", "FANTASY_PPG", TARGET, "PRED"]]
print(young.to_string(index=False))
