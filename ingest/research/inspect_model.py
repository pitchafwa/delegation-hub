from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset.csv")

TRAIN_MAX_YEAR = 2021
TEST_YEARS = [2022, 2023, 2024]
train = df[df["SEASON_YEAR"] <= TRAIN_MAX_YEAR].copy()
test = df[df["SEASON_YEAR"].isin(TEST_YEARS)].copy()

NUMERIC = [
    "AGE", "EXPERIENCE", "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV",
    "USG_PCT", "USG_PCT_PREV", "TS_PCT", "TS_PCT_PREV", "AST_PCT", "REB_PCT",
    "PACE", "PACE_PREV", "PIE", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "DRAFT_NUMBER_FILLED", "UNDRAFTED", "TEAM_CHANGED_OFFSEASON", "TRADED_MIDSEASON",
]
CATEGORICAL = ["POSITION"]
TARGET = "TARGET_FANTASY_PPG"

preprocess = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
pipe = Pipeline([("prep", preprocess), ("gbt", HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42))])
pipe.fit(train[NUMERIC + CATEGORICAL], train[TARGET])

perm = permutation_importance(pipe, test[NUMERIC + CATEGORICAL], test[TARGET], n_repeats=8, random_state=42, scoring="r2")
order = np.argsort(perm.importances_mean)[::-1]
cols = NUMERIC + CATEGORICAL
print("=== Permutation importance (drop in R^2 when shuffled) ===")
for i in order[:15]:
    print(f"{cols[i]:24s} {perm.importances_mean[i]:.4f}  (+/- {perm.importances_std[i]:.4f})")

print("\n=== Age partial-dependence sanity check (holding others at median) ===")
median_row = train[NUMERIC].median()
ages = range(19, 40)
rows = []
for age in ages:
    row = median_row.copy()
    row["AGE"] = age
    rows.append(row)
pdf = pd.DataFrame(rows)
pdf["POSITION"] = train["POSITION"].mode()[0]
preds = pipe.predict(pdf[NUMERIC + CATEGORICAL])
for age, pred in zip(ages, preds):
    print(f"age {age}: predicted next-season fantasy PPG (median-context player) = {pred:.1f}")
