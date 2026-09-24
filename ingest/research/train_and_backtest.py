"""Train candidate valuation models and backtest them on real, held-out
future seasons the model never saw during training (2023, 2024, 2025
transitions) -- not just a random train/test split, which would leak
future information via shared players/eras.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "valuation_dataset.csv")

TRAIN_MAX_YEAR = 2021   # transitions with SEASON_YEAR <= 2021 (targets through 2022)
TEST_YEARS = [2022, 2023, 2024]  # targets 2023, 2024, 2025 -- real backtest, unseen

train = df[df["SEASON_YEAR"] <= TRAIN_MAX_YEAR].copy()
test = df[df["SEASON_YEAR"].isin(TEST_YEARS)].copy()
print(f"Train rows: {len(train)} (seasons through {TRAIN_MAX_YEAR})")
print(f"Test rows: {len(test)} (seasons {TEST_YEARS}, genuinely held out)\n")

NUMERIC = [
    "AGE", "EXPERIENCE", "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV",
    "USG_PCT", "USG_PCT_PREV", "TS_PCT", "TS_PCT_PREV", "AST_PCT", "REB_PCT",
    "PACE", "PACE_PREV", "PIE", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "DRAFT_NUMBER_FILLED", "UNDRAFTED", "TEAM_CHANGED_OFFSEASON", "TRADED_MIDSEASON",
]
CATEGORICAL = ["POSITION"]
TARGET = "TARGET_FANTASY_PPG"


def metrics(y_true, y_pred, label):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"{label:32s} MAE={mae:6.2f}  RMSE={rmse:6.2f}  Pearson r={r:.3f}  Spearman rho={rho:.3f}  R2={r2:.3f}")
    return dict(label=label, mae=mae, rmse=rmse, pearson=r, spearman=rho, r2=r2)


results = []

# --- Baseline: naive persistence (this season's rate IS next season's guess) ---
pred_baseline = test["FANTASY_PPG"].values
results.append(metrics(test[TARGET].values, pred_baseline, "Baseline (persistence)"))

# --- Baseline 2: blend current + prior season (simple 2-pt moving average) ---
blend = test["FANTASY_PPG"].values * 0.65 + test["FANTASY_PPG_PREV"].fillna(test["FANTASY_PPG"]).values * 0.35
results.append(metrics(test[TARGET].values, blend, "Baseline (65/35 blend, no aging adj)"))

# --- Model A: Ridge regression with explicit age term (aging-curve style) ---
train_a = train.copy()
test_a = test.copy()
for frame in (train_a, test_a):
    frame["AGE_SQ"] = frame["AGE"] ** 2
    for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV", "PACE_PREV"]:
        frame[col] = frame[col].fillna(frame[col.replace("_PREV", "")])

ridge_features = [
    "AGE", "AGE_SQ", "EXPERIENCE", "GP", "MIN_PG", "FANTASY_PPG", "FANTASY_PPG_PREV",
    "USG_PCT", "USG_PCT_PREV", "TS_PCT", "TS_PCT_PREV", "TEAM_CHANGED_OFFSEASON",
    "TRADED_MIDSEASON",
]
ridge_pipe = Pipeline([("ridge", Ridge(alpha=5.0))])
Xtr = train_a[ridge_features].fillna(train_a[ridge_features].median())
Xte = test_a[ridge_features].fillna(train_a[ridge_features].median())
ridge_pipe.fit(Xtr, train_a[TARGET])
pred_ridge = ridge_pipe.predict(Xte)
results.append(metrics(test_a[TARGET].values, pred_ridge, "Model A: Ridge + explicit age curve"))

# --- Model B: Gradient boosted trees, full feature set, native NaN handling ---
gbt_features = NUMERIC
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
], remainder="passthrough")
gbt_pipe = Pipeline([
    ("prep", preprocess),
    ("gbt", HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)),
])
Xtr_full = train[gbt_features + CATEGORICAL]
Xte_full = test[gbt_features + CATEGORICAL]
gbt_pipe.fit(Xtr_full, train[TARGET])
pred_gbt = gbt_pipe.predict(Xte_full)
results.append(metrics(test[TARGET].values, pred_gbt, "Model B: Gradient boosted trees"))

print("\n=== Per-year backtest breakdown (Model B) ===")
test_b = test.copy()
test_b["PRED"] = pred_gbt
for yr in TEST_YEARS:
    sub = test_b[test_b["SEASON_YEAR"] == yr]
    metrics(sub[TARGET].values, sub["PRED"].values, f"  target season {yr + 1}")

results_df = pd.DataFrame(results)
results_df.to_csv(ROOT / "data" / "backtest_results.csv", index=False)

# Save the fitted GBT pipeline choice info + feature importances via permutation-free
# approach (HGB doesn't expose feature_importances_ directly the way RF does,
# so we report split-based importance is unavailable; use partial dependence
# sign checks manually below instead for interpretability).
print("\nSaved backtest_results.csv")
