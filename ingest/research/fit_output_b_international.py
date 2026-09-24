"""Output B, INTERNATIONAL prospects: a SEPARATE model from the college one
(fit_output_b.py / output_b_model.pkl), deliberately kept untouched. College
prospects have ~28 rich features (porpag, BPM splits, rebounding/foul rates,
shooting splits, combine testing); almost none of that exists for
international prospects (checked: 0/121 non-null for all of it except
talent_pctile, draft_age, three_pct, pick). Forcing international prospects
through the college feature set means real signal gets replaced with
league-median filler for nearly every column -- a separate, honestly-scoped
model using what's ACTUALLY available for international prospects should do
better than that.

Real per-game box score stats exist in international_features.csv (one
pre-draft season per player, confirmed against known real stat lines for
Wembanyama/Doncic) but were previously collapsed down to a single
percentile (talent_pctile) for the unified Output A pipeline. This script
converts them to real per-minute rates instead, giving international
prospects a comparable production profile to college's per-minute features.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

intl_raw = pd.read_csv(ROOT / "data" / "international_features.csv")
unified = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
intl_unified = unified[unified["data_source"] == "international"][
    ["PLAYER_ID", "player", "real_draft_year", "real_draft_number", "draft_age", "talent_pctile"]
].drop_duplicates(subset=["PLAYER_ID"])

# international_features.csv's PERSON_ID is the real NBA player id (verified
# against unified's PLAYER_ID for Wembanyama/Doncic above)
intl_raw = intl_raw.rename(columns={"PERSON_ID": "PLAYER_ID"})
df = intl_unified.merge(
    intl_raw[["PLAYER_ID", "G", "MP", "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PTS", "game_score"]],
    on="PLAYER_ID", how="left",
)
print(f"International prospects: {len(df)}, with raw box-score row: {df['MP'].notna().sum()}", flush=True)

for stat, col in [("PTS", "PTS"), ("REB", "TRB"), ("AST", "AST"), ("STL", "STL"), ("BLK", "BLK"), ("TOV", "TOV")]:
    df[f"int_{stat}_per_min"] = df[col] / df["MP"].replace(0, np.nan)
df["int_fg_pct"] = df["FG%"]
df["int_three_pct"] = df["3P%"]
df["int_ft_pct"] = df["FT%"]
df["int_game_score_per_min"] = df["game_score"] / df["MP"].replace(0, np.nan)
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())

INTL_FEATURES = [
    "int_PTS_per_min", "int_REB_per_min", "int_AST_per_min", "int_STL_per_min",
    "int_BLK_per_min", "int_TOV_per_min", "int_fg_pct", "int_three_pct", "int_ft_pct",
    "int_game_score_per_min", "draft_age_filled", "pick_filled",
]

# --- real rookie-season NBA target, same construction as the college side ---
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
rookie_rows = season_base.merge(df[["PLAYER_ID", "real_draft_year"]], on="PLAYER_ID", how="inner")
rookie_rows = rookie_rows[rookie_rows["SEASON_YEAR"] == rookie_rows["real_draft_year"]]

TARGET_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "STL"]
for stat in TARGET_STATS:
    rookie_rows[f"rookie_{stat}_per_min"] = rookie_rows[stat] / rookie_rows["MIN"].replace(0, np.nan)
rookie_rows["rookie_MIN_per_game"] = rookie_rows["MIN"] / rookie_rows["GP"].replace(0, np.nan)
target_cols = ["PLAYER_ID"] + [f"rookie_{s}_per_min" for s in TARGET_STATS] + ["rookie_MIN_per_game"]
df = df.merge(rookie_rows[target_cols], on="PLAYER_ID", how="left")

trainable = df.dropna(subset=["rookie_MIN_per_game"] + INTL_FEATURES[:9]).copy()
print(f"Trainable international rookies (real box-score input + real rookie-season target): {len(trainable)}")
print(f"Draft years covered: {sorted(trainable['real_draft_year'].astype(int).unique().tolist())}\n", flush=True)

if len(trainable) < 20:
    print("Sample too small for a real train/test split -- reporting what we have "
          "and stopping short of a false-confidence model fit.", flush=True)
    print(trainable[["player", "real_draft_year"] + INTL_FEATURES + ["rookie_MIN_per_game"]].to_string())
    sys.exit(0)

TARGET_COLS = {s: f"rookie_{s}_per_min" for s in TARGET_STATS}
TARGET_COLS["MIN"] = "rookie_MIN_per_game"

X = trainable[INTL_FEATURES].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median()).to_numpy(dtype=float)
years = trainable["real_draft_year"].to_numpy()
n = len(trainable)


def r2(pred, actual):
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-9 else None


def mae(pred, actual):
    return np.mean(np.abs(actual - pred))


from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# small-sample reality: use leave-one-OUT at the PLAYER level (not by draft
# class -- too few international players per class to make LOCO meaningful
# here) since every fold still needs a real, ungamed held-out test point.
loo = LeaveOneOut()

print("=== INTERNATIONAL Output B (leave-one-player-out, n={}) ===\n".format(n), flush=True)
RESULTS = {}
for stat, target_col in TARGET_COLS.items():
    y = trainable[target_col].to_numpy(dtype=float)
    baseline_pred = np.full(n, np.nan)
    ridge_pred = np.full(n, np.nan)
    for train_idx, test_idx in loo.split(X):
        baseline_pred[test_idx] = y[train_idx].mean()
        scaler = StandardScaler().fit(X[train_idx])
        model = Ridge(alpha=10.0)
        model.fit(scaler.transform(X[train_idx]), y[train_idx])
        ridge_pred[test_idx] = model.predict(scaler.transform(X[test_idx]))
    base_r2, base_mae = r2(baseline_pred, y), mae(baseline_pred, y)
    ridge_r2, ridge_mae = r2(ridge_pred, y), mae(ridge_pred, y)
    beat = "BEATS baseline" if ridge_r2 > base_r2 else "below baseline"
    print(f"{stat:5s}  baseline R2={base_r2:.3f} MAE={base_mae:.4f}   ridge R2={ridge_r2:.3f} MAE={ridge_mae:.4f}  {beat}", flush=True)
    RESULTS[stat] = {"baseline_r2": base_r2, "ridge_r2": ridge_r2}

print("\n(Sample size is small -- treat these as directional, not final, and "
      "revisit as more international rookies accumulate real NBA seasons.)")
