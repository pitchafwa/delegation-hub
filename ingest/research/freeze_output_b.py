"""Freeze Output B: fit each stat's winning model (per fit_output_b.py's
real LOCO-CV comparison) on ALL 704 trainable rookies, save the fitted
models + imputation medians + feature list to a single artifact so any
prospect (including a brand-new draftee not yet in this dataset) can get a
projected rookie-season per-stat baseline from their pre-NBA profile alone.

Winning model per stat (from fit_output_b_log.txt, real LOCO-CV R2):
  PTS ridge(0.156)  REB ridge(0.447)  AST ridge(0.519)  BLK ridge(0.281)
  TOV rf(0.134)     FG3M ridge(0.378) FTM rf(0.111)     FTA rf(0.138)
  STL ridge(0.003, essentially no signal -- kept for a non-zero prior,
       better than the flat 0.0 kalman_vor.py currently falls back to, but
       don't expect much from it)
  MIN rf(0.367)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
rookie_rows = season_base.merge(
    df[["PLAYER_ID", "real_draft_year"]].drop_duplicates(), on="PLAYER_ID", how="inner"
)
rookie_rows = rookie_rows[rookie_rows["SEASON_YEAR"] == rookie_rows["real_draft_year"]]
rookie_rows["rookie_STL_per_min"] = rookie_rows["STL"] / rookie_rows["MIN"].replace(0, np.nan)
df = df.merge(rookie_rows[["PLAYER_ID", "rookie_STL_per_min"]], on="PLAYER_ID", how="left")

STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "STL"]
TARGET_COLS = {s: f"rookie_{s}_per_min" for s in STATS}
TARGET_COLS["MIN"] = "rookie_MIN_per_game"

WINNING_MODEL = {
    "PTS": "ridge", "REB": "ridge", "AST": "ridge", "BLK": "ridge", "TOV": "rf",
    "FG3M": "ridge", "FTM": "rf", "FTA": "rf", "STL": "ridge", "MIN": "rf",
}
# ridge alpha per stat -- PTS re-tuned in explore_pts_models.py (0.156 -> 0.164 R2
# at alpha=100; other stats kept at the originally-validated alpha=10)
RIDGE_ALPHA = {"PTS": 100.0}

df = df[df["real_draft_year"] >= 2008].copy()
trainable = df.dropna(subset=["rookie_MIN_per_game"]).copy()

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
trainable["exp_numeric"] = trainable["exp"].map(EXP_MAP)
trainable["exp_numeric"] = trainable["exp_numeric"].fillna(trainable["exp_numeric"].median())
trainable["rec_filled"] = trainable["rec"].fillna(0)
trainable["draft_age_filled"] = trainable["draft_age"].fillna(trainable["draft_age"].median())
trainable["pick_filled"] = trainable["real_draft_number"].fillna(61.0)

FEATURES = [
    "talent_pctile", "rec_filled", "exp_numeric", "draft_age_filled", "breakout_age_filled",
    "porpag", "usg", "ts", "ortg", "obpm", "dbpm", "bpm", "stops",
    "oreb_rate", "dreb_rate", "ast_to", "ftr", "pfr",
    "WINGSPAN_PCTILE", "STANDING_REACH_PCTILE", "STANDING_VERTICAL_LEAP_PCTILE",
    "MAX_VERTICAL_LEAP_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE",
    "three_pct", "rim_pct", "mid_pct", "pick_filled",
]
FEATURES = [f for f in FEATURES if f in trainable.columns]
X_raw = trainable[FEATURES].apply(pd.to_numeric, errors="coerce")
medians = X_raw.median()
X_imputed = X_raw.fillna(medians)
Xi = X_imputed.to_numpy(dtype=float)

models = {}
for stat, target_col in TARGET_COLS.items():
    y = trainable[target_col].to_numpy(dtype=float)
    kind = WINNING_MODEL[stat]
    if kind == "ridge":
        scaler = StandardScaler().fit(Xi)
        model = Ridge(alpha=RIDGE_ALPHA.get(stat, 10.0), random_state=42)
        model.fit(scaler.transform(Xi), y)
        models[stat] = {"kind": "ridge", "model": model, "scaler": scaler}
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1)
        model.fit(Xi, y)
        models[stat] = {"kind": "rf", "model": model, "scaler": None}
    print(f"Fit {stat} ({kind}) on {len(y)} rookies.", flush=True)

artifact = {
    "features": FEATURES,
    "medians": medians.to_dict(),
    "target_cols": TARGET_COLS,
    "winning_model": WINNING_MODEL,
    "real_loco_cv_r2": {  # from fit_output_b_log.txt -- honest expectation-setting, not training-set fit quality
        "PTS": 0.156, "REB": 0.447, "AST": 0.519, "BLK": 0.281, "TOV": 0.134,
        "FG3M": 0.378, "FTM": 0.111, "FTA": 0.138, "STL": 0.003, "MIN": 0.367,
    },
    "n_training_rows": len(trainable),
    "training_years": [int(trainable["real_draft_year"].min()), int(trainable["real_draft_year"].max())],
    "known_gaps": [
        "International prospects have ZERO Output B training coverage (0 of 121) -- "
        "the feature values will still compute for an international prospect, but the "
        "models never saw a real international rookie-season outcome during training.",
        "STL has essentially no predictive signal (R2=0.003) -- kept only because a weak, "
        "individualized prior beats the flat 0.0 kalman_vor.py currently falls back to for "
        "a true zero-game rookie, not because it's a good projection.",
    ],
}
with open(ROOT / "data" / "output_b_model.pkl", "wb") as f:
    pickle.dump({**artifact, "models": models}, f)

# also save a human-readable metadata copy (not the pickled model objects)
with open(ROOT / "data" / "output_b_model.meta.json", "w") as f:
    json.dump(artifact, f, indent=2)

print(f"\nSaved output_b_model.pkl ({len(models)} stats) and output_b_model.meta.json")


def project_rookie(feature_row: dict) -> dict:
    """feature_row: dict of {feature_name: value} for one prospect (missing
    keys are fine, filled with training medians). Returns projected
    rookie-season per-stat rates (per-minute for the 9 counting stats,
    per-game for MIN)."""
    x = np.array([[feature_row.get(f, medians[f]) for f in FEATURES]], dtype=float)
    x = np.where(np.isnan(x), [medians[f] for f in FEATURES], x)
    out = {}
    for stat, m in models.items():
        if m["kind"] == "ridge":
            out[stat] = float(m["model"].predict(m["scaler"].transform(x))[0])
        else:
            out[stat] = float(m["model"].predict(x)[0])
    return out


if __name__ == "__main__":
    # sanity check: project a few real 2025 rookies from their own real profile row
    for name in ["Cooper Flagg", "Zaccharie Risacher", "Alex Sarr"]:
        rows = trainable[trainable["player"].str.contains(name.split()[-1], case=False, na=False)]
        if rows.empty:
            print(f"{name}: not found in trainable set (may be international / no match)")
            continue
        row = rows.iloc[0]
        feats = {f: row[f] for f in FEATURES}
        proj = project_rookie(feats)
        actual = {s: row[TARGET_COLS[s]] for s in TARGET_COLS}
        print(f"\n{name} (draft {int(row['real_draft_year'])}, pick {row['pick_filled']:.0f}):")
        for s in TARGET_COLS:
            print(f"  {s:5s} projected={proj[s]:.4f}  actual={actual[s]:.4f}")
