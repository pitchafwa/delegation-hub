"""Freeze the international Output B model as its own separate artifact
(output_b_model_international.pkl) -- kept fully independent from the
college model (output_b_model.pkl), which is untouched. Same ridge
approach validated in fit_output_b_international.py (beat baseline on 9/10
stats via leave-one-player-out on n=52).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent

intl_raw = pd.read_csv(ROOT / "data" / "international_features.csv")
unified = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
intl_unified = unified[unified["data_source"] == "international"][
    ["PLAYER_ID", "player", "real_draft_year", "real_draft_number", "draft_age", "talent_pctile"]
].drop_duplicates(subset=["PLAYER_ID"])

intl_raw = intl_raw.rename(columns={"PERSON_ID": "PLAYER_ID"})
df = intl_unified.merge(
    intl_raw[["PLAYER_ID", "G", "MP", "FG%", "3P%", "FT%", "TRB", "AST", "STL", "BLK", "TOV", "PTS", "game_score"]],
    on="PLAYER_ID", how="left",
)
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
print(f"Fitting international Output B on {len(trainable)} rookies.", flush=True)

TARGET_COLS = {s: f"rookie_{s}_per_min" for s in TARGET_STATS}
TARGET_COLS["MIN"] = "rookie_MIN_per_game"

X = trainable[INTL_FEATURES].apply(pd.to_numeric, errors="coerce")
medians = X.median()
Xi = X.fillna(medians).to_numpy(dtype=float)

REAL_LOO_R2 = {  # from fit_output_b_intl_log.txt
    "PTS": 0.131, "REB": 0.440, "AST": 0.643, "BLK": 0.574, "TOV": 0.233,
    "FG3M": 0.147, "FTM": 0.134, "FTA": 0.313, "STL": -0.022, "MIN": 0.238,
}

models = {}
for stat, target_col in TARGET_COLS.items():
    y = trainable[target_col].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xi)
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(scaler.transform(Xi), y)
    models[stat] = {"model": model, "scaler": scaler}
    print(f"Fit {stat} (ridge) on {len(y)} international rookies.", flush=True)

artifact = {
    "features": INTL_FEATURES,
    "medians": medians.to_dict(),
    "target_cols": TARGET_COLS,
    "real_leave_one_out_r2": REAL_LOO_R2,
    "n_training_rows": len(trainable),
    "training_years": [int(trainable["real_draft_year"].min()), int(trainable["real_draft_year"].max())],
    "known_gaps": [
        "Small sample (n=52) -- real signal (beats baseline on 9/10 stats) but "
        "treat as directional, revisit as more international rookies accumulate "
        "real NBA rookie seasons.",
        "STL has no real signal here either (R2=-0.022), consistent with the "
        "college model and the original in-season Kalman filter test -- steals "
        "appear genuinely close to unpredictable from any static profile.",
    ],
}
with open(ROOT / "data" / "output_b_model_international.pkl", "wb") as f:
    pickle.dump({**artifact, "models": models}, f)
with open(ROOT / "data" / "output_b_model_international.meta.json", "w") as f:
    json.dump(artifact, f, indent=2)

print(f"\nSaved output_b_model_international.pkl ({len(models)} stats)")


def project_international_rookie(feature_row: dict) -> dict:
    x = np.array([[feature_row.get(f, medians[f]) for f in INTL_FEATURES]], dtype=float)
    x = np.where(np.isnan(x), [medians[f] for f in INTL_FEATURES], x)
    out = {}
    for stat, m in models.items():
        out[stat] = float(m["model"].predict(m["scaler"].transform(x))[0])
    return out


if __name__ == "__main__":
    for name in ["Wembanyama", "Doncic"]:
        rows = trainable[trainable["player"].str.contains(name, case=False, na=False)]
        if rows.empty:
            print(f"{name}: not found")
            continue
        row = rows.iloc[0]
        feats = {f: row[f] for f in INTL_FEATURES}
        proj = project_international_rookie(feats)
        actual = {s: row[TARGET_COLS[s]] for s in TARGET_COLS}
        print(f"\n{row['player']} (draft {int(row['real_draft_year'])}):")
        for s in TARGET_COLS:
            print(f"  {s:5s} projected={proj[s]:.4f}  actual={actual[s]:.4f}")
