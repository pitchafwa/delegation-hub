"""Output B: project a prospect's ROOKIE-SEASON per-stat baseline from their
pre-NBA profile, to replace the flat league-average x0 that kalman_vor.py
currently uses for every player regardless of who they are (see year0
construction there: x0 = population-wide per-minute rate, identical for a
projected #1 pick and a fringe second-rounder).

10 targets: the 9 Kalman-relevant rates (PTS/REB/AST/BLK/TOV/FG3M/FTM/FTA
per-minute + MIN per-game) plus STL/min (not Kalman-filtered but currently
defaults to a flat 0.0 for a zero-game rookie in kalman_vor.py -- worse than
the other 9 stats' global-average fallback, so it's included here too).

Validated the same way as every model in this project: LOCO-CV by draft
class. The bar to clear here is much lower than "beat draft order" --  it's
"beat predicting the same single number for every rookie regardless of who
they are," which is the actual status quo being replaced.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")

# --- add rookie STL/min, the one Output B target build_rookie_features.py
# didn't build (it only covered the 9 Kalman-relevant stats) ---
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

df = df[df["real_draft_year"] >= 2008].copy()
print(f"Total prospect rows: {len(df)} (college: {(df['data_source']=='college').sum()}, "
      f"international: {(df['data_source']=='international').sum()})")
print(f"Rows with a resolvable Output B target: {df['rookie_MIN_per_game'].notna().sum()} "
      f"(years {int(df.loc[df['rookie_MIN_per_game'].notna(),'real_draft_year'].min())}-"
      f"{int(df.loc[df['rookie_MIN_per_game'].notna(),'real_draft_year'].max())})\n", flush=True)

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
X_imputed = X_raw.fillna(X_raw.median())
years = trainable["real_draft_year"].to_numpy()
classes = sorted(trainable["real_draft_year"].unique())


def loco_predict(fit_fn, X, y, seed=42):
    """Real player-level out-of-fold predictions via leave-one-draft-class-out."""
    oof = np.full(len(y), np.nan)
    for held_out in classes:
        test_mask = years == held_out
        train_mask = ~test_mask
        if test_mask.sum() < 5:
            continue
        oof[test_mask] = fit_fn(X[train_mask], y[train_mask], X[test_mask], seed)
    return oof


def r2(pred, actual):
    valid = ~np.isnan(pred) & ~np.isnan(actual)
    if valid.sum() < 5:
        return None
    ss_res = np.sum((actual[valid] - pred[valid]) ** 2)
    ss_tot = np.sum((actual[valid] - actual[valid].mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-9 else None


def mae(pred, actual):
    valid = ~np.isnan(pred) & ~np.isnan(actual)
    return np.mean(np.abs(actual[valid] - pred[valid])) if valid.sum() > 0 else None


def corr(pred, actual):
    valid = ~np.isnan(pred) & ~np.isnan(actual)
    if valid.sum() < 5:
        return None
    return np.corrcoef(pred[valid], actual[valid])[0, 1]


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

MODEL_FACTORIES = {
    "ridge": lambda seed: ("imputed", Ridge(alpha=10.0, random_state=seed)),
    "rf": lambda seed: ("imputed", RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=seed, n_jobs=1)),
    "gbr": lambda seed: ("imputed", GradientBoostingRegressor(learning_rate=0.05, n_estimators=150, max_depth=3, subsample=0.8, random_state=seed)),
    "histgbr": lambda seed: ("raw", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, random_state=seed)),
}


def make_fit_fn(model_name):
    def fit_fn(X_train, y_train, X_test, seed):
        kind, model = MODEL_FACTORIES[model_name](seed)
        if kind == "imputed" and model_name == "ridge":
            scaler = StandardScaler().fit(X_train)
            model.fit(scaler.transform(X_train), y_train)
            return model.predict(scaler.transform(X_test))
        model.fit(X_train, y_train)
        return model.predict(X_test)
    return fit_fn


Xi = X_imputed.to_numpy(dtype=float)
Xn = X_raw.to_numpy(dtype=float)

RESULTS = {}
for stat, target_col in TARGET_COLS.items():
    y = trainable[target_col].to_numpy(dtype=float)
    valid_mask = ~np.isnan(y)
    y_valid = np.where(valid_mask, y, np.nan)

    # baseline this replaces: same global mean rate for every prospect
    global_mean = np.nanmean(y_valid)
    baseline_pred = np.full(len(y_valid), global_mean)
    base_r2 = r2(baseline_pred, y_valid)
    base_mae = mae(baseline_pred, y_valid)
    base_corr = corr(baseline_pred, y_valid)

    print(f"=== {stat} (target={target_col}, n={valid_mask.sum()}) ===", flush=True)
    print(f"  GLOBAL-AVERAGE baseline: R2={base_r2:.3f}  MAE={base_mae:.4f}  corr={base_corr}", flush=True)

    best_name, best_r2 = None, -np.inf
    for model_name in MODEL_FACTORIES:
        X_use = Xn if model_name == "histgbr" else Xi
        fit_fn = make_fit_fn(model_name)
        oof = loco_predict(fit_fn, X_use, y, seed=42)
        g_r2 = r2(oof, y)
        g_mae = mae(oof, y)
        g_corr = corr(oof, y)
        beat = "BEATS baseline" if (g_r2 is not None and g_r2 > base_r2) else ""
        print(f"    {model_name:10s} R2={g_r2:.3f}  MAE={g_mae:.4f}  corr={g_corr:.3f}  {beat}", flush=True)
        if g_r2 is not None and g_r2 > best_r2:
            best_r2, best_name = g_r2, model_name
    RESULTS[stat] = {"baseline_r2": base_r2, "best_model": best_name, "best_r2": best_r2}
    print(f"  --> best: {best_name} (R2={best_r2:.3f} vs baseline {base_r2:.3f})\n", flush=True)

print("=== SUMMARY ===")
for stat, r in RESULTS.items():
    gain = r["best_r2"] - r["baseline_r2"]
    print(f"  {stat:5s} best={r['best_model']:10s} R2 {r['baseline_r2']:.3f} -> {r['best_r2']:.3f}  (+{gain:.3f})")
