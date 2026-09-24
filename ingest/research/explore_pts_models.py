"""Points was the weakest Output B stat (ridge R2=0.156 vs the 4 models
already tried in fit_output_b.py). Trying a wider net specifically for
points: different regularization (ElasticNet/Lasso), a comp-based approach
(KNN -- literally averaging the real rookie outcomes of the most similar
prospects, which is close to how a scout thinks about it), SVR, and a
rank-averaged blend of the best performers. Same real LOCO-CV-by-draft-class
discipline as everything else in this project.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
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
X_imputed = X_raw.fillna(X_raw.median()).to_numpy(dtype=float)
y = trainable["rookie_PTS_per_min"].to_numpy(dtype=float)
years = trainable["real_draft_year"].to_numpy()
classes = sorted(trainable["real_draft_year"].unique())
n = len(trainable)
print(f"n={n}, features={len(FEATURES)}", flush=True)


def r2(pred, actual):
    valid = ~np.isnan(pred) & ~np.isnan(actual)
    ss_res = np.sum((actual[valid] - pred[valid]) ** 2)
    ss_tot = np.sum((actual[valid] - actual[valid].mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 1e-9 else None


def mae(pred, actual):
    valid = ~np.isnan(pred) & ~np.isnan(actual)
    return np.mean(np.abs(actual[valid] - pred[valid]))


def loco_predict(fit_fn):
    oof = np.full(n, np.nan)
    for held_out in classes:
        test_mask = years == held_out
        train_mask = ~test_mask
        if test_mask.sum() < 5:
            continue
        oof[test_mask] = fit_fn(X_imputed[train_mask], y[train_mask], X_imputed[test_mask])
    return oof


baseline_pred = np.full(n, np.nanmean(y))
print(f"baseline (global mean)      R2={r2(baseline_pred, y):.3f}  MAE={mae(baseline_pred, y):.4f}")
print(f"ridge (established)         R2=0.156  MAE=0.0777  (from fit_output_b.py, for comparison)\n")

from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def scaled_fit(model_factory):
    def fit_fn(X_train, y_train, X_test):
        scaler = StandardScaler().fit(X_train)
        model = model_factory()
        model.fit(scaler.transform(X_train), y_train)
        return model.predict(scaler.transform(X_test))
    return fit_fn


CANDIDATES = {
    "elasticnet_a0.1_l0.5": scaled_fit(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    "elasticnet_a0.01_l0.5": scaled_fit(lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
    "lasso_a0.01": scaled_fit(lambda: Lasso(alpha=0.01, max_iter=5000)),
    "knn_k10": scaled_fit(lambda: KNeighborsRegressor(n_neighbors=10)),
    "knn_k20": scaled_fit(lambda: KNeighborsRegressor(n_neighbors=20)),
    "knn_k30_weighted": scaled_fit(lambda: KNeighborsRegressor(n_neighbors=30, weights="distance")),
    "svr_rbf_C1": scaled_fit(lambda: SVR(kernel="rbf", C=1.0)),
    "svr_rbf_C10": scaled_fit(lambda: SVR(kernel="rbf", C=10.0)),
    "ridge_a50": scaled_fit(lambda: Ridge(alpha=50.0)),
    "ridge_a100": scaled_fit(lambda: Ridge(alpha=100.0)),
}

preds = {}
for name, fit_fn in CANDIDATES.items():
    oof = loco_predict(fit_fn)
    g_r2, g_mae = r2(oof, y), mae(oof, y)
    preds[name] = oof
    beat = "BEATS 0.156" if g_r2 > 0.156 else ""
    print(f"  {name:24s} R2={g_r2:.3f}  MAE={g_mae:.4f}  {beat}", flush=True)

# rank-averaged blend of best 2-3
from scipy.stats import rankdata
ranked = sorted(preds.items(), key=lambda kv: -r2(kv[1], y))
for k in [2, 3]:
    names_k = [n for n, _ in ranked[:k]]
    blend = np.mean([rankdata(preds[n]) for n in names_k], axis=0)
    # rank-average isn't directly comparable to real per-minute value scale for R2/MAE,
    # so also do a plain value-average blend (more meaningful for a regression target)
    val_blend = np.mean([preds[n] for n in names_k], axis=0)
    g_r2, g_mae = r2(val_blend, y), mae(val_blend, y)
    beat = "BEATS 0.156" if g_r2 > 0.156 else ""
    print(f"  BLEND top-{k} (value-avg): {names_k}  R2={g_r2:.3f}  MAE={g_mae:.4f}  {beat}", flush=True)

print("\nDone.")
