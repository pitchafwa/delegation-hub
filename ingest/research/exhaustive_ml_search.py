"""Overnight exhaustive search: no longer constrained to the additive
bounded-ramp scoring function. Tries purpose-built learning-to-rank models
(XGBoost/LightGBM, trained directly against NDCG) and flexible tree
ensembles (which capture feature interactions automatically, unlike the
additive model), all evaluated via the same real LOCO-CV NDCG discipline
used throughout this project. Every model gets draft position (pick_filled)
as a raw input and has to beat the draft-alone baseline using it plus
everything else, not just the fixed capital curve shape.

Run: python exhaustive_ml_search.py <vor_linear|vor_convex>
"""
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET_COL = sys.argv[1] if len(sys.argv) > 1 else "vor_convex"

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
target = pd.read_csv(ROOT / "data" / "target_window_test.csv")
df = df.merge(
    target[["PLAYER_ID", "vor_linear", "vor_convex", "confidence_adjusted_outcome", "outcome_is_projected"]],
    on="PLAYER_ID", how="left",
)
df = df.dropna(subset=["vor_linear"]).copy()
df = df[df["real_draft_year"] >= 2008].copy()
print(f"Trainable rows: {len(df)} (college: {(df['data_source']=='college').sum()}, "
      f"international: {(df['data_source']=='international').sum()}), classes "
      f"{int(df['real_draft_year'].min())}-{int(df['real_draft_year'].max())}", flush=True)

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)

RAW_FEATURES = [
    "talent_pctile", "rec_filled", "exp_numeric", "draft_age_filled", "breakout_age_filled",
    "porpag", "usg", "ts", "ortg", "obpm", "dbpm", "bpm", "stops",
    "oreb_rate", "dreb_rate", "ast_to", "ftr", "pfr",
    "WINGSPAN_PCTILE", "STANDING_REACH_PCTILE", "STANDING_VERTICAL_LEAP_PCTILE",
    "MAX_VERTICAL_LEAP_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE",
    "three_pct", "rim_pct", "mid_pct", "pick_filled",
]
RAW_FEATURES = [f for f in RAW_FEATURES if f in df.columns]
print(f"Feature set ({len(RAW_FEATURES)}): {RAW_FEATURES}", flush=True)

X_raw = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
X_imputed = X_raw.fillna(X_raw.median())
y = df[TARGET_COL].to_numpy(dtype=float)
years = df["real_draft_year"].to_numpy()
classes = sorted(df["real_draft_year"].unique())


def to_relevance_grades(y_arr, n_bins=9):
    """XGBRanker/LGBMRanker with an NDCG-style objective require small
    nonnegative INTEGER relevance labels, not continuous values. Below-
    replacement (VOR <= 0) all get relevance 0 -- consistent with not
    caring about ordering among replacement-level players. Positive VOR
    values get graded into n_bins increasing integer buckets by magnitude.
    This is only used for the ranker's training labels; real evaluation
    always uses the true continuous y via the real ndcg() function."""
    rel = np.zeros(len(y_arr), dtype=int)
    pos_mask = y_arr > 1e-9
    if pos_mask.sum() > 0:
        grades = pd.qcut(y_arr[pos_mask], q=min(n_bins, pos_mask.sum()), labels=False, duplicates="drop")
        rel[pos_mask] = grades.astype(int) + 1
    return rel


y_rel = to_relevance_grades(y)


def ndcg(pred, rel):
    order = np.argsort(-pred)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = np.sum(rel[order] * discounts)
    ideal_order = np.argsort(-rel)
    idcg = np.sum(rel[ideal_order] * discounts)
    if idcg <= 1e-9:
        return None
    return dcg / idcg


def baseline_ndcg():
    scores = []
    for yr in classes:
        mask = years == yr
        if mask.sum() < 10:
            continue
        pred = -df.loc[mask, "pick_filled"].to_numpy(dtype=float)
        rel = y[mask]
        g = ndcg(pred, rel)
        if g is not None:
            scores.append(g)
    return np.mean(scores)


BASELINE = baseline_ndcg()
print(f"DRAFT-ALONE baseline NDCG vs {TARGET_COL}: {BASELINE:.4f}\n", flush=True)

RESULTS = {}
RESULTS_OOF = {}


def loco_eval(fit_predict_fn, X, use_groups=False, seeds=(42, 7, 123), y_train_source=None):
    """fit_predict_fn(X_train, y_train, X_test, groups_train, seed) -> pred array for test.
    y_train_source lets ranker models train on integer relevance grades
    while everything is still SCORED against the true continuous y.
    Averages predictions over multiple seeds per fold by default -- tree
    models are cheap enough that we can bake in the seed-stability
    discipline the earlier additive-model search needed a separate,
    expensive round-trip to discover was necessary."""
    if y_train_source is None:
        y_train_source = y
    fold_scores = []
    oof_pred = np.full(len(y), np.nan)
    for held_out in classes:
        test_mask = years == held_out
        train_mask = ~test_mask
        if test_mask.sum() < 10:
            continue
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y_train_source[train_mask]
        rel = y[test_mask]
        train_years = years[train_mask]
        groups_train = None
        if use_groups:
            order = np.argsort(train_years)
            X_train, y_train, train_years = X_train[order], y_train[order], train_years[order]
            _, counts = np.unique(train_years, return_counts=True)
            groups_train = counts
        seed_preds = [fit_predict_fn(X_train, y_train, X_test, groups_train, s) for s in seeds]
        pred = np.mean(seed_preds, axis=0)
        g = ndcg(pred, rel)
        if g is not None:
            fold_scores.append(g)
        oof_pred[test_mask] = pred
    return np.mean(fold_scores), fold_scores, oof_pred


# ---------------------------------------------------------------------------
# STAGE 1: XGBoost Ranker (rank:ndcg objective) -- purpose-built for this
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRanker

    Xn = X_raw.to_numpy(dtype=float)  # native NaN support

    for depth, lr, n_est in [
        (3, 0.05, 200), (4, 0.05, 150), (3, 0.1, 100), (2, 0.05, 250),
        (5, 0.03, 300), (3, 0.02, 400), (6, 0.05, 100),
    ]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, depth=depth, lr=lr, n_est=n_est):
            model = XGBRanker(
                objective="rank:ndcg", max_depth=depth, learning_rate=lr, n_estimators=n_est,
                random_state=seed, tree_method="hist", n_jobs=1,
            )
            model.fit(X_train, y_train, group=groups_train)
            return model.predict(X_test)

        g, folds, oof = loco_eval(fit_predict, Xn, use_groups=True, y_train_source=y_rel)
        name = f"xgb_ranker_d{depth}_lr{lr}_n{n_est}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE1] {name:35s} NDCG: {g:.4f}", flush=True)
except Exception:
    print("[STAGE1] XGBRanker FAILED:", flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------------------
# STAGE 2: LightGBM Ranker (lambdarank objective)
# ---------------------------------------------------------------------------
try:
    from lightgbm import LGBMRanker

    Xn = X_raw.to_numpy(dtype=float)

    for depth, lr, n_est in [
        (3, 0.05, 200), (4, 0.05, 150), (3, 0.1, 100), (2, 0.05, 250),
        (5, 0.03, 300), (3, 0.02, 400), (6, 0.05, 100),
    ]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, depth=depth, lr=lr, n_est=n_est):
            model = LGBMRanker(
                objective="lambdarank", max_depth=depth, learning_rate=lr, n_estimators=n_est,
                random_state=seed, n_jobs=1, verbose=-1, min_child_samples=5,
            )
            model.fit(X_train, y_train, group=groups_train)
            return model.predict(X_test)

        g, folds, oof = loco_eval(fit_predict, Xn, use_groups=True, y_train_source=y_rel)
        name = f"lgbm_ranker_d{depth}_lr{lr}_n{n_est}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE2] {name:35s} NDCG: {g:.4f}", flush=True)
except Exception:
    print("[STAGE2] LGBMRanker FAILED:", flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------------------
# STAGE 3: XGBoost / sklearn regressors trained directly on the VOR value
# (captures arbitrary interactions automatically, unlike the additive model)
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBRegressor

    Xn = X_raw.to_numpy(dtype=float)
    for depth, lr, n_est in [(3, 0.05, 200), (4, 0.05, 150), (2, 0.05, 250), (5, 0.03, 300)]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, depth=depth, lr=lr, n_est=n_est):
            model = XGBRegressor(
                max_depth=depth, learning_rate=lr, n_estimators=n_est, random_state=seed,
                tree_method="hist", n_jobs=1, subsample=0.8, colsample_bytree=0.8,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

        g, folds, oof = loco_eval(fit_predict, Xn)
        name = f"xgb_reg_d{depth}_lr{lr}_n{n_est}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE3] {name:35s} NDCG: {g:.4f}", flush=True)
except Exception:
    print("[STAGE3] XGBRegressor FAILED:", flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------------------
# STAGE 4: sklearn HistGradientBoostingRegressor (native NaN support)
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    Xn = X_raw.to_numpy(dtype=float)
    for depth, lr, n_est in [
        (3, 0.05, 200), (5, 0.05, 150), (None, 0.1, 100), (2, 0.05, 250), (4, 0.03, 300),
    ]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, depth=depth, lr=lr, n_est=n_est):
            model = HistGradientBoostingRegressor(
                max_depth=depth, learning_rate=lr, max_iter=n_est, random_state=seed,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

        g, folds, oof = loco_eval(fit_predict, Xn)
        name = f"histgbr_d{depth}_lr{lr}_n{n_est}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE4] {name:35s} NDCG: {g:.4f}", flush=True)
except Exception:
    print("[STAGE4] HistGradientBoostingRegressor FAILED:", flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------------------
# STAGE 5: RandomForest / GradientBoosting (median-imputed)
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    Xi = X_imputed.to_numpy(dtype=float)

    for n_est, max_depth in [(300, 5), (500, 8), (400, 3), (600, 12)]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, n_est=n_est, max_depth=max_depth):
            model = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_depth, random_state=seed, n_jobs=1, min_samples_leaf=5,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

        g, folds, oof = loco_eval(fit_predict, Xi)
        name = f"rf_n{n_est}_d{max_depth}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE5] {name:35s} NDCG: {g:.4f}", flush=True)

    for lr, n_est, depth in [(0.05, 200, 3), (0.03, 300, 4), (0.1, 100, 2)]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, lr=lr, n_est=n_est, depth=depth):
            model = GradientBoostingRegressor(
                learning_rate=lr, n_estimators=n_est, max_depth=depth, random_state=seed,
                subsample=0.8,
            )
            model.fit(X_train, y_train)
            return model.predict(X_test)

        g, folds, oof = loco_eval(fit_predict, Xi)
        name = f"gbr_lr{lr}_n{n_est}_d{depth}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE5] {name:35s} NDCG: {g:.4f}", flush=True)
except Exception:
    print("[STAGE5] RF/GBR FAILED:", flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------------------
# STAGE 5b: regularized linear models (Ridge / ElasticNet / SVR-RBF) on
# standardized, median-imputed features -- covers the "maybe a properly
# regularized linear model beats the hand-built additive one" angle too.
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler

    Xi = X_imputed.to_numpy(dtype=float)

    for alpha in [1.0, 10.0, 50.0]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, alpha=alpha):
            scaler = StandardScaler().fit(X_train)
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(scaler.transform(X_train), y_train)
            return model.predict(scaler.transform(X_test))

        g, folds, oof = loco_eval(fit_predict, Xi)
        name = f"ridge_a{alpha}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE5b] {name:35s} NDCG: {g:.4f}", flush=True)

    for alpha, l1 in [(1.0, 0.5), (0.1, 0.5)]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, alpha=alpha, l1=l1):
            scaler = StandardScaler().fit(X_train)
            model = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=seed, max_iter=5000)
            model.fit(scaler.transform(X_train), y_train)
            return model.predict(scaler.transform(X_test))

        g, folds, oof = loco_eval(fit_predict, Xi)
        name = f"elasticnet_a{alpha}_l1{l1}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE5b] {name:35s} NDCG: {g:.4f}", flush=True)

    for C, gamma in [(1.0, "scale"), (10.0, "scale")]:
        def fit_predict(X_train, y_train, X_test, groups_train, seed, C=C, gamma=gamma):
            scaler = StandardScaler().fit(X_train)
            model = SVR(kernel="rbf", C=C, gamma=gamma)
            model.fit(scaler.transform(X_train), y_train)
            return model.predict(scaler.transform(X_test))

        g, folds, oof = loco_eval(fit_predict, Xi, seeds=(42,))  # SVR is deterministic given data
        name = f"svr_C{C}_g{gamma}"
        RESULTS[name] = g
        RESULTS_OOF[name] = oof
        print(f"[STAGE5b] {name:35s} NDCG: {g:.4f}", flush=True)
except Exception:
    print("[STAGE5b] Ridge/ElasticNet/SVR FAILED:", flush=True)
    traceback.print_exc()

# ---------------------------------------------------------------------------
# STAGE 6: ensemble (rank-average) of the top performers found above
# ---------------------------------------------------------------------------
print(f"\n=== ALL RESULTS vs baseline {BASELINE:.4f} ===", flush=True)
for name, g in sorted(RESULTS.items(), key=lambda kv: -kv[1]):
    beat = "BEATS baseline" if g > BASELINE else ""
    print(f"  {name:35s} {g:.4f}  {beat}", flush=True)

# ---------------------------------------------------------------------------
# STAGE 6: rank-average ensembles of the top individual models
# ---------------------------------------------------------------------------
from scipy.stats import rankdata

ranked_names = [n for n, _ in sorted(RESULTS.items(), key=lambda kv: -kv[1])]


def ensemble_ndcg(names):
    preds = np.array([RESULTS_OOF[n] for n in names])
    valid = ~np.any(np.isnan(preds), axis=0)
    scores = []
    for yr in classes:
        mask = (years == yr) & valid
        if mask.sum() < 10:
            continue
        sub_ranks = np.mean([rankdata(preds[i][mask]) for i in range(len(names))], axis=0)
        g = ndcg(sub_ranks, y[mask])
        if g is not None:
            scores.append(g)
    return np.mean(scores) if scores else None


print(f"\n=== ENSEMBLES (rank-averaged top-K models) vs baseline {BASELINE:.4f} ===", flush=True)
for k in [2, 3, 5, len(ranked_names)]:
    if k > len(ranked_names):
        continue
    names_k = ranked_names[:k]
    eg = ensemble_ndcg(names_k)
    if eg is not None:
        beat = "BEATS baseline" if eg > BASELINE else ""
        print(f"  top-{k:<2d} ensemble: {eg:.4f}  {beat}  ({names_k})", flush=True)

print("\n[STAGE6] Done. See above for best individual model(s) and ensembles.", flush=True)
