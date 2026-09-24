"""Held-out validation of the best convex-target ensemble found in
exhaustive_ml_search.py: rank-averaged (LightGBM ranker + XGBoost ranker).
That combo was picked AFTER comparing ~37 candidates evaluated on seeds
(42, 7, 123) -- picking the best of many tries on the same seeds is exactly
the kind of overfitting that sank the two earlier "winners" in this
project. This script re-evaluates the SAME two model configs on entirely
fresh seeds never used during model selection, as real out-of-sample
confirmation.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lightgbm import LGBMRanker
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parent
TARGET_COL = "vor_convex"

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
target = pd.read_csv(ROOT / "data" / "target_window_test.csv")
df = df.merge(target[["PLAYER_ID", "vor_linear", "vor_convex"]], on="PLAYER_ID", how="left")
df = df.dropna(subset=["vor_linear"]).copy()
df = df[df["real_draft_year"] >= 2008].copy()

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
X_raw = df[RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
y = df[TARGET_COL].to_numpy(dtype=float)
years = df["real_draft_year"].to_numpy()
classes = sorted(df["real_draft_year"].unique())
Xn = X_raw.to_numpy(dtype=float)


def to_relevance_grades(y_arr, n_bins=9):
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
        g = ndcg(-df.loc[mask, "pick_filled"].to_numpy(dtype=float), y[mask])
        if g is not None:
            scores.append(g)
    return np.mean(scores)


BASELINE = baseline_ndcg()


def fit_predict_lgbm(X_train, y_train, X_test, groups_train, seed):
    model = LGBMRanker(
        objective="lambdarank", max_depth=6, learning_rate=0.05, n_estimators=100,
        random_state=seed, n_jobs=1, verbose=-1, min_child_samples=5,
    )
    model.fit(X_train, y_train, group=groups_train)
    return model.predict(X_test)


def fit_predict_xgb(X_train, y_train, X_test, groups_train, seed):
    model = XGBRanker(
        objective="rank:ndcg", max_depth=3, learning_rate=0.05, n_estimators=200,
        random_state=seed, tree_method="hist", n_jobs=1,
    )
    model.fit(X_train, y_train, group=groups_train)
    return model.predict(X_test)


def loco_predict(fit_predict_fn, seed):
    oof = np.full(len(y), np.nan)
    for held_out in classes:
        test_mask = years == held_out
        train_mask = ~test_mask
        if test_mask.sum() < 10:
            continue
        X_train, X_test = Xn[train_mask], Xn[test_mask]
        y_train = y_rel[train_mask]
        train_years = years[train_mask]
        order = np.argsort(train_years)
        X_train, y_train, train_years_s = X_train[order], y_train[order], train_years[order]
        _, counts = np.unique(train_years_s, return_counts=True)
        pred = fit_predict_fn(X_train, y_train, X_test, counts, seed)
        oof[test_mask] = pred
    return oof


def score_oof(oof):
    scores = []
    for yr in classes:
        mask = (years == yr) & ~np.isnan(oof)
        if mask.sum() < 10:
            continue
        g = ndcg(oof[mask], y[mask])
        if g is not None:
            scores.append(g)
    return np.mean(scores)


if __name__ == "__main__":
    print(f"DRAFT-ALONE baseline NDCG vs vor_convex: {BASELINE:.4f}\n", flush=True)
    # HOLD-OUT seeds -- never used during the model-selection search (which used 42, 7, 123)
    HOLDOUT_SEEDS = [999, 2024, 55, 1776, 314]
    lgbm_scores, xgb_scores, ens_scores = [], [], []
    for seed in HOLDOUT_SEEDS:
        lgbm_oof = loco_predict(fit_predict_lgbm, seed)
        xgb_oof = loco_predict(fit_predict_xgb, seed)
        lg = score_oof(lgbm_oof)
        xg = score_oof(xgb_oof)
        # rank-average ensemble, computed per class
        ens_oof = np.full(len(y), np.nan)
        valid = ~np.isnan(lgbm_oof) & ~np.isnan(xgb_oof)
        for yr in classes:
            mask = (years == yr) & valid
            if mask.sum() < 10:
                continue
            r = np.mean([rankdata(lgbm_oof[mask]), rankdata(xgb_oof[mask])], axis=0)
            ens_oof[mask] = r
        eg = score_oof(ens_oof)
        lgbm_scores.append(lg)
        xgb_scores.append(xg)
        ens_scores.append(eg)
        print(f"seed={seed:5d}  lgbm_ranker={lg:.4f}  xgb_ranker={xg:.4f}  ENSEMBLE={eg:.4f}"
              f"  {'BEATS baseline' if eg > BASELINE else 'below baseline'}", flush=True)

    print(f"\nMean across {len(HOLDOUT_SEEDS)} holdout seeds:")
    print(f"  lgbm_ranker solo mean: {np.mean(lgbm_scores):.4f}")
    print(f"  xgb_ranker solo mean:  {np.mean(xgb_scores):.4f}")
    print(f"  ENSEMBLE mean:         {np.mean(ens_scores):.4f}  (baseline={BASELINE:.4f})")
