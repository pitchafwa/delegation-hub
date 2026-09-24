"""Real forward-holdout validation: train the winning ensembles ONLY on
draft classes 2008-2019 (never touched during model selection), then test
on 2020-2023 -- classes that had zero influence on which model got picked.
This is the actual defense against "tried 37 models, picked the best one"
selection bias, and it also mirrors Tommy's real use case: predicting
classes whose long-run outcomes aren't known yet.

Winners being validated (from exhaustive_ml_search.py):
  convex: lgbm_ranker(d6,lr.05,n100) + xgb_ranker(d3,lr.05,n200), rank-avg
  linear: rf(n300,d5) + lgbm_ranker(d6,lr.05,n100), rank-avg
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lightgbm import LGBMRanker
from xgboost import XGBRanker
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent

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
X_imputed = X_raw.fillna(X_raw.median())
years = df["real_draft_year"].to_numpy()

SELECTION_YEARS = list(range(2008, 2020))  # used for everything so far
HOLDOUT_YEARS = list(range(2020, 2024))    # untouched until now

train_mask = np.isin(years, SELECTION_YEARS)
test_mask = np.isin(years, HOLDOUT_YEARS)
print(f"Train (selection) years: {SELECTION_YEARS} -> {train_mask.sum()} rows")
print(f"Holdout (confirmation) years: {HOLDOUT_YEARS} -> {test_mask.sum()} rows\n", flush=True)


def to_relevance_grades(y_arr, n_bins=9):
    rel = np.zeros(len(y_arr), dtype=int)
    pos_mask = y_arr > 1e-9
    if pos_mask.sum() > 0:
        grades = pd.qcut(y_arr[pos_mask], q=min(n_bins, pos_mask.sum()), labels=False, duplicates="drop")
        rel[pos_mask] = grades.astype(int) + 1
    return rel


def ndcg(pred, rel):
    order = np.argsort(-pred)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = np.sum(rel[order] * discounts)
    ideal_order = np.argsort(-rel)
    idcg = np.sum(rel[ideal_order] * discounts)
    if idcg <= 1e-9:
        return None
    return dcg / idcg


def eval_by_class(pred, rel_true, mask, yrs):
    scores = []
    for yr in HOLDOUT_YEARS:
        sub = mask & (yrs == yr)
        if sub.sum() < 5:
            continue
        g = ndcg(pred[sub[mask]] if False else pred[(yrs[mask] == yr)], rel_true[(yrs[mask] == yr)])
        if g is not None:
            scores.append(g)
    return np.mean(scores) if scores else None


def run_target(target_col):
    y = df[target_col].to_numpy(dtype=float)
    y_rel_train = to_relevance_grades(y[train_mask])
    Xn_train, Xn_test = X_raw.to_numpy(dtype=float)[train_mask], X_raw.to_numpy(dtype=float)[test_mask]
    Xi_train, Xi_test = X_imputed.to_numpy(dtype=float)[train_mask], X_imputed.to_numpy(dtype=float)[test_mask]
    y_test = y[test_mask]
    test_years = years[test_mask]

    # baseline
    base_scores = []
    for yr in HOLDOUT_YEARS:
        sub = test_years == yr
        if sub.sum() < 5:
            continue
        g = ndcg(-df.loc[test_mask, "pick_filled"].to_numpy(dtype=float)[sub], y_test[sub])
        if g is not None:
            base_scores.append(g)
    baseline = np.mean(base_scores)
    print(f"[{target_col}] DRAFT-ALONE baseline on holdout classes: {baseline:.4f}", flush=True)

    # sort train by year for group-based rankers
    train_years_sorted_idx = np.argsort(years[train_mask])
    Xn_train_s = Xn_train[train_years_sorted_idx]
    y_rel_train_s = y_rel_train[train_years_sorted_idx]
    _, group_counts = np.unique(years[train_mask][train_years_sorted_idx], return_counts=True)

    lgbm = LGBMRanker(objective="lambdarank", max_depth=6, learning_rate=0.05, n_estimators=100,
                       random_state=42, n_jobs=1, verbose=-1, min_child_samples=5)
    lgbm.fit(Xn_train_s, y_rel_train_s, group=group_counts)
    lgbm_pred = lgbm.predict(Xn_test)

    xgbr = XGBRanker(objective="rank:ndcg", max_depth=3, learning_rate=0.05, n_estimators=200,
                      random_state=42, tree_method="hist", n_jobs=1)
    xgbr.fit(Xn_train_s, y_rel_train_s, group=group_counts)
    xgb_pred = xgbr.predict(Xn_test)

    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42, n_jobs=1, min_samples_leaf=5)
    rf.fit(Xi_train, y[train_mask])
    rf_pred = rf.predict(Xi_test)

    def score_per_class(pred):
        scores = []
        for yr in HOLDOUT_YEARS:
            sub = test_years == yr
            if sub.sum() < 5:
                continue
            g = ndcg(pred[sub], y_test[sub])
            if g is not None:
                scores.append(g)
        return np.mean(scores) if scores else None

    def ensemble_score(preds_list):
        scores = []
        for yr in HOLDOUT_YEARS:
            sub = test_years == yr
            if sub.sum() < 5:
                continue
            r = np.mean([rankdata(p[sub]) for p in preds_list], axis=0)
            g = ndcg(r, y_test[sub])
            if g is not None:
                scores.append(g)
        return np.mean(scores) if scores else None

    lgbm_g = score_per_class(lgbm_pred)
    xgb_g = score_per_class(xgb_pred)
    rf_g = score_per_class(rf_pred)
    convex_ens_g = ensemble_score([lgbm_pred, xgb_pred])
    linear_ens_g = ensemble_score([rf_pred, lgbm_pred])

    print(f"[{target_col}] lgbm_ranker solo:            {lgbm_g:.4f}", flush=True)
    print(f"[{target_col}] xgb_ranker solo:             {xgb_g:.4f}", flush=True)
    print(f"[{target_col}] rf solo:                     {rf_g:.4f}", flush=True)
    print(f"[{target_col}] ENSEMBLE (lgbm+xgb, convex winner combo):  {convex_ens_g:.4f}"
          f"  {'BEATS' if convex_ens_g > baseline else 'below'} baseline {baseline:.4f}", flush=True)
    print(f"[{target_col}] ENSEMBLE (rf+lgbm, linear winner combo):   {linear_ens_g:.4f}"
          f"  {'BEATS' if linear_ens_g > baseline else 'below'} baseline {baseline:.4f}\n", flush=True)


if __name__ == "__main__":
    run_target("vor_convex")
    run_target("vor_linear")
