"""Fit the final "why is he hot" persistence model and write research/form_model.json.
Target = his points per game over the next 12 games minus his baseline (previous 365 days). Features = the pieces of his recent (last-10-game) change from baseline.
Reports out-of-sample (2024-25 and 2025-26, fit on 2010-11..2023-24) error for: no form, plain blend, and the split; then refits on everything for production."""
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
import form_common as F

P = pd.read_csv(F.D.parent / "form" / "panel.csv")
bd = pd.read_csv(F.D.parent / "birthdates_all.csv")
bd["bd"] = pd.to_datetime(bd.BIRTHDATE)
P = P.merge(bd[["PERSON_ID", "bd"]], left_on="pid", right_on="PERSON_ID", how="left")
P["age"] = ((pd.to_datetime(P.end) - P.bd).dt.days / 365.25).fillna(26.0)
P["d"] = P.fp_w - P.fp_b
P["target"] = P.fp_f - P.fp_b
P["trend"] = P.fp_l3 - P.fp_w
P["mtrend"] = P.m_l3 - P.m_w
P["young"] = np.clip(24 - P.age, 0, None)
P["old"] = np.clip(P.age - 31, 0, None)
P["o_misc"] = P.o_tov + P.o_rest
COMP = ["minutes", "luck3", "luck2", "luckf", "volume", "o_reb", "o_ast", "o_stk", "o_misc"]
FEAT = COMP + ["trend", "mtrend", "young", "old"]
tr, te = P[~P.season.isin(["2024-25", "2025-26"])], P[P.season.isin(["2024-25", "2025-26"])]


def fit(df, cols, lam=0.0):
    X = np.c_[np.ones(len(df)), df[cols].values]
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ df.target.values)


def rmse(df, cols, b):
    e = df.target.values - (np.c_[np.ones(len(df)), df[cols].values] @ b if cols else b[0])
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(abs(e)))


res = {}
res["no_form"] = (float(np.sqrt(np.mean(te.target ** 2))), float(np.mean(abs(te.target))))
b1 = fit(tr, ["d"])
res["plain_blend"] = rmse(te, ["d"], b1)
b2 = fit(tr, FEAT, lam=50.0)
res["split"] = rmse(te, FEAT, b2)
for k, v in res.items():
    print(f"{k:12s} RMSE {v[0]:.3f} MAE {v[1]:.3f}")
print("blend persistence", b1.round(2))
print("split coef", dict(zip(["int"] + FEAT, b2.round(2))))
te = te.assign(pred=np.c_[np.ones(len(te)), te[FEAT].values] @ b2)
te["bin"] = pd.qcut(te.pred, 8)
cal = te.groupby("bin", observed=True)[["pred", "target"]].mean().round(2)
print(cal)
# full refit for production
bF = fit(P, FEAT, lam=50.0)
bBlend = fit(P, ["d"])
print("final coef", dict(zip(["int"] + FEAT, bF.round(2))), "blend", bBlend.round(2))
model = {"features": FEAT, "coef": dict(zip(["int"] + FEAT, [round(float(x), 4) for x in bF])), "blend": [round(float(x), 4) for x in bBlend], "window": 10, "horizon": 12, "baseline_days": 365, "k": F.K, "lg": F.LG,
         "n_train": int(len(P)), "test": {k: {"rmse": round(v[0], 3), "mae": round(v[1], 3)} for k, v in res.items()}, "calibration": [[float(a), float(b)] for a, b in cal.values]}
json.dump(model, open(Path := __file__.rsplit("form_fit_final.py", 1)[0] + "form_model.json", "w"), indent=1)
print("wrote", Path)
