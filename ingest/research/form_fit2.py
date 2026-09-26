import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
import form_common as F

P = pd.read_csv(F.D.parent / "form" / "panel.csv")
bd = pd.read_csv(F.D.parent / "birthdates_all.csv")
bd["bd"] = pd.to_datetime(bd.BIRTHDATE)
P = P.merge(bd[["PERSON_ID", "bd"]], left_on="pid", right_on="PERSON_ID", how="left")
P["age"] = (pd.to_datetime(P.end) - P.bd).dt.days / 365.25
P["d"] = P.fp_w - P.fp_b
P["dabs"] = P.abs_w - P.abs_b
P["target"] = P.fp_f - P.fp_b
P["trend"] = P.fp_l3 - P.fp_w
P["mtrend"] = P.m_l3 - P.m_w
P["dm"] = P.m_w - P.m_b
P["absdm"] = P.dm.abs()
P["bigmin"] = np.where(P.absdm > 4, P.dm, 0.0)
P["age_c"] = (P.age.fillna(26) - 26)
P["young"] = np.clip(24 - P.age.fillna(26), 0, None)
P["old"] = np.clip(P.age.fillna(26) - 31, 0, None)
tr, te = P[~P.season.isin(["2024-25", "2025-26"])], P[P.season.isin(["2024-25", "2025-26"])]


def ols(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def ev(cols, name):
    b = ols(tr[cols].values, tr.target.values)
    err = te.target.values - np.c_[np.ones(len(te)), te[cols].values] @ b
    print(f"{name:40s} MAE {np.mean(abs(err)):.3f} RMSE {np.sqrt(np.mean(err**2)):.3f}  " + " ".join(f"{c}={v:+.2f}" for c, v in zip(["int"] + cols, b)))


comp = ["minutes", "luck", "volume", "other"]
ev(["d"], "blend")
ev(comp, "split")
ev(comp + ["trend", "mtrend"], "split + last-3")
ev(comp + ["trend", "mtrend", "young", "old"], "split + last-3 + age")
ev(comp + ["trend", "mtrend", "young", "old", "bigmin"], "... + big-minute-change")
ev(comp + ["trend", "mtrend", "young", "old", "fp_b"], "... + baseline level")
feats = comp + ["trend", "mtrend", "young", "old", "fp_b", "m_b", "m_w", "dabs", "age_c"]
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4, min_samples_leaf=60, l2_regularization=1.0).fit(tr[feats], tr.target)
err = te.target.values - gb.predict(te[feats])
print(f"{'GBM (ceiling check)':40s} MAE {np.mean(abs(err)):.3f} RMSE {np.sqrt(np.mean(err**2)):.3f}")
# season-by-season out of sample stability of split vs blend
for s in ("2024-25", "2025-26"):
    a, c = tr, P[P.season == s]
    for nm, cols in (("blend", ["d"]), ("split+last3+age", comp + ["trend", "mtrend", "young", "old"])):
        b = ols(a[cols].values, a.target.values)
        e = c.target.values - np.c_[np.ones(len(c)), c[cols].values] @ b
        print(s, nm, "RMSE", round(float(np.sqrt(np.mean(e ** 2))), 3))
# calibration of the chosen model by decile of predicted change
cols = comp + ["trend", "mtrend", "young", "old"]
b = ols(tr[cols].values, tr.target.values)
te = te.assign(pred=np.c_[np.ones(len(te)), te[cols].values] @ b)
te["bin"] = pd.qcut(te.pred, 8)
print(te.groupby("bin", observed=True)[["pred", "target"]].mean().round(2))
