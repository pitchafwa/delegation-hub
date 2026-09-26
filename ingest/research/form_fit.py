import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
import form_common as F

P = pd.read_csv(F.D.parent / "form" / "panel.csv")
P["d"] = P.fp_w - P.fp_b
P["dabs"] = P.abs_w - P.abs_b
print("identity check (max abs err):", (P.d - P[["minutes", "luck", "volume", "other"]].sum(axis=1)).abs().max())
P["target"] = P.fp_f - P.fp_b
tr, te = P[~P.season.isin(["2024-25", "2025-26"])], P[P.season.isin(["2024-25", "2025-26"])]
print(len(tr), len(te))


def ols(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b


def ev(cols, name, tr=tr, te=te):
    b = ols(tr[cols].values, tr.target.values)
    pred = np.c_[np.ones(len(te)), te[cols].values] @ b
    err = te.target.values - pred
    print(f"{name:34s} test MAE {np.mean(abs(err)):.3f} RMSE {np.sqrt(np.mean(err**2)):.3f}   coef: " + " ".join(f"{c}={v:+.2f}" for c, v in zip(["int"] + cols, b)))
    return b


base_err = te.target.values
print(f"{'baseline only (no form)':34s} test MAE {np.mean(abs(base_err)):.3f} RMSE {np.sqrt(np.mean(base_err**2)):.3f}")
ev([], "intercept")
ev(["d"], "blend (recent - baseline)")
ev(["minutes", "luck", "volume", "other"], "split")
ev(["minutes", "luck", "volume", "other", "dabs"], "split + absence")
ev(["minutes", "luck", "volume", "other", "dabs", "fp_b"], "split + absence + level")
# by player tier
for lo, hi in [(0, 22), (22, 30), (30, 40), (40, 99)]:
    a, c = tr[(tr.fp_b >= lo) & (tr.fp_b < hi)], te[(te.fp_b >= lo) & (te.fp_b < hi)]
    print(f"--- baseline fp {lo}-{hi}: train {len(a)} test {len(c)}")
    ev(["d"], "  blend", a, c)
    ev(["minutes", "luck", "volume", "other", "dabs"], "  split+abs", a, c)

print("\n=== extras ===")
beta = ols(tr[["dabs"]].values, tr.d.values)[1]
print("fp change per 1 extra absent-regular minute in window:", round(beta, 4))
for D_ in (tr, te, P):
    D_["d_abs"] = beta * D_["dabs"]
    D_["d_rest"] = D_["d"] - D_["d_abs"]
    D_["dabs_f"] = D_.abs_f - D_.abs_b
    D_["trend"] = D_.fp_l3 - D_.fp_w
    D_["mtrend"] = D_.m_l3 - D_.m_w
ev(["d_abs", "d_rest"], "blend split absence/rest")
ev(["d_abs", "minutes", "luck", "volume", "other"], "split + absence part")
ev(["minutes", "luck", "volume", "other", "dabs_f"], "split + future absence (oracle)")
ev(["minutes", "luck", "volume", "other", "trend", "mtrend"], "split + last-3 trend")
for name, sub in [("minutes up >3", P[P.m_w - P.m_b > 3]), ("minutes flat", P[(P.m_w - P.m_b).abs() <= 1]), ("minutes down >3", P[P.m_w - P.m_b < -3])]:
    print(name, len(sub), "mean target", round(sub.target.mean(), 2), "mean d", round(sub.d.mean(), 2), "-> persistence", round(sub.target.mean() / sub.d.mean(), 2))
# does absence-driven minutes gain persist less?
hi = P[(P.dabs > 10) & ((P.m_w - P.m_b) > 2)]
lo = P[(P.dabs < 3) & ((P.m_w - P.m_b) > 2)]
for nm, sub in (("minutes up WITH teammates out", hi), ("minutes up, teammates healthy", lo)):
    print(nm, len(sub), "mean minute gain", round((sub.m_w - sub.m_b).mean(), 2), "-> next-12 minutes gain vs baseline", round((sub.m_f - sub.m_b).mean(), 2), " fp d", round(sub.d.mean(), 2), "target", round(sub.target.mean(), 2))
