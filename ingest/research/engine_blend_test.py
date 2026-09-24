"""Two projection engines for players WITH NBA data:
   A = Kalman filter state + (corrected) aging path        (what the VOR / trajectory column uses)
   B = empirical ridge forecast from last-season production, minutes, usage, age, pedigree (what Asset value used)
Which is more accurate on held-out seasons, and what blend is best? Same players, same targets, same grading set.
Kalman side: STL = career rate to date, TD3 = last season's rate (no peeking at the target season).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.argv = ["x"]
src = open(ROOT / "fit_aging_shape.py", encoding="utf-8").read()
exec(src.split("# ---- validation: fit only on targets <= 2019, test on later targets")[0])  # R, hat, fit_shape, old_slope, ...

TRAIN_MAXT = 2019
coefs_val = {s: fit_shape(R[R["origin"] + R["h"] <= TRAIN_MAXT], s) for s in KSTATS}
test = R[(R["origin"] + R["h"] > TRAIN_MAXT) & (R["origin"] >= 2016)].copy()


def mixed_path(df_, coefs, tmax):
    out = np.zeros(len(df_))
    for stat in df_["stat"].unique():
        for h in df_["h"].unique():
            m = ((df_["stat"] == stat) & (df_["h"] == h)).to_numpy()
            if not m.any():
                continue
            a0, gp = df_.loc[m, "age0"].to_numpy(), df_.loc[m, "gap"].to_numpy()

            def u(a):
                new, old = hat(a) @ coefs[stat], old_slope(stat, a)
                return old + tmax * np.clip((a - 28.0) / 3.0, 0, 1) * (new - old)

            tot = gp * u(a0)
            for k in range(0, int(h) - 1):
                tot = tot + u(a0 + k)
            out[m] = tot
    return out


test["pred"] = (test["b"] + mixed_path(test, coefs_val, 0.6)).clip(lower=0)
wide = test.pivot_table(index=["origin", "h", "PLAYER_ID"], columns="stat", values="pred").dropna()

# STL career rate and TD3 last-season rate as known at each origin
g = df.groupby(["PLAYER_ID", "sy"]).agg(stl=("STL", "sum"), mins=("MIN", "sum"), td3=("TD3", "sum"), gp=("MIN", "size")).reset_index().sort_values(["PLAYER_ID", "sy"])
g["cstl"] = g.groupby("PLAYER_ID")["stl"].cumsum() / g.groupby("PLAYER_ID")["mins"].cumsum()
g["td3pg"] = g["td3"] / g["gp"]
gk = g.set_index(["PLAYER_ID", "sy"])
idx = pd.MultiIndex.from_arrays([wide.index.get_level_values("PLAYER_ID"), wide.index.get_level_values("origin")])
stl = gk["cstl"].reindex(idx).to_numpy()
td3 = gk["td3pg"].reindex(idx).to_numpy()
m_ = wide["MIN"]
kal = (wide["PTS"] * m_ + 1.5 * wide["REB"] * m_ + 2 * wide["AST"] * m_ + 3 * stl * m_ + 3 * wide["BLK"] * m_ + wide["FG3M"] * m_
       + 2 * wide["FTM"] * m_ - wide["FTA"] * m_ - wide["TOV"] * m_ + 3 * td3)
K = kal.rename("A_kalman").reset_index()

# ---- panel: actuals and ridge engine B
panel = pd.read_csv(ROOT / "data" / "breakout_panel_ctx2.csv")
panel["young"] = (panel["AGE"] <= 24) * (24 - panel["AGE"])
panel["logpick"] = np.log(panel["draft_pick"].clip(1, 61))
F = ["fpg", "AGE", "young", "mpg", "late_mpg", "late_fpg", "late_fp_per36", "USG_PCT", "PIE", "logpick", "exp", "d_fpg", "d_mpg"]
pk = panel.set_index(["PLAYER_ID", "yr"])
for h in (1, 2, 3, 4):
    idx2 = pd.MultiIndex.from_arrays([panel["PLAYER_ID"], panel["yr"] + h])
    f, gp = pk["fpg"].reindex(idx2).to_numpy(), pk["GP"].reindex(idx2).to_numpy()
    panel[f"v{h}"] = np.where(gp >= 20, f, np.nan)
POOL = panel[(panel["GP"] >= 20) & (panel["mpg"] >= 10)]
recs = []
for (oy, h), grp in K.groupby(["origin", "h"]):
    tr = POOL[(POOL["yr"] < oy) & (POOL["yr"] + h <= oy) & POOL[f"v{h}"].notna()]
    te = POOL[POOL["yr"] == oy].set_index("PLAYER_ID")
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(tr[F], tr[f"v{h}"])
    common = grp["PLAYER_ID"][grp["PLAYER_ID"].isin(te.index)]
    tt = te.loc[common.to_numpy()]
    recs.append(pd.DataFrame({"origin": oy, "h": h, "PLAYER_ID": common.to_numpy(), "B_ridge": m.predict(tt[F]), "y": tt[f"v{h}"].to_numpy(),
                              "age": tt["AGE"].to_numpy() + 1, "fpg0": tt["fpg"].to_numpy(), "d_fpg": tt["d_fpg"].to_numpy()}))
E = pd.concat(recs).merge(K, on=["origin", "h", "PLAYER_ID"]).dropna(subset=["y", "A_kalman", "B_ridge"])
E["exp_rel"] = (E["A_kalman"] + E["B_ridge"]) / 2 >= 20  # known at the origin
E = E[E["exp_rel"]]
print(f"held-out rows: {len(E)} (origins 2016-2024, horizons 1-4, players both engines rate as fantasy-relevant)\n")
rm = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
W = [0, 0.25, 0.5, 0.75, 1.0]
print("RMSE by horizon:   w = weight on ridge (B);  w=0 is pure Kalman (A), w=1 pure ridge (B);  persistence = last season's pts/g")
print(f"{'h':>2s} {'n':>5s} " + " ".join(f"w={w:<5}" for w in W) + "  persistence")
best = {}
for h, t in E.groupby("h"):
    r = [rm(t["y"], (1 - w) * t["A_kalman"] + w * t["B_ridge"]) for w in W]
    best[int(h)] = W[int(np.argmin(r))]
    print(f"{h:2d} {len(t):5d} " + " ".join(f"{x:6.2f} " for x in r) + f"  {rm(t['y'], t['fpg0']):6.2f}")
print("\nby age group (all horizons pooled), RMSE and bias(actual - forecast):")
E["grp"] = pd.cut(E["age"], [17, 21.5, 23.5, 26.5, 30.5, 45], labels=["<=21", "22-23", "24-26", "27-30", "31+"])
print(f"{'age':>6s} {'n':>5s} {'A rmse':>7s} {'B rmse':>7s} {'50/50':>7s} | {'A bias':>7s} {'B bias':>7s}")
for gname, t in E.groupby("grp"):
    print(f"{str(gname):>6s} {len(t):5d} {rm(t['y'], t['A_kalman']):7.2f} {rm(t['y'], t['B_ridge']):7.2f} {rm(t['y'], (t['A_kalman']+t['B_ridge'])/2):7.2f} | "
          f"{(t['y']-t['A_kalman']).mean():+7.2f} {(t['y']-t['B_ridge']).mean():+7.2f}")
J = E[E["d_fpg"] >= 10]
print(f"\nbreakout-jump players (last season +10 pts/g or more vs the year before; the Rollins case), n={len(J)}:")
for h, t in J.groupby("h"):
    print(f"  h={h}: n={len(t):3d}  Kalman rmse {rm(t['y'], t['A_kalman']):5.2f} (bias {(t['y']-t['A_kalman']).mean():+5.2f}) | ridge rmse {rm(t['y'], t['B_ridge']):5.2f} (bias {(t['y']-t['B_ridge']).mean():+5.2f}) | 50/50 {rm(t['y'], (t['A_kalman']+t['B_ridge'])/2):5.2f}")
E.to_pickle(ROOT / "data" / "_engine_blend_rows.pkl")
