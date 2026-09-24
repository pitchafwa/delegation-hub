"""Prospect engines, tested leave-one-draft-class-out (nothing predicts a class it was fit on):
   A_raw  = Output B rookie stat line (refit without the class) aged forward with the Kalman aging curves
   A_cal  = A_raw + pick-dependent ceiling calibration (fit without the class)
   B      = empirical ridge on draft slot + draft age + pre-NBA talent percentile (fit without the class)
   blends of A_cal and B.
Target: real fantasy pts/g in seasons k=0..4 after the draft (>=20 GP), picks 1-15 (where the calibration applies)
and picks 16-30.
"""
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from project_prospect_trajectory import build_prospect_trajectory  # noqa: E402

D = ROOT / "data"
df = pd.read_csv(D / "rookie_model_dataset_unified.csv")
df = df[(df["real_draft_year"] >= 2010) & (df["data_source"] == "college")].copy()
EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
FEATURES = ["talent_pctile", "rec_filled", "exp_numeric", "draft_age_filled", "breakout_age_filled", "porpag", "usg", "ts", "ortg", "obpm",
            "dbpm", "bpm", "stops", "oreb_rate", "dreb_rate", "ast_to", "ftr", "pfr", "WINGSPAN_PCTILE", "STANDING_REACH_PCTILE",
            "STANDING_VERTICAL_LEAP_PCTILE", "MAX_VERTICAL_LEAP_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE",
            "three_pct", "rim_pct", "mid_pct", "pick_filled"]
FEATURES = [f for f in FEATURES if f in df.columns]
for f in FEATURES:
    df[f] = pd.to_numeric(df[f], errors="coerce")
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["rookie_STL_per_min"] = sb["STL"] / sb["MIN"].replace(0, np.nan)
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
             - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
rk = sb[sb["yr"] == sb["yr"]][["PLAYER_ID", "yr", "rookie_STL_per_min"]].rename(columns={"yr": "real_draft_year"})
df = df.merge(rk, on=["PLAYER_ID", "real_draft_year"], how="left")
STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "STL"]
TGT = {s: f"rookie_{s}_per_min" for s in STATS}
TGT["MIN"] = "rookie_MIN_per_game"
WIN = {"PTS": "ridge", "REB": "ridge", "AST": "ridge", "BLK": "ridge", "TOV": "rf", "FG3M": "ridge", "FTM": "rf", "FTA": "rf", "STL": "ridge", "MIN": "rf"}
ALPHA = {"PTS": 100.0}
tr_all = df.dropna(subset=["rookie_MIN_per_game"]).copy()
med = tr_all[FEATURES].median()
Xi = tr_all[FEATURES].fillna(med).to_numpy(float)
years = tr_all["real_draft_year"].to_numpy()

# ---- OOF Output B stat lines -> OOF trajectories
oof = {s: np.full(len(tr_all), np.nan) for s in TGT}
for held in sorted(set(years)):
    te, trm = years == held, years != held
    if te.sum() < 3:
        continue
    for s, col in TGT.items():
        y = tr_all[col].to_numpy(float)
        ok = trm & ~np.isnan(y)
        if WIN[s] == "ridge":
            sc = StandardScaler().fit(Xi[ok])
            m = Ridge(alpha=ALPHA.get(s, 10.0), random_state=42).fit(sc.transform(Xi[ok]), y[ok])
            oof[s][te] = m.predict(sc.transform(Xi[te]))
        else:
            m = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=1).fit(Xi[ok], y[ok])
            oof[s][te] = m.predict(Xi[te])
trajs = []
for i in range(len(tr_all)):
    proj = {s: max(0.0, float(oof[s][i])) for s in TGT}
    trajs.append(build_prospect_trajectory(proj, float(tr_all["draft_age_filled"].iloc[i])))
tr_all["traj"] = trajs

# ---- actuals by season since draft
act = sb[sb["GP"] >= 20].set_index(["PLAYER_ID", "yr"])["fpg"]
rows = []
for r in tr_all.itertuples():
    for k in range(5):
        a = act.get((r.PLAYER_ID, int(r.real_draft_year) + k), np.nan)
        if np.isfinite(a):
            rows.append(dict(pid=r.PLAYER_ID, cls=int(r.real_draft_year), pick=r.pick_filled, dage=r.draft_age_filled, talent=r.talent_pctile,
                             k=k, y=a, A_raw=r.traj[k]))
E = pd.DataFrame(rows)
E["talent"] = E["talent"].fillna(E["talent"].median())
E["lp"] = np.log(E["pick"].clip(1, 61))

# ---- LOCO: ceiling calibration (A_cal) and empirical ridge (B)
E["A_cal"], E["B"] = np.nan, np.nan
for cls in sorted(E["cls"].unique()):
    te = E["cls"] == cls
    for k in range(5):
        trk = E[(~te) & (E["k"] == k)]
        top = trk[trk["pick"] <= 15]
        Aa = np.column_stack([np.ones(len(top)), top["lp"]])
        a, b = np.linalg.lstsq(Aa, (top["y"] - top["A_raw"]).to_numpy(), rcond=None)[0]
        sel = te & (E["k"] == k)
        if not sel.any() or len(top) < 10:
            continue
        taper = np.clip((16 - E.loc[sel, "pick"]) / 6.0, 0, 1)
        E.loc[sel, "A_cal"] = E.loc[sel, "A_raw"] + taper * (a + b * E.loc[sel, "lp"])
        sc = StandardScaler().fit(trk[["lp", "dage", "talent"]])
        m = Ridge(alpha=10).fit(sc.transform(trk[["lp", "dage", "talent"]]), trk["y"])
        E.loc[sel, "B"] = m.predict(sc.transform(E.loc[sel, ["lp", "dage", "talent"]]))
rm = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
for label, sub in [("PICKS 1-15", E[E["pick"] <= 15]), ("PICKS 16-30", E[(E["pick"] > 15) & (E["pick"] <= 30)])]:
    print(f"\n{label}  (RMSE, pts/g, held-out draft classes)  [w = weight on the empirical model B in a blend with A_cal]")
    print(f"{'k':>2s} {'n':>4s} {'A_raw':>7s} {'A_cal':>7s} {'B':>7s} {'w=.25':>7s} {'w=.5':>7s} {'w=.75':>7s} | bias A_raw  A_cal   B")
    for k in range(5):
        t = sub[sub["k"] == k]
        print(f"{k:2d} {len(t):4d} {rm(t['y'], t['A_raw']):7.2f} {rm(t['y'], t['A_cal']):7.2f} {rm(t['y'], t['B']):7.2f} "
              + " ".join(f"{rm(t['y'], (1 - w) * t['A_cal'] + w * t['B']):7.2f}" for w in (0.25, 0.5, 0.75))
              + f" | {(t['y']-t['A_raw']).mean():+6.2f} {(t['y']-t['A_cal']).mean():+6.2f} {(t['y']-t['B']).mean():+6.2f}")
E.to_pickle(D / "_prospect_engine_rows.pkl")
