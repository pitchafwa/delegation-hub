"""Cross-check: for CURRENT young veterans, are the hub's Kalman trajectories lower than what an
empirical (history-trained) forecast says for the same players? Uses ridge models on the breakout
panel features for fantasy pts/g 1, 2 and 3 seasons ahead, and compares by age group.
"""
import json
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
D = ROOT / "data"

p = pd.read_csv(D / "breakout_panel_ctx2.csv")
p["young"] = (p["AGE"] <= 24) * (24 - p["AGE"])
for h in (2, 3):
    s = p[["PLAYER_ID", "yr", "fpg"]].copy(); s["yr"] -= h
    p = p.merge(s.rename(columns={"fpg": f"fpg_next{h}"}), on=["PLAYER_ID", "yr"], how="left")
p["fpg_next1"] = p["fpg_next"]
F = ["fpg", "AGE", "young", "mpg", "late_mpg", "late_fpg", "late_fp_per36", "USG_PCT", "PIE", "draft_pick", "exp", "d_fpg", "d_mpg"]
pool = p[(p["GP"] >= 20) & (p["mpg"] >= 10)]
live = pool[pool["yr"] == pool["yr"].max()].copy()

hub = {q["id"]: q for q in json.loads((ROOT.parent.parent / "dashboard" / "hub_data.json").read_text(encoding="utf-8"))["players"]}
rows = []
for h in (1, 2, 3):
    tr = pool[pool["yr"] < pool["yr"].max() - h + 1]
    tr = tr[tr[f"fpg_next{h}"].notna() & ((tr["GP_next"] >= 20) if h == 1 else True)]
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(tr[F], tr[f"fpg_next{h}"])
    live[f"emp{h}"] = m.predict(live[F])
live["kalman1"] = [hub.get(f"c{i}", {}).get("trajectory", [np.nan] * 7)[0] for i in live["PLAYER_ID"]]
live["kalman2"] = [hub.get(f"c{i}", {}).get("trajectory", [np.nan] * 7)[1] for i in live["PLAYER_ID"]]
live["kalman3"] = [hub.get(f"c{i}", {}).get("trajectory", [np.nan] * 7)[2] for i in live["PLAYER_ID"]]
live = live.dropna(subset=["kalman1"])
live["grp"] = pd.cut(live["AGE"] + 1, [17, 21.5, 23.5, 26.5, 30.5, 45], labels=["<=21", "22-23", "24-26", "27-30", "31+"])
print("mean (empirical forecast - hub Kalman trajectory), pts/g, current players (pool: >=20 GP, >=10 mpg)")
print(f"{'age next yr':>12s} {'n':>4s} {'+1yr':>7s} {'+2yr':>7s} {'+3yr':>7s}")
for g, s in live.groupby("grp"):
    print(f"{g:>12s} {len(s):4d} {(s['emp1']-s['kalman1']).mean():+7.1f} {(s['emp2']-s['kalman2']).mean():+7.1f} {(s['emp3']-s['kalman3']).mean():+7.1f}")
