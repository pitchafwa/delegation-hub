"""Does an elite pre-NBA profile predict LATER (year 3-5) fantasy output beyond draft slot?
Our rookie model only tested year-1 relevance; dynasty value is about ceilings years out.
Sample: past top-10 picks (classes 2010-2021) with a year-3..5 season.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
D = Path(__file__).resolve().parent / "data"

u = pd.read_csv(D / "rookie_model_dataset_unified.csv")
u = u[(u["real_draft_year"].between(2010, 2021)) & (u["real_draft_number"] <= 10) & (u["data_source"] == "college")].copy()
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
             - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
act = sb[sb["GP"] >= 20].set_index(["PLAYER_ID", "yr"])["fpg"]
peak = []
for r in u.itertuples():
    v = [act.get((r.PLAYER_ID, int(r.real_draft_year) + k), np.nan) for k in (2, 3, 4)]
    peak.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
u["f345"] = peak
u["lp"] = np.log(u["real_draft_number"])
u = u.dropna(subset=["f345", "talent_pctile"])
print(f"top-10 college picks 2010-2021 with years 3-5 data: n={len(u)}")
print(f"correlation with avg pts/g in years 3-5:  ln(pick) {u['lp'].corr(u['f345']):+.2f} | talent pctile {u['talent_pctile'].corr(u['f345']):+.2f} | BPM {u['bpm'].corr(u['f345']):+.2f} | draft age {u['draft_age'].corr(u['f345']):+.2f}")
X = np.column_stack([np.ones(len(u)), u["lp"], u["talent_pctile"]])
beta, res, *_ = np.linalg.lstsq(X, u["f345"].to_numpy(), rcond=None)
resid = u["f345"].to_numpy() - X @ beta
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=3))
print(f"regression f345 ~ ln(pick) + talent_pctile: talent coef {beta[2]:+.1f} pts/g per full percentile range (t = {beta[2]/se[2]:+.1f}); pick coef {beta[1]:+.1f} (t = {beta[1]/se[1]:+.1f})")
top = u[u["real_draft_number"] <= 5]
Xt = np.column_stack([np.ones(len(top)), top["lp"], top["talent_pctile"]])
bt = np.linalg.lstsq(Xt, top["f345"].to_numpy(), rcond=None)[0]
print(f"within picks 1-5 only (n={len(top)}): talent coef {bt[2]:+.1f} pts/g")
u["elite"] = u["talent_pctile"] >= 0.9
for name, g in u.groupby("elite"):
    print(f"  talent pctile {'>=90' if name else '<90 '}: n={len(g):3d}  mean pick {g['real_draft_number'].mean():4.1f}  avg pts/g yrs 3-5: {g['f345'].mean():5.1f}  share >=45: {(g['f345']>=45).mean():.0%}  share >=55: {(g['f345']>=55).mean():.0%}")
print("\nPicks 1-5 by talent tier:")
for lo, hi, lab in [(0.0, 0.8, "<80"), (0.8, 0.95, "80-95"), (0.95, 1.01, "95+")]:
    g = top[(top["talent_pctile"] >= lo) & (top["talent_pctile"] < hi)]
    if len(g):
        print(f"  talent {lab:>5s}: n={len(g):2d} avg pts/g yrs 3-5 {g['f345'].mean():5.1f}  ({', '.join(g.sort_values('f345', ascending=False)['player'].head(4))}...)")
