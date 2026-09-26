"""Build the form-split panel: for each player, checkpoints every STRIDE games; window = last W games played, baseline = his games in the 365 days before the window, outcome = next H games (same season).
Writes data/form/panel.csv."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
import form_common as F

W, H, STRIDE = 10, 12, 5
g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
if "absent" not in g:
    g["absent"] = F.add_absence(g)
rows = []
for pid, pl in g.groupby("pid", sort=False):
    pl = pl.reset_index(drop=True)
    n = len(pl)
    if n < W + H + 30:
        continue
    dates = pl["date"].values
    seas = pl["season"].values
    for e in range(W + 10, n - H + 1, STRIDE):
        s0 = e - W
        if seas[s0] != seas[e - 1] or seas[e - 1] != seas[e + H - 1]:
            continue
        lo = np.searchsorted(dates, dates[s0] - np.timedelta64(365, "D"))
        b = pl.iloc[lo:s0]
        if len(b) < 30:
            continue
        w = pl.iloc[s0:e]
        f = pl.iloc[e:e + H]
        if (dates[e + H - 1] - dates[e]) / np.timedelta64(1, "D") > 60:
            continue
        if b["min"].mean() < 12 or w["min"].mean() < 8:
            continue
        r = F.split(w, b)
        r.update(pid=pid, name=pl["name"].iloc[0], season=seas[e - 1], end=str(dates[e - 1])[:10], fp_f=f["fp"].mean(), m_f=f["min"].mean(), n_b=len(b),
                 abs_w=w["absent"].mean(), abs_b=b["absent"].mean(), abs_f=f["absent"].mean(), fp_l3=pl["fp"].iloc[e - 3:e].mean(), m_l3=pl["min"].iloc[e - 3:e].mean(),
                 age_days=0)
        rows.append(r)
P = pd.DataFrame(rows)
import os
P.to_csv(F.D.parent / "form" / ("panel.csv" if not os.environ.get("FORM_K_MULT") else f"panel_k{os.environ['FORM_K_MULT']}.csv"), index=False)
print(P.shape, P.season.nunique())
print(P[["fp_b", "fp_w", "fp_f", "minutes", "luck", "volume", "other", "abs_w", "abs_b"]].describe().round(2))
