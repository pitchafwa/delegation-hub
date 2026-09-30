"""Does knowing a teammate will specifically be INSERTED INTO THE STARTING LINEUP (not just get more bench run) add real information beyond
what the live usage-flow model already uses, on nights a rotation player is out? (Tommy, 2026-09-30, following up on start_bench_study_v2.py's
finding that a beneficiary who starts still shows a real fp gap over one who doesn't, even within teammate-out games.)

The live model (usage_flow.py) distributes vacated production by `share_j` (a fp^gamma-weighted share) and `tier_of(mpg)` (his OWN minutes
tier) -- it never looks at whether HE specifically starts tonight. This tests whether `start` explains real residual variance beyond `w`/`wm`
(the panel's own share-of-vacated-production/minutes fields, the same concept the live model's share_j is built from) and `b_min` (a tier proxy).

Held out by season. Usage: uv run python research/usage_flow_start_refinement_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"

d = pd.read_pickle(D / "usage_flow_panel.pkl")
d["start"] = d["start"].astype(bool)
d = d[(d.V_min > 0) & d.b_fp.notna() & d.w.notna() & d.wm.notna()].copy()   # teammate-out games only
d["dev"] = d.fp - d.b_fp
print(f"{len(d):,} teammate-out player-games with a real baseline and share weights")

BASE_COLS = ["w", "wm", "b_min"]
FULL_COLS = BASE_COLS + ["start"]


def loso(cols):
    errs = []
    for s in sorted(d.season_year.unique()):
        tr, te = d[d.season_year != s], d[d.season_year == s]
        if len(te) < 200:
            continue
        mm = LinearRegression().fit(tr[cols].astype(float).values, tr.dev.values)
        pred = mm.predict(te[cols].astype(float).values)
        errs.append(te.dev.values - pred)
    e = np.concatenate(errs)
    return np.sqrt(np.mean(e ** 2))


rb = loso(BASE_COLS)
rf = loso(FULL_COLS)
print(f"held-out RMSE, current model's inputs (w, wm, b_min) only: {rb:.4f}")
print(f"held-out RMSE, + whether he actually started tonight:      {rf:.4f}")

m_full = LinearRegression().fit(d[FULL_COLS].astype(float).values, d.dev.values)
print(f"\nfull-sample coefficients: {dict(zip(FULL_COLS, m_full.coef_.round(3)))}")
print(f"'start' coefficient: {m_full.coef_[-1]:.2f} fp/g -- this is the extra bump from actually starting, holding share/tier fixed")

print("\n=== sanity: within similar share-of-vacated-production, does starting still separate outcomes? ===")
d["share_bucket"] = pd.qcut(d.w, 4, duplicates="drop")
print(d.groupby(["share_bucket", "start"], observed=True).dev.agg(["mean", "count"]))
