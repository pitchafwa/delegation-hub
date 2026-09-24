"""Exhaustive interaction screen: two features can each look weak alone yet
combine into a real signal (e.g. age only matters AT a certain talent
level). Rather than DE-fitting every pairwise product (too slow to be
practical), screen cheaply first: take the RESIDUAL of the target after the
current champion model's prediction, then rank every candidate pairwise
interaction by how well it correlates with that leftover variance. Only the
top real candidates get a full LOCO-CV DE-fit confirmation afterward.
"""
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from fit_rookie_model_a2 import df, ALL_FEATURE_BOUNDS, score

ROOT_DATA = json.loads(open("data/output_a_model.frozen.json").read())
POST = ROOT_DATA["post_draft_model"]
FEATURES = POST["features"]
PARAMS = POST["params"]
N_PRE = POST["n_pre_draft_params"]


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


pre_part = score(PARAMS[:N_PRE], df, FEATURES)
cap_part = capital_curve(PARAMS[N_PRE:], df["real_draft_number"].to_numpy(dtype=float))
champion_pred = pre_part + cap_part

# work in rank space (Spearman-consistent) -- residual after removing the
# champion's own rank-predictive power
target_rank = rankdata(df["age_22_29_best3"])
pred_rank = rankdata(champion_pred)
# residualize target_rank on pred_rank via simple linear regression in rank space
slope, intercept = np.polyfit(pred_rank, target_rank, 1)
residual = target_rank - (slope * pred_rank + intercept)

candidates = [f for f in ALL_FEATURE_BOUNDS.keys() if f != "age_bpm_interaction"]
# add breakout/never_broke_out/porpag/ts/ortg already in ALL_FEATURE_BOUNDS via a2 import
results = []
for a, b in itertools.combinations(candidates, 2):
    xa = pd.Series(ALL_FEATURE_BOUNDS[a][0]).to_numpy(dtype=float)
    xb = pd.Series(ALL_FEATURE_BOUNDS[b][0]).to_numpy(dtype=float)
    za = (xa - np.nanmean(xa)) / (np.nanstd(xa) + 1e-9)
    zb = (xb - np.nanmean(xb)) / (np.nanstd(xb) + 1e-9)
    interaction = za * zb
    if np.nanstd(interaction) < 1e-9:
        continue
    rho, _ = spearmanr(interaction, residual, nan_policy="omit")
    if np.isfinite(rho):
        results.append((a, b, rho))

results.sort(key=lambda r: -abs(r[2]))
print("Top 20 candidate pairwise interactions vs champion-model RESIDUAL (leftover signal):")
for a, b, rho in results[:20]:
    print(f"  {a:28s} x {b:28s}  rho={rho:+.4f}")

print(f"\nTotal pairs screened: {len(results)}")
