"""Does our EXISTING frozen model (learned from 2008-2018 outcomes) still
predict well for recent classes, or has the relationship between pre-draft
indicators and NBA success drifted as the league has changed? Real test,
not a guess: score every player with the already-frozen model (no
refitting), then check real rank correlation against a FAST-resolving
near-term outcome (best 2-of-3 seasons right after draft) across eras --
including classes way too recent for the long-term age_22_29_best3 target
to ever resolve (2019-2023).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
target = pd.read_csv(ROOT / "data" / "target_window_test.csv")
df = df.merge(target[["PLAYER_ID", "entry_3yr_best2"]], on="PLAYER_ID", how="left")

EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["LANE_AGILITY_TIME_PCTILE"] = df["LANE_AGILITY_TIME_PCTILE"].fillna(df["LANE_AGILITY_TIME_PCTILE"].median())
df["breakout_age_filled"] = df["breakout_age"].fillna(df["breakout_age"].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["talent_pctile"] = df["talent_pctile"].fillna(df["talent_pctile"].median())

frozen = json.loads((ROOT / "data" / "output_a_model_unified.frozen.json").read_text(encoding="utf-8"))
PRE = frozen["pre_draft_model"]
POST = frozen["post_draft_model"]


def score(params, data, feature_list):
    idx = 0
    total = np.zeros(len(data))
    for f in feature_list:
        lo_x, span, weight = params[idx], params[idx + 1], params[idx + 2]
        idx += 3
        hi_x = lo_x + span
        x = data[f].to_numpy(dtype=float)
        frac = np.clip((x - lo_x) / max(hi_x - lo_x, 1e-6), 0.0, 1.0)
        total += weight * frac
    total += params[idx]
    return total


def capital_curve(params, pick):
    k, c, p, floor = params
    return floor + k * (pick + c) ** (-p)


df["pre_draft_score"] = score(PRE["params"], df, PRE["features"])

component_ranks = []
for name, cfg in POST["components"].items():
    n_pre = cfg["n_pre_draft_params"]
    p = cfg["params"]
    pre_part = score(p[:n_pre], df, cfg["features"])
    cap_part = capital_curve(p[n_pre:], df["pick_filled"].to_numpy(dtype=float))
    component_ranks.append(rankdata(pre_part + cap_part))
df["post_draft_score"] = np.mean(component_ranks, axis=0)

resolved = df.dropna(subset=["entry_3yr_best2"])
resolved = resolved[resolved["real_draft_year"] >= 2008]
print(f"Real population with resolved entry_3yr_best2: {len(resolved)}")
print(f"Draft year range: {int(resolved['real_draft_year'].min())}-{int(resolved['real_draft_year'].max())}")

ERAS = [("2008-2013 (old)", 2008, 2013), ("2014-2018 (mid)", 2014, 2018),
        ("2019-2023 (recent, unreachable by age_22_29_best3)", 2019, 2023)]

print("\n=== Real correlation: EXISTING frozen model's score vs near-term outcome, by era ===")
print(f"{'era':55s} {'n':>5s} {'post-draft rho':>15s} {'pre-draft rho':>14s} {'pick-alone rho':>15s}")
for label, lo, hi in ERAS:
    sub = resolved[(resolved["real_draft_year"] >= lo) & (resolved["real_draft_year"] <= hi)]
    if len(sub) < 10:
        print(f"{label:55s} {len(sub):5d}  (too few)")
        continue
    r_post, _ = spearmanr(sub["post_draft_score"], sub["entry_3yr_best2"])
    r_pre, _ = spearmanr(sub["pre_draft_score"], sub["entry_3yr_best2"])
    r_pick, _ = spearmanr(-sub["pick_filled"], sub["entry_3yr_best2"])
    print(f"{label:55s} {len(sub):5d} {r_post:15.4f} {r_pre:14.4f} {r_pick:15.4f}")

print("\n=== Same thing, by real draft YEAR (finer-grained trend) ===")
for y in sorted(resolved["real_draft_year"].unique()):
    sub = resolved[resolved["real_draft_year"] == y]
    if len(sub) < 15:
        continue
    r_post, _ = spearmanr(sub["post_draft_score"], sub["entry_3yr_best2"])
    r_pick, _ = spearmanr(-sub["pick_filled"], sub["entry_3yr_best2"])
    print(f"{int(y)}: n={len(sub):3d}  post-draft rho={r_post:.3f}  pick-alone rho={r_pick:.3f}")
