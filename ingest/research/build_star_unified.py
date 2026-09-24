"""Star calibration for the UNIFIED (college + international) model.
Diamond stays college-only for now (its indicator set -- porpag, fg_pct,
usg, power_conf, etc -- is heavily college-specific and international
equivalents weren't pulled in this first pass; a disclosed scope limit,
not a bug). Star only needs the unified pre/post scores themselves, so it
extends cleanly.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from fit_rookie_model_unified import df as full_df, score, capital_curve

ROOT = Path(__file__).resolve().parent

HIT_BAR = 28.0
df = full_df.copy()
df["hit"] = (df["age_22_29_best3"] >= HIT_BAR).astype(int)
print(f"Hit bar: age_22_29_best3 >= {HIT_BAR} -> base hit rate = {df['hit'].mean():.3f} ({df['hit'].sum()}/{len(df)})")
print(f"  college: {df[df['data_source']=='college']['hit'].mean():.3f}, "
      f"international: {df[df['data_source']=='international']['hit'].mean():.3f}")

frozen = json.loads((ROOT / "data" / "output_a_model_unified.frozen.json").read_text(encoding="utf-8"))
PRE = frozen["pre_draft_model"]
POST = frozen["post_draft_model"]

pre_score = score(PRE["params"], df, PRE["features"])
component_ranks = []
for name, cfg in POST["components"].items():
    n_pre = cfg["n_pre_draft_params"]
    p = cfg["params"]
    pre_part = score(p[:n_pre], df, cfg["features"])
    cap_part = capital_curve(p[n_pre:], df["pick_filled"].to_numpy(dtype=float))
    component_ranks.append(rankdata(pre_part + cap_part))
post_score = np.mean(component_ranks, axis=0)

df["pre_draft_score"] = pre_score
df["post_draft_score"] = post_score


def to_percentile(x, ref):
    ref_sorted = np.sort(ref)
    return np.searchsorted(ref_sorted, x, side="right") / len(ref_sorted)


df["pre_draft_pctl"] = to_percentile(df["pre_draft_score"].to_numpy(), df["pre_draft_score"].to_numpy())
df["post_draft_pctl"] = to_percentile(df["post_draft_score"].to_numpy(), df["post_draft_score"].to_numpy())


def decile_table(pctl_col):
    d = df.copy()
    d["_t"] = np.ceil(d[pctl_col] * 10).clip(1, 10).astype(int)
    g = d.groupby("_t").agg(n=("hit", "size"), hit_rate=("hit", "mean")).reset_index()
    return g


print("\n=== PRE-DRAFT decile hit-rate table (unified) ===")
print(decile_table("pre_draft_pctl").to_string(index=False))
print("\n=== POST-DRAFT decile hit-rate table (unified) ===")
print(decile_table("post_draft_pctl").to_string(index=False))

STAR_PCTL = 0.90
df["is_star_pre"] = (df["pre_draft_pctl"] >= STAR_PCTL).astype(int)
df["is_star_post"] = (df["post_draft_pctl"] >= STAR_PCTL).astype(int)
print(f"\nis_star_pre hit rate: {df[df['is_star_pre']==1]['hit'].mean():.3f} (n={df['is_star_pre'].sum()})")
print(f"is_star_post hit rate: {df[df['is_star_post']==1]['hit'].mean():.3f} (n={df['is_star_post'].sum()})")

frozen["star_calibration"] = {"hit_bar_fantasy_ppg": HIT_BAR, "star_pctl": STAR_PCTL,
    "base_hit_rate": float(df["hit"].mean())}
(ROOT / "data" / "output_a_model_unified.frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
print("\nSaved star calibration to output_a_model_unified.frozen.json")
