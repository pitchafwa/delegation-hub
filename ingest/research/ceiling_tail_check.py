"""Is the model's UPPER TAIL for top prospects too thin?

For every past top pick (held-out draft class), build the model's predictive distribution for pts/g in season k after
the draft (empirical-ridge mean + the residual spread of same-tier picks from OTHER classes) and record where the
actual result fell (PIT = fraction of the predictive distribution below the actual). A well-calibrated model gives
PITs spread evenly 0-1: ~25% above 0.75, ~10% above 0.90. If top picks pile up near 1, the model's ceiling is too low.
Also: how much of a top prospect's Asset value comes from the spread, and what do the real peak seasons look like.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
E = pd.read_pickle(D / "_prospect_engine_rows.pkl").dropna(subset=["B"]).copy()
E["res"] = E["y"] - E["B"]
E["tier"] = pd.cut(E["pick"], [0, 3.5, 10.5, 30.5], labels=["1-3", "4-10", "11-30"])
rows = []
for cls in sorted(E["cls"].unique()):
    for k in range(5):
        te = E[(E["cls"] == cls) & (E["k"] == k)]
        for r in te.itertuples():
            pool = E[(E["cls"] != cls) & (E["k"] == k) & (E["tier"] == r.tier)]["res"].to_numpy()
            if len(pool) < 15:
                continue
            pred = r.B + pool
            rows.append(dict(tier=str(r.tier), k=k, pit=float((pred <= r.y).mean()), y=r.y, mean=r.B))
P = pd.DataFrame(rows)
print("PIT calibration of the model's predictive distribution (held-out classes). Ideal: mean 0.50, >0.75: 25%, >0.90: 10%")
print(f"{'tier':>6s} {'k':>2s} {'n':>4s} {'mean PIT':>9s} {'>0.75':>7s} {'>0.90':>7s} {'<0.10':>7s}")
for tier in ["1-3", "4-10", "11-30"]:
    for k in range(5):
        t = P[(P["tier"] == tier) & (P["k"] == k)]
        print(f"{tier:>6s} {k:2d} {len(t):4d} {t['pit'].mean():9.2f} {(t['pit'] > .75).mean():7.0%} {(t['pit'] > .90).mean():7.0%} {(t['pit'] < .10).mean():7.0%}")
    t = P[P["tier"] == tier]
    print(f"{tier:>6s} all {len(t):4d} {t['pit'].mean():9.2f} {(t['pit'] > .75).mean():7.0%} {(t['pit'] > .90).mean():7.0%} {(t['pit'] < .10).mean():7.0%}")

# real peak seasons (best of first 5 seasons) for picks 1-3 vs 4-10, i.e. the empirical ceiling distribution
peak = E.groupby(["pid", "tier"])["y"].max().reset_index()
print("\nREAL best-of-first-5-seasons pts/g by draft tier (players with >=1 qualifying season):")
for tier in ["1-3", "4-10", "11-30"]:
    y = peak[peak["tier"] == tier]["y"]
    print(f"  picks {tier:>5s}: n={len(y):3d}  p25 {y.quantile(.25):5.1f}  median {y.median():5.1f}  p75 {y.quantile(.75):5.1f}  p90 {y.quantile(.9):5.1f}  max {y.max():5.1f}")
