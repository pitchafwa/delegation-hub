"""Short-term program, test 2 (RESEARCH_short_term_projection_plan.md): does moving into or out of the starting five add anything on top of model A?
Starter flags for 2024-25 on come from box_v3 (pull_box_v3.py). Signals: last game started vs his share over the 10 games before the last 5 (chg1),
last-5 start share vs that (chg5), and chg5 scaled by his scoring rate.  Run from ingest/:  .venv/Scripts/python research/st_test2.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_harness as H                                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
P = H.P
rate = P["delco"] / P["delco_min"].clip(lower=8)
A = H.a_feats()
chg1 = (P["start_l1"] - P["start_prev10"])
chg5 = (P["start_l5"] - P["start_prev10"])
print(f"info points with a start change (|chg5| >= 0.4): {(chg5.abs() >= 0.4).mean():.1%}")
variants = {
    "A (live)": A,
    "A + chg1": {**A, "chg1": chg1},
    "A + chg5": {**A, "chg5": chg5},
    "A + chg1 + chg5": {**A, "chg1": chg1, "chg5": chg5},
    "A + chg5 x rate": {**A, "chg5": chg5, "chg5_rate": chg5 * rate},
}
preds = {}
for name, f in variants.items():
    c = H.fit_by_k(f)
    preds[name] = H.predict_by_k(f, c)
    extra = [k for k in f if k not in A]
    if extra:
        print(name, {k: [round(c[b][list(f).index(k) + 1], 2) for b in range(len(H.KB))] for k in extra})
lt = H.level_table(preds)
cols = [c for _, _, c in H.KB] + [c for c in lt.columns if c.endswith("(g1-6)")]
print("\nTypical miss (fantasy pts/g), test seasons 2022-23..2025-26; change vs A in brackets")
first = lt.index[0]
out = lt[cols].round(3).astype(str)
for c in cols:
    out[c] = lt[c].round(3).astype(str) + " (" + (lt[c] - lt.loc[first, c]).round(3).map(lambda v: f"{v:+.3f}") + ")"
print(out.T.to_string())
lean = [c for c in lt.columns if c.endswith("lean")]
print("\nLean (+ = scored more than predicted)")
print(lt[lean].round(2).T.to_string())
