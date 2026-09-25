"""Chance of playing by official status x rotation/bench x recent participation, from 5 seasons of reports (see RESEARCH_injuries.md).
Writes availability_table.json (committed, read by build_week_plan.py).  Needs data/injury_rows_05pm.pkl from injury_playrate_study.py.
state: first = not listed in the previous 7 days; played = listed before and played his last game; missed = listed before and missed it.
Small cells are shrunk toward the status x rotation average (k = 30).
Run from ingest/:  uv run python research/build_availability_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
e = pd.read_pickle(D / "injury_rows_05pm.pkl")
inj = e[e.kind.isin(["injury", "illness"]) & e.status.isin(["Questionable", "Probable", "Doubtful", "Available"]) & e.rot.isin(["rotation", "bench"])].copy()
inj["state"] = np.where(inj.first_day == 1, "first", np.where(inj.prev_played == 1, "played", "missed"))
K = 30
table = {}
for (st, rot), g in inj.groupby(["status", "rot"]):
    pooled = g.played.mean()
    table[f"{st}|{rot}|all"] = {"p": round(float(pooled), 3), "n": int(len(g))}
    for state, gg in g.groupby("state"):
        n = len(gg)
        table[f"{st}|{rot}|{state}"] = {"p": round(float((K * pooled + n * gg.played.mean()) / (K + n)), 3), "n": int(n)}
table["Out|any|all"] = {"p": round(float(e[e.status == "Out"].played.mean()), 3), "n": int((e.status == "Out").sum())}
json.dump({"note": "P(play) by official status | rotation (20+ mpg over previous 15) or bench | state; from NBA injury reports 2021-22 to 2025-26", "table": table},
          open(Path(__file__).resolve().parent / "availability_table.json", "w"), indent=1)
for k, v in sorted(table.items()):
    print(f"{k:34s} p={v['p']:.3f}  n={v['n']}")
