"""Phase 6.1 report: compare the replay results of each next-few-days number (st_replay.py), paired by team-week.
Value of a week = points scored + the measured future cost of its drops (negative), for the rules now live ("now") and for no moves ("none").
Wins: share of team-weeks where the source beats v1 (live) on that value.  Run from ingest/:  .venv/Scripts/python research/st_replay_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
SRCS = ["ewma", "v1", "A", "A_avail"]
df = pd.concat([pd.read_csv(D / f"st_replay_{s}.csv").assign(src=s) for s in SRCS if (D / f"st_replay_{s}.csv").exists()])
df["value"] = df["points"] + df["future"].fillna(0.0)
rows = []
for var in ["none", "now"]:
    x = df[df["variant"] == var].pivot_table(index=["mp", "team"], columns="src", values="value")
    st = df[df["variant"] == var].pivot_table(index=["mp", "team"], columns="src", values="starts")
    ad = df[df["variant"] == var].pivot_table(index=["mp", "team"], columns="src", values="adds")
    x = x.dropna()
    for s in x.columns:
        d = x[s] - x["v1"]
        rows.append({"rules": var, "number": s, "weeks": len(x), "value per week": x[s].mean(), "vs live (v1)": d.mean(), "se": d.std() / np.sqrt(len(d)),
                     "beats live": (d > 0).mean(), "ties": (d == 0).mean(), "starts": st[s].mean(), "adds": ad[s].mean()})
r = pd.DataFrame(rows)
print("Per team-week, 2025-26 replay (points this week + measured future cost of drops):")
print(r.round(2).to_string(index=False))
