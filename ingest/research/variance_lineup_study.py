"""Strategy study #1 (2026-10-24): should lineups pick boom-or-bust players when you are the underdog and steady ones when you are favored?

Step 1, measured: how much do players with the SAME average differ in game-to-game spread? (2025-26 box scores, players with 30+ games and 15+ minutes)
Step 2, simulated with those real spreads: the biggest win-chance swing that choosing the high-spread over the low-spread version of a same-average player can buy,
for 1-4 such choices, on the last day of a matchup (the whole day's spread is the live model's 1-day sigma), at margins from tied to 150 points behind.
If the best case is a fraction of a percentage point, the idea is not worth building into the planner.
Run from ingest/:  .venv/Scripts/python research/variance_lineup_study.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
WPM = json.load(open(R / "win_prob_model.json"))
SIG = {int(k): v for k, v in WPM["sigma"].items()}
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

df = pd.read_csv(R / "data" / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA"])
df = df[df["SEASON"] == "2025-26"]
df["fp"] = df.PTS + 1.5 * df.REB + 2 * df.AST + 3 * df.STL + 3 * df.BLK + df.FG3M + 2 * df.FTM - df.FTA - df.TOV
g = df.groupby("PLAYER_ID").agg(n=("fp", "size"), mean=("fp", "mean"), sd=("fp", "std"), mpg=("MIN", "mean"))
g = g[(g.n >= 30) & (g.mpg >= 15)]
print(f"{len(g)} players")
bins = pd.cut(g["mean"], [0, 20, 25, 30, 35, 40, 50, 80])
t = g.groupby(bins, observed=True)["sd"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[["count", "10%", "25%", "50%", "75%", "90%"]].round(1)
print("game-to-game spread (SD of fantasy points) by average:\n", t.to_string())
cv = (g.sd / g["mean"])
print(f"spread as a share of average: median {cv.median():.2f}, 10th-90th pct {cv.quantile(0.1):.2f}-{cv.quantile(0.9):.2f}")

# step 2: same-average pair at ~30 pts/g, low spread = 10th pct, high = 90th pct of that bin
row = t.loc[t.index[[i for i, b in enumerate(t.index) if b.left == 25][0]]] if any(b.left == 25 for b in t.index) else t.iloc[2]
lo, hi = float(row["10%"]), float(row["90%"])
print(f"\nexample pair (25-30 pts/g): steady SD {lo}, boom-or-bust SD {hi}")
day_sd = SIG[1]
print(f"whole last day spread (both teams, live model): {day_sd:.0f}\n")
for k in (1, 2, 4):
    for margin in (0, -25, -50, -100, -150):
        base_var = day_sd ** 2
        s_lo = math.sqrt(base_var)
        s_hi = math.sqrt(base_var + k * (hi ** 2 - lo ** 2))
        wp_lo, wp_hi = Phi(margin / s_lo), Phi(margin / s_hi)
        # what the same win chance would cost in plain points (margin needed for the same change)
        print(f"{k} choice(s), behind by {-margin:3d}: win chance {wp_lo:.1%} -> {wp_hi:.1%}  ({(wp_hi - wp_lo) * 100:+.2f} pts of win chance)")
    print()
# and with only the last few games of the night left (a quarter of a day)
q_sd = SIG[1] * 0.5
print(f"late evening, a quarter of the day left (spread {q_sd:.0f}):")
for k in (1, 2):
    for margin in (-10, -25, -50):
        s_lo, s_hi = q_sd, math.sqrt(q_sd ** 2 + k * (hi ** 2 - lo ** 2))
        print(f"{k} choice(s), behind by {-margin}: {Phi(margin / s_lo):.1%} -> {Phi(margin / s_hi):.1%}")
