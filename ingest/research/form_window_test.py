"""How much should recent form matter for a single game's fantasy points?  (baseline for the daily projection)

For each player-game (players with >=15 mpg baseline, 2023-24..2025-26) predict fantasy points from: last 5 / 10 / 20 / 40 game
means, the season-to-date mean, and exponentially weighted means with different half-lives. Reports RMSE (lower = better) so the
daily model can weight recent games vs the longer-run level the season model already knows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
lg = pd.read_csv(R / "data" / "game_logs_unified.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
lg = lg[lg.SEASON.isin(["2022-23", "2023-24", "2024-25", "2025-26"])].copy()
lg["date"] = pd.to_datetime(lg.GAME_DATE)
lg["fp"] = lg.PTS + 1.5 * lg.REB + 2 * lg.AST + 3 * lg.STL + 3 * lg.BLK + lg.FG3M + 2 * lg.FTM - lg.FTA - lg.TOV + 3 * lg.TD3
lg = lg.sort_values(["PLAYER_ID", "date"])
g = lg.groupby("PLAYER_ID")
preds = {}
for w in (5, 10, 20, 40, 80):
    preds[f"last {w}"] = g.fp.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=min(w, 5)).mean())
for hl in (5, 10, 20, 40):
    preds[f"EWMA half-life {hl}"] = g.fp.transform(lambda s, hl=hl: s.shift(1).ewm(halflife=hl, min_periods=5).mean())
preds["season to date"] = lg.groupby(["PLAYER_ID", "SEASON"]).fp.transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
base_min = g.MIN.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
ok = (base_min >= 15) & (lg.SEASON.isin(["2023-24", "2024-25", "2025-26"]))
for k in preds:
    ok &= preds[k].notna()
y = lg.fp[ok]
print(f"{ok.sum():,} player-games")
res = {k: np.sqrt(((y - v[ok]) ** 2).mean()) for k, v in preds.items()}
for k, v in sorted(res.items(), key=lambda kv: kv[1]):
    print(f"  {k:22s} RMSE {v:.3f}")
# blends: recent window with the long-run level
for a in (0.2, 0.35, 0.5):
    b = a * preds["last 10"][ok] + (1 - a) * preds["last 80"][ok]
    print(f"  {a:.0%} last-10 + {1 - a:.0%} last-80    RMSE {np.sqrt(((y - b) ** 2).mean()):.3f}")
