"""Do rotation players sit more on the second night of a back-to-back, and by how much?  (games-played risk for start/sit)

Team schedule = every date any of the team's players logged a game. For each regular rotation player (>=24 mpg over his
previous 15 games) and each team game, "absent" = he did not play although he played in one of the previous 3 team games AND
one of the next 3 team games (a short, mid-run absence: mostly rest / day-to-day, NOT a long injury, which would fail the
'played next 3 games' test). Compare absence rates by rest situation, age and star status.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
lg = pd.read_csv(R / "data" / "game_logs_unified.csv", usecols=["PLAYER_ID", "PLAYER_NAME", "SEASON", "GAME_DATE", "TEAM", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
lg = lg[lg.SEASON.isin(["2022-23", "2023-24", "2024-25", "2025-26"])].copy()
lg["date"] = pd.to_datetime(lg.GAME_DATE)
lg["fp"] = lg.PTS + 1.5 * lg.REB + 2 * lg.AST + 3 * lg.STL + 3 * lg.BLK + lg.FG3M + 2 * lg.FTM - lg.FTA - lg.TOV + 3 * lg.TD3
bio = pd.read_csv(R / "data" / "player_bio.csv", usecols=["PERSON_ID", "PLAYER_FIRST_NAME", "PLAYER_LAST_NAME"]) if (R / "data" / "player_bio.csv").exists() else None

rows = []
for (season, team), tg in lg.groupby(["SEASON", "TEAM"]):
    dates = np.sort(tg.date.unique())
    idx = {d: i for i, d in enumerate(dates)}
    b2b = np.array([0] + [int((dates[i] - dates[i - 1]) / np.timedelta64(1, "D") == 1) for i in range(1, len(dates))])
    rest3 = np.array([0, 0] + [int((dates[i] - dates[i - 2]) / np.timedelta64(1, "D") <= 3) for i in range(2, len(dates))])  # 3 games in 4 nights style
    for pid, pg in tg.groupby("PLAYER_ID"):
        played = np.zeros(len(dates), dtype=int)
        mins = np.zeros(len(dates))
        for d, mn in zip(pg.date, pg.MIN):
            played[idx[d]] = 1
            mins[idx[d]] = mn
        if played.sum() < 20:
            continue
        for i in range(15, len(dates) - 3):
            prev = mins[i - 15:i]
            n_prev = (prev > 0).sum()
            if n_prev < 10 or prev[prev > 0].mean() < 24:
                continue
            mid = (played[i - 3:i].sum() > 0) and (played[i + 1:i + 4].sum() > 0)
            if not mid and played[i] == 0:
                continue                     # long absence (injury) -> excluded rather than counted
            rows.append((season, team, pid, i, int(played[i] == 0), b2b[i], prev[prev > 0].mean()))
df = pd.DataFrame(rows, columns=["season", "team", "pid", "g", "absent", "b2b", "mpg"])
print(f"{len(df):,} rotation player-games ({df.pid.nunique()} players)")
print(f"short mid-run absence rate overall: {df.absent.mean():.1%}")
print("\nby second-night-of-back-to-back:")
print(df.groupby("b2b").absent.agg(["mean", "size"]).round(3).to_string())
df["tier"] = pd.cut(df.mpg, [23.9, 30, 34, 60], labels=["24-30 mpg", "30-34 mpg", "34+ mpg (stars)"])
print("\nby minutes tier and b2b:")
print(df.groupby(["tier", "b2b"], observed=True).absent.agg(["mean", "size"]).round(3).to_string())
print("\nby season (b2b games only vs others):")
print(df.groupby(["season", "b2b"]).absent.mean().unstack().round(3).to_string())
