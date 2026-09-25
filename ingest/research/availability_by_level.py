"""Chance a player suits up in a scheduled team game, by his recent level (fantasy pts/g) and role, using only past games.
Population: player-team-games where the player had >=8 games this season and played within the last 10 days (so long injuries are excluded)."""
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
DD = Path(__file__).resolve().parent / "data" / "game_logs"
g = pd.concat([pd.read_csv(DD / f"nba_api_{s}.csv").assign(S=s) for s in ("2024-25", "2025-26")])
g["GAME_DATE"] = pd.to_datetime(g.GAME_DATE)
g["fp"] = g.PTS + 1.5*g.REB + 2*g.AST + 3*g.STL + 3*g.BLK + g.FG3M + 2*g.FTM - g.FTA - g.TOV
rows = []
for (s, t), tg in g.groupby(["S", "TEAM_ABBREVIATION"]):
    dates = np.sort(tg.GAME_DATE.unique())
    for pid, pg in tg.groupby("PLAYER_ID"):
        played = set(pg.GAME_DATE)
        if len(played) < 15:
            continue
        pgs = pg.sort_values("GAME_DATE")
        fps = pgs.fp.to_numpy(); ds = pgs.GAME_DATE.to_numpy(); mins = pgs.MIN.to_numpy()
        for d in dates:
            past = ds < d
            n = past.sum()
            if n < 8:
                continue
            last = ds[past].max()
            if (d - last) / np.timedelta64(1, "D") > 10:
                continue
            w = 0.5 ** (np.arange(n)[::-1] / 8.0)
            lvl = float((fps[past] * w).sum() / w.sum())
            mpg = float(mins[past][-10:].mean())
            rows.append((lvl, mpg, int(d in played)))
df = pd.DataFrame(rows, columns=["level", "mpg10", "played"])
print(f"{len(df):,} player-scheduled-games; overall play rate {df.played.mean():.1%}")
df["band"] = pd.cut(df.level, [-1, 15, 20, 25, 30, 40, 80], labels=["<15", "15-20", "20-25", "25-30", "30-40", "40+"])
print(df.groupby("band", observed=True).played.agg(["mean", "size"]).round(3).to_string())
df["mb"] = pd.cut(df.mpg10, [-1, 12, 18, 24, 30, 60])
print(df.groupby("mb", observed=True).played.agg(["mean", "size"]).round(3).to_string())
