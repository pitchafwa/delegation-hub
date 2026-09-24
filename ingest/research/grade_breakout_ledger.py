"""Grade a frozen breakout ledger against what actually happened (run after the season ends and
the season's player_season_base rows exist). Writes data/breakout_ledger/<season>_graded.csv and
prints hit-rates for the model's top 10/25 vs base rate. Same label definitions as breakout_model.py.

  python grade_breakout_ledger.py 2026-27
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
season = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
Q, TOL = 32.4, 1.0
led = pd.read_csv(D / "breakout_ledger" / f"{season}.csv")
b = pd.read_csv(D / "player_season_base.csv")
b = b[b["SEASON"] == season].sort_values("GP", ascending=False).drop_duplicates("PLAYER_ID")
if b.empty:
    sys.exit(f"No {season} rows in player_season_base.csv yet -- refresh data first.")
b["fpg_actual"] = (b["PTS"] + 1.5 * b["REB"] + 2 * b["AST"] + 3 * b["STL"] + 3 * b["BLK"] + b["FG3M"] + 2 * b["FTM"]
                   - b["FTA"] - b["TOV"] + 3 * b["TD3"]) / b["GP"].replace(0, np.nan)
g = led.merge(b[["PLAYER_ID", "GP", "fpg_actual"]].rename(columns={"GP": "gp_actual"}), on="PLAYER_ID", how="left")
g["valid"] = g["gp_actual"] >= 30
g["d_actual"] = g["fpg_actual"] - g["fpg_last"]
g["hit"] = np.where(~g["valid"], np.nan, np.where(g["tier"] == "emergence", (g["fpg_actual"] >= Q - TOL) & (g["d_actual"] >= 6),
                                                   g["d_actual"] >= 6).astype(float))
g["bust_hit"] = np.where(~g["valid"] | (g["tier"] != "elevation"), np.nan, (g["d_actual"] <= -6).astype(float))
g.to_csv(D / "breakout_ledger" / f"{season}_graded.csv", index=False)
v = g[g["valid"]].sort_values("p_break", ascending=False)
print(f"{season}: {len(v)} graded ({len(g) - len(v)} n/a). base breakout rate {v['hit'].mean():.1%}")
for n in (10, 25):
    print(f"  top {n} by breakout chance: {int(v.head(n)['hit'].sum())}/{n}")
vb = v[v["p_bust"].notna()].sort_values("p_bust", ascending=False)
print(f"  top 10 by bust risk: {int(vb.head(10)['bust_hit'].sum())}/10 dropped 6+ (base {vb['bust_hit'].mean():.1%})")
