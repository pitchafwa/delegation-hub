"""Strategy study #7 (2026-10-24): how much is stashing injured free agents on IR worth, and how often is it possible?

2025-26 replay. For every player who missed 8+ of his NBA team's games in a row (mid-season, from the box scores) after averaging 28+ fantasy points before:
  * was he a FREE AGENT at some point during the absence (not on any of the 12 rosters that day, league_days_2026.json)? -> a stash opportunity
  * when did he come back, and what did he score after returning: through the end of the fantasy regular season (week 19) and in the playoff weeks (20-22)?
  * what that is worth to a roster: an IR slot costs no roster spot while he is out; once back he replaces the weakest player (about 25 points a game), so
    value = (his points per game after return - 25) x his games, with playoff-week games also shown separately (they count far more toward the title: study #6).
Run from ingest/:  .venv/Scripts/python research/ir_stash_study.py
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0]]
import backtest_weekly_planner as B
import season_calendar as SC

sys.stdout.reconfigure(encoding="utf-8")
REPL = 25.0
g = B.g.copy()
g["d"] = g.GAME_DATE.dt.date
rostered = {B.sp_date(int(sp)): {B.norm(e[1]) for r in recs for e in r["e"]} for sp, recs in B.league.items()}
r_dates = sorted(rostered)
cal = {2026: None}
# fantasy calendar for 2025-26: 19 regular weeks from the opener, then 3 playoff weeks (the same shape as this season; the 14-day All-Star week is ignored here)
opener = min(r_dates)
reg_end = opener + timedelta(days=6 + 7 * 18 + 7)                    # approximate: matchup 1 is 6 days, then 7-day weeks (+1 extra week for the 14-day matchup)
po_end = reg_end + timedelta(days=21)
rows = []
for n, sub in g.sort_values("GAME_DATE").groupby("n"):
    sub = sub.reset_index(drop=True)
    for i in range(1, len(sub)):
        tm = sub.TEAM_ABBREVIATION[i - 1]
        d0, d1 = sub.d[i - 1], sub.d[i]
        missed = sum(1 for dd in B.team_days.get(tm, ()) if d0 < dd < d1)
        if missed < 8 or d0 > reg_end - timedelta(days=14):
            continue
        pre = sub.fp[max(0, i - 20):i].mean()
        if pre < 28:
            continue
        out_days = [x for x in r_dates if d0 < x < d1]
        fa_days = [x for x in out_days if n not in rostered[x]]
        post = sub[(sub.d >= d1)]
        reg = post[post.d <= reg_end]
        po = post[(post.d > reg_end) & (post.d <= po_end)]
        rows.append({"player": sub.PLAYER_NAME[i], "pre": round(pre, 1), "missed": missed, "out_from": d0, "back": d1, "fa_days": len(fa_days), "out_days": len(out_days),
                     "post_ppg": round(post.fp.mean(), 1) if len(post) else np.nan, "reg_games": len(reg), "po_games": len(po),
                     "value_reg": round(((reg.fp - REPL).clip(lower=-REPL)).sum(), 0), "value_po": round(((po.fp - REPL).clip(lower=-REPL)).sum(), 0)})
df = pd.DataFrame(rows)
print(f"{len(df)} absences of 8+ games by 28+ point players in 2025-26")
opp = df[df.fa_days > 0]
print(f"  of those, {len(opp)} were free agents at some point while out ({len(opp) / max(1, len(df)):.0%}); typical: out {opp.missed.median():.0f} games, available {opp.fa_days.median():.0f} of {opp.out_days.median():.0f} days")
print(f"  back before the fantasy playoffs: {(opp.back <= reg_end).mean():.0%};  played playoff-week games: {(opp.po_games > 0).mean():.0%}")
print(f"  after returning: {opp.post_ppg.median():.1f} pts/game (before the injury {opp.pre.median():.1f})")
print(f"  value over a 25-point replacement: regular season {opp.value_reg.median():+.0f} pts (mean {opp.value_reg.mean():+.0f}), playoff weeks {opp.value_po.median():+.0f} (mean {opp.value_po.mean():+.0f})")
print(f"  stash opportunities per week of the season: {len(opp) / 19:.1f}")
print("\nlargest opportunities:")
print(opp.sort_values("value_reg", ascending=False).head(12)[["player", "pre", "missed", "fa_days", "back", "post_ppg", "reg_games", "po_games", "value_reg", "value_po"]].to_string(index=False))
df.to_csv(Path(__file__).resolve().parent / "data" / "ir_stash_study.csv", index=False)
