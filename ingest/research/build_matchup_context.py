"""Builds dashboard/team_matchup.json: each NBA team's trailing defensive rating (points allowed per 100 pace-proxy possessions) and defense
allowed by position (fp allowed to point-guard-style / forward-style / center-style scorers this season). Feeds the matchup adjustment
(matchup_context.py, matchup_model.json, RESEARCH_gamelevel.md) that build_week_plan.py applies per day based on that day's actual opponent.

Early in the season (or if a team's current-season table can't be built yet) each figure shrinks toward the trained LEAGUE AVERAGE (matchup_model.
json) rather than a prior season's real numbers -- there is no reliable live source for last season's opponent pairings once the schedule file
rolls over to the new season, so shrinking toward the population mean (standard practice elsewhere in this project, e.g. the free-agent level
shrink) is simpler and safer than trying to reconstruct it. Needs real local game logs (form_common.load_all) -- local daily refresh only, not
GitHub Actions (no nba_api there). Uses team_defense_shared.py, the SAME math matchup_fit_final.py trained the model on.
  uv run python research/build_matchup_context.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F
import team_defense_shared as TD

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
BLEND_GAMES = 15          # this-season weight ramps from 0 (pure league average) to 1 (pure current-season) over this many of the team's games
MODEL = json.load(open(Path(__file__).resolve().parent / "matchup_model.json"))
LG = MODEL["league_avg"]

g = F.load_all(cache=False)
g["fp"] = F.fp(g)
season = sorted(g.season.unique())[-1]
print(f"season {season}, through {g[g.season == season].date.max().date() if (g.season == season).any() else 'no games yet'}")

sched = json.load(open(HUB / "nba_schedule.json", encoding="utf-8"))
FIX = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "CHO": "CHA", "PHL": "PHI"}
canon = lambda t: FIX.get(t, t)
opp_by_team_date = {}
for ds, gl in sched.get("games", {}).items():
    for a, h, _ in gl:
        a, h = canon(a), canon(h)
        opp_by_team_date.setdefault(a, {})[ds] = h
        opp_by_team_date.setdefault(h, {})[ds] = a
ALL_TEAMS = sorted(opp_by_team_date.keys())

cur = g[g.season == season].copy()
cur["date_s"] = cur.date.dt.strftime("%Y-%m-%d")
cur["opp"] = [opp_by_team_date.get(t, {}).get(d) for t, d in zip(cur.team, cur.date_s)]
pos_lookup = TD.player_position_mix(cur, min_gp=5)
table = TD.team_defense_table(cur, pos_lookup) if cur.opp.notna().any() else pd.DataFrame()

out = {}
for t in ALL_TEAMS:
    n = float(table.loc[t, "n_games"]) if (len(table) and t in table.index) else 0.0
    w = min(n, BLEND_GAMES) / BLEND_GAMES
    row = {}
    for col, avg_key in [("drtg", "drtg"), ("fpC_pg", "fpC"), ("fpF_pg", "fpF"), ("fpG_pg", "fpG")]:
        c_ = float(table.loc[t, col]) if (len(table) and t in table.index and pd.notna(table.loc[t, col])) else None
        row[avg_key] = (w * c_ + (1 - w) * LG[avg_key]) if c_ is not None else LG[avg_key]
    row["n_games_current"] = n
    out[t] = row
result = {"generated": datetime.now(timezone.utc).isoformat(), "season": season, "blend_games": BLEND_GAMES, "league_avg": LG, "teams": out}
(HUB / "team_matchup.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
n_live = sum(1 for v in out.values() if v["n_games_current"] > 0)
print(f"wrote team_matchup.json for {len(out)} teams ({n_live} with at least one real game so far this season)")
if n_live:
    for t in sorted(out, key=lambda x: -out[x]["drtg"])[:5]:
        print(f"  softest defense so far: {t}  drtg {out[t]['drtg']:.1f}  ({out[t]['n_games_current']:.0f} games)")
