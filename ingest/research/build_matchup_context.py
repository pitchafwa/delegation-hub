"""Builds dashboard/team_matchup.json: each NBA team's trailing defensive rating (points allowed per 100 pace-proxy possessions), defense allowed
by position (fp allowed to point-guard-style / forward-style / center-style scorers this season), and trailing pace (used to derive a specific
matchup's expected pace, e.g. two fast teams meeting -- see matchup_context.expected_pace()). Feeds the matchup adjustment (matchup_context.py,
matchup_model.json, RESEARCH_gamelevel.md) that build_week_plan.py applies per day based on that day's actual opponent.

Early in the season (or if a team's current-season table can't be built yet), each figure shrinks toward that SAME TEAM'S real numbers from LAST
season -- not the population-wide league average -- and that prior-season weight fades to 0 by BLEND_GAMES real games this season (Tommy, 2026-09-27:
"it should be based on last year's pace and updated with data from this year as we get more of it"). Last season's real, team-specific numbers come
from team_box_features.pkl (real historical team box scores with real opponent pairings, independent of nba_schedule.json -- which only holds the
CURRENT season's schedule and so can't tell us who a team played last year); this is the same source matchup_fit_final.py already uses for every
historical season. The population-wide league average is kept only as a last-resort fallback (e.g. an expansion team with no prior-season history).
Needs real local game logs (form_common.load_all) -- local daily refresh only, not GitHub Actions (no nba_api there). Uses team_defense_shared.py,
the SAME math matchup_fit_final.py trained the model on.
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
BLEND_GAMES = 15          # this-season weight ramps from 0 (pure prior-season/league-average) to 1 (pure current-season) over this many of the team's games
MODEL = json.load(open(Path(__file__).resolve().parent / "matchup_model.json"))
LG = MODEL["league_avg"]

# The season we're planning FOR, not the most recent one with real games -- matches build_week_plan.py's SEASON_ID (bump both together each year).
# Deriving this from g.season.unique() instead would be wrong for the whole off-season: right up until opening night, the latest season with any
# real games is still LAST season, which would make "prior season" (the shrink target below) point two years back instead of one.
SEASON_ID = 2027
season = f"{SEASON_ID - 1}-{str(SEASON_ID)[2:]}"
prior_season = f"{SEASON_ID - 2}-{str(SEASON_ID - 1)[2:]}"

g = F.load_all(cache=False)
g["fp"] = F.fp(g)
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

# last season's real, team-specific numbers -- the shrink target, using team_box_features.pkl for real opponent identity (nba_schedule.json only
# has the current season, so it can't tell us who a team played last year).
prior_table = pd.DataFrame()
if prior_season:
    tb_path = Path(__file__).resolve().parent / "data" / "gamelevel" / "team_box_features.pkl"
    if tb_path.exists():
        tb = pd.read_pickle(tb_path)
        FIX_TB = {"NOH": "NOP", "NJN": "BKN", "CHO": "CHA", "SEA": "OKC"}
        tb["team"] = tb.TEAM_ABBREVIATION.map(lambda t: FIX_TB.get(t, t))
        tb["date_s"] = tb.GAME_DATE.dt.strftime("%Y-%m-%d")
        opp_by_team_date_prior = tb.set_index(["team", "date_s"]).opp.to_dict()
        prior = g[g.season == prior_season].copy()
        prior["date_s"] = prior.date.dt.strftime("%Y-%m-%d")
        prior["opp"] = [opp_by_team_date_prior.get((t, d)) for t, d in zip(prior.team, prior.date_s)]
        if prior.opp.notna().any():
            pos_lookup_prior = TD.player_position_mix(prior, min_gp=5)
            prior_table = TD.team_defense_table(prior, pos_lookup_prior)
n_prior = sum(1 for t in ALL_TEAMS if t in prior_table.index) if len(prior_table) else 0
print(f"prior season {prior_season}: real numbers for {n_prior}/{len(ALL_TEAMS)} teams" + ("" if n_prior else " (falling back to league average)"))

out = {}
for t in ALL_TEAMS:
    n = float(table.loc[t, "n_games"]) if (len(table) and t in table.index) else 0.0
    w = min(n, BLEND_GAMES) / BLEND_GAMES
    row = {}
    for col, avg_key in [("drtg", "drtg"), ("fpC_pg", "fpC"), ("fpF_pg", "fpF"), ("fpG_pg", "fpG"), ("pace", "pace")]:
        c_ = float(table.loc[t, col]) if (len(table) and t in table.index and pd.notna(table.loc[t, col])) else None
        prior_c = float(prior_table.loc[t, col]) if (len(prior_table) and t in prior_table.index and pd.notna(prior_table.loc[t, col])) else LG[avg_key]
        row[avg_key] = (w * c_ + (1 - w) * prior_c) if c_ is not None else prior_c
    row["n_games_current"] = n
    out[t] = row
result = {"generated": datetime.now(timezone.utc).isoformat(), "season": season, "prior_season": prior_season, "blend_games": BLEND_GAMES, "league_avg": LG, "teams": out}
(HUB / "team_matchup.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
n_live = sum(1 for v in out.values() if v["n_games_current"] > 0)
print(f"wrote team_matchup.json for {len(out)} teams ({n_live} with at least one real game so far this season)")
if n_live:
    for t in sorted(out, key=lambda x: -out[x]["drtg"])[:5]:
        print(f"  softest defense so far: {t}  drtg {out[t]['drtg']:.1f}  ({out[t]['n_games_current']:.0f} games)")
