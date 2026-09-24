"""Team opportunity context, shared by the prospect model and the diagnostics.

For a team going into a season, from the previous season's rosters (per-team-game fantasy production of each player):
  (all "production" = each rotation player (>=10 mpg) at his per-game fantasy level, i.e. a HEALTHY roster; an
   availability-weighted version was tested and predicted rookie output worse -- it makes injured stars look small)
  returning_fp   production from players still on the team
  vacated_fp     production from players who left (traded / FA / retired)
  arriving_fp    last season's production of veterans who joined from elsewhere
  open_fp        vacated - arriving   (net production up for grabs; the useful one -- see prospect_context_test.py)
  crowd_fp       returning + arriving (what a newcomer has to beat for minutes)
  star_ret       best returning player's pts/g
"""
import numpy as np
import pandas as pd


def prep_panel(panel):
    p = panel.copy()
    p["maxgp"] = p.groupby("yr")["GP"].transform("max")
    p["fpc"] = p["fpg"] * p["GP"] / p["maxgp"]  # availability-weighted (kept for reference; injuries make it understate stars)
    p["fp_h"] = np.where(p["mpg"] >= 10, p["fpg"], 0.0)  # HEALTHY-roster production: each rotation player at his per-game level
    return p


def team_context(panel, yr_prev, next_team_col="team_id_next"):
    """context for every team entering season yr_prev+1. `next_team_col` = each player's team at the start of that season
    (historical: first game of the season; live: ESPN's current team)."""
    p = panel[panel["yr"] == yr_prev]
    rows = {}
    for T in set(p["team_id"].dropna()).union(set(p[next_team_col].dropna())):
        prior = p[p["team_id"] == T]
        ret = prior[prior[next_team_col] == T]
        dep = prior[prior[next_team_col] != T]
        arr = p[(p[next_team_col] == T) & (p["team_id"] != T)]
        rows[int(T)] = dict(returning_fp=ret["fp_h"].sum(), vacated_fp=dep["fp_h"].sum(), arriving_fp=arr["fpc"].sum() * 0 + arr["fp_h"].sum(),
                            star_ret=ret["fpg"].max() if len(ret) else 0.0)
    out = pd.DataFrame(rows).T
    out["open_fp"] = out["vacated_fp"] - out["arriving_fp"]
    out["crowd_fp"] = out["returning_fp"] + out["arriving_fp"]
    return out
