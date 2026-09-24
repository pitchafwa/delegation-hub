"""Breakout model, situation features: where a player is playing NEXT season
vs. where they played this season.

For player i with final team X in season t and (known-in-the-offseason) team Y
in season t+1:
  moved            X != Y
  dest_net_usg     usage-minutes that LEFT team Y since season t, minus usage-minutes
                   that ARRIVED at Y (excluding i), as a share of Y's team minutes.
                   Positive = opportunity opened up at the destination.
  dest_vac_min     minutes share that left Y (raw vacated opportunity)
  coach_change     Y's head coach in t+1 shares no name with Y's head coaches in t
                   (mid-season changes inside t+1 are NOT counted -- not knowable
                   in the offseason; coaches come from nba_api CommonTeamRoster)

Reads data/breakout_panel.csv + team_coaches.csv, writes data/breakout_panel_ctx2.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"

p = pd.read_csv(D / "breakout_panel.csv")
b = pd.read_csv(D / "player_season_base.csv")
b["yr"] = b["SEASON"].str[:4].astype(int)
b = b.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
# X = team at the END of season t (last game); Y = team at the START of t+1 (first
# game) -- so an in-season trade during t+1 (unknowable in the offseason) can't leak in
gl = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "TEAM"])
gl["yr"] = gl["SEASON"].str[:4].astype(int)
gl = gl.sort_values(["PLAYER_ID", "yr", "GAME_DATE"])
last_t = gl.groupby(["PLAYER_ID", "yr"])["TEAM"].last().rename("team").reset_index()
first_t = gl.groupby(["PLAYER_ID", "yr"])["TEAM"].first().rename("team_first").reset_index()
abbr_id = b.drop_duplicates(["TEAM_ABBREVIATION", "yr"]).set_index(["TEAM_ABBREVIATION", "yr"])["TEAM_ID"].to_dict()
last_t["team_id"] = [abbr_id.get((t, y)) for t, y in zip(last_t["team"], last_t["yr"])]
first_t["team_id_first"] = [abbr_id.get((t, y)) for t, y in zip(first_t["team_first"], first_t["yr"])]
tot = b[["PLAYER_ID", "yr", "MIN"]].rename(columns={"MIN": "totmin"})
team = last_t.merge(tot, on=["PLAYER_ID", "yr"], how="left")
p = p.merge(team, on=["PLAYER_ID", "yr"], how="left")
nxt = first_t[["PLAYER_ID", "yr", "team_id_first", "team_first"]].copy()
nxt["yr"] -= 1
p = p.merge(nxt.rename(columns={"team_id_first": "team_id_next", "team_first": "team_next"}), on=["PLAYER_ID", "yr"], how="left")


def load_coach_sets():
    path = D / "team_coaches.csv"
    if not path.exists():
        return None
    co = pd.read_csv(path)
    co = co[co["coach_type"] == "Head Coach"]  # exclude Associate/Assistant "Head" titles
    return co.groupby(["team_id", "yr"])["coach"].apply(set).to_dict()


def situation_features(p, coach_sets=None):
    """Adds moved / dest_net_usg / dest_vac_min / coach_change to a frame with
    yr, team_id (end of season t), team_id_next (start of t+1), totmin, USG_PCT.
    Shared by the training build and live scoring so both use identical logic."""
    p = p.copy()
    p["usgmin"] = p["USG_PCT"].fillna(0.18) * p["totmin"]
    p["moved"] = (p["team_id"] != p["team_id_next"]).astype(float).where(p["team_id_next"].notna())
    rows = []
    for yr, g in p.groupby("yr"):
        tm = g.groupby("team_id")["totmin"].sum()
        for Y, tg in g.groupby("team_id"):
            dep = tg[tg["team_id_next"] != Y]
            rows.append(dict(yr=yr, team_id_next=Y, vac_usg=dep["usgmin"].sum(), vac_min=dep["totmin"].sum(), team_min=tm[Y]))
    p = p.merge(pd.DataFrame(rows), on=["yr", "team_id_next"], how="left")
    arr = p[p["moved"] == 1].groupby(["yr", "team_id_next"])["usgmin"].sum().rename("arr_usg").reset_index()
    p = p.merge(arr, on=["yr", "team_id_next"], how="left")
    p["arr_usg"] = p["arr_usg"].fillna(0)
    self_arr = np.where(p["moved"] == 1, p["usgmin"], 0)
    p["dest_net_usg"] = (p["vac_usg"] - (p["arr_usg"] - self_arr)) / p["team_min"]
    p["dest_vac_min"] = p["vac_min"] / p["team_min"]
    if coach_sets:
        def changed(r):
            if pd.isna(r["team_id_next"]):
                return np.nan
            a = coach_sets.get((int(r["team_id_next"]), int(r["yr"])))
            b2 = coach_sets.get((int(r["team_id_next"]), int(r["yr"]) + 1))
            return np.nan if not a or not b2 else float(len(a & b2) == 0)
        p["coach_change"] = p.apply(changed, axis=1)
    return p.drop(columns=["vac_usg", "vac_min", "team_min", "arr_usg", "usgmin"])


if __name__ == "__main__":
    out = situation_features(p, load_coach_sets())
    if "coach_change" in out:
        print("coach_change rate:", round(out["coach_change"].mean(), 3), " coverage:", round(out["coach_change"].notna().mean(), 3))
    out.to_csv(D / "breakout_panel_ctx2.csv", index=False)
    print("saved breakout_panel_ctx2.csv", len(out))
