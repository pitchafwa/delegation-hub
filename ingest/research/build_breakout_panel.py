"""Breakout-candidate model, step 1: build the labeled player-season panel.

One row per (player, season t) with ONLY information available at the end of
season t (lagged features), plus the season t+1 outcome used to label a
breakout. Everything downstream (signal tests, model) reads this panel.

League scoring (same as build_hub_data.py):
  PTS + 1.5 REB + 2 AST + 3 STL + 3 BLK + FG3M + 2 FTM - FTA - TOV + 3 TD3
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"


def fantasy_total(df):
    return (df["PTS"] + 1.5 * df["REB"] + 2 * df["AST"] + 3 * df["STL"] + 3 * df["BLK"]
            + df["FG3M"] + 2 * df["FTM"] - df["FTA"] - df["TOV"] + 3 * df["TD3"])


base = pd.read_csv(D / "player_season_base.csv")
adv = pd.read_csv(D / "player_season_advanced.csv")
base["yr"] = base["SEASON"].str[:4].astype(int)
adv["yr"] = adv["SEASON"].str[:4].astype(int)

# a traded player has one row per stint + a TOT row in nba_api splits? keep the
# largest-GP row per player-season to be safe
base = base.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
adv = adv.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])

base["fpg"] = fantasy_total(base) / base["GP"].replace(0, np.nan)
base["mpg"] = base["MIN"] / base["GP"].replace(0, np.nan)
base["fp_per36"] = base["fpg"] / base["mpg"] * 36

panel = base[["PLAYER_ID", "PLAYER_NAME", "yr", "AGE", "GP", "mpg", "fpg", "fp_per36",
              "FG3A", "FTA", "PTS", "TOV", "AST"]].copy()
panel = panel.merge(
    adv[["PLAYER_ID", "yr", "USG_PCT", "TS_PCT", "PIE", "AST_PCT", "REB_PCT", "OFF_RATING", "NET_RATING"]],
    on=["PLAYER_ID", "yr"], how="left")

# --- late-season trend, from real game logs (season t only) -----------------
gl = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "MIN", "PTS", "REB", "AST",
                                                 "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
gl["yr"] = gl["SEASON"].str[:4].astype(int)
gl["fp"] = fantasy_total(gl)
gl = gl.sort_values(["PLAYER_ID", "yr", "GAME_DATE"])
LATE = 20


def late_feats(g):
    if len(g) < LATE + 10:
        return pd.Series({"late_mpg": np.nan, "late_fpg": np.nan, "late_min_delta": np.nan, "late_fpg_delta": np.nan,
                          "late_fp_per36": np.nan})
    early, late = g.iloc[:-LATE], g.iloc[-LATE:]
    return pd.Series({
        "late_mpg": late["MIN"].mean(), "late_fpg": late["fp"].mean(),
        "late_min_delta": late["MIN"].mean() - early["MIN"].mean(),
        "late_fpg_delta": late["fp"].mean() - early["fp"].mean(),
        "late_fp_per36": late["fp"].sum() / max(late["MIN"].sum(), 1) * 36,
    })


late = gl.groupby(["PLAYER_ID", "yr"]).apply(late_feats, include_groups=False).reset_index()
panel = panel.merge(late, on=["PLAYER_ID", "yr"], how="left")

# --- draft / experience -----------------------------------------------------
bio = pd.read_csv(D / "player_bio.csv")[["PERSON_ID", "DRAFT_YEAR", "DRAFT_NUMBER"]].rename(columns={"PERSON_ID": "PLAYER_ID"})
panel = panel.merge(bio, on="PLAYER_ID", how="left")
panel["draft_pick"] = panel["DRAFT_NUMBER"].fillna(61)  # undrafted = 61
first_yr = base.groupby("PLAYER_ID")["yr"].min().rename("first_obs_yr")
panel = panel.merge(first_yr, on="PLAYER_ID", how="left")
# NBA experience: seasons since draft year when known (draft year Y -> rookie season Y), else since first observed
panel["exp"] = np.where(panel["DRAFT_YEAR"].notna() & (panel["DRAFT_YEAR"] >= 2005),
                        panel["yr"] - panel["DRAFT_YEAR"], panel["yr"] - panel["first_obs_yr"])

# --- prior-season deltas ----------------------------------------------------
prev = panel[["PLAYER_ID", "yr", "fpg", "mpg", "fp_per36", "USG_PCT", "TS_PCT"]].copy()
prev["yr"] += 1
prev = prev.rename(columns={c: c + "_prev" for c in ["fpg", "mpg", "fp_per36", "USG_PCT", "TS_PCT"]})
panel = panel.merge(prev, on=["PLAYER_ID", "yr"], how="left")
panel["d_fpg"] = panel["fpg"] - panel["fpg_prev"]
panel["d_mpg"] = panel["mpg"] - panel["mpg_prev"]
panel["d_fp36"] = panel["fp_per36"] - panel["fp_per36_prev"]
panel["d_usg"] = panel["USG_PCT"] - panel["USG_PCT_prev"]
panel["d_ts"] = panel["TS_PCT"] - panel["TS_PCT_prev"]

# --- next-season outcome ----------------------------------------------------
nxt = panel[["PLAYER_ID", "yr", "fpg", "GP", "mpg"]].copy()
nxt["yr"] -= 1
nxt = nxt.rename(columns={"fpg": "fpg_next", "GP": "GP_next", "mpg": "mpg_next"})
panel = panel.merge(nxt, on=["PLAYER_ID", "yr"], how="left")
panel["d_next"] = panel["fpg_next"] - panel["fpg"]

panel.to_csv(D / "breakout_panel.csv", index=False)
print(f"panel: {len(panel)} rows, {panel['PLAYER_ID'].nunique()} players, seasons {panel['yr'].min()}-{panel['yr'].max()}")
print(f"with next-season outcome: {panel['fpg_next'].notna().sum()}")
print(panel.describe().T[["count", "mean", "std", "min", "max"]].round(2).to_string())
