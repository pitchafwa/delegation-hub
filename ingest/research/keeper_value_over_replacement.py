"""The real answer to Tommy's question: what's the right time horizon for a
league with only 3-5 (of ~15-19) keepers, structurally between full redraft
(K=0) and full dynasty (K=large)?

Core idea, grounded in real data at every step:
1. Opportunity cost (what you'd get back in the draft instead of keeping
   someone -- NOT the standard positional-scarcity "replacement level" from
   fantasy VBD, which is a different concept and doesn't depend on keeper
   count at all) is empirically a function of how many keepers exist
   league-wide -- fit from THIS LEAGUE's own real history (0, 6, 8
   keepers/team observed across 2021-2025).
2. Project each player's own future trajectory using the fitted age curve
   (the dominant multi-year driver per permutation importance), anchored at
   their real current context.
3. Keeper value = sum of (trajectory value - opportunity cost) for every
   future year until the trajectory first drops below it, capped at a fixed
   7-year horizon (see value_over_opportunity_cost below for why it's capped
   rather than run out to a true crossing point).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fit_talent_model import RAMP_FEATURES

ROOT = Path(__file__).resolve().parent

with open(ROOT / "data" / "layer_a_model.frozen.json") as f:
    params = np.array(json.load(f)["params_raw"])

# Real, empirically-fit opportunity cost by keeper count (from
# fit_opportunity_cost_curve -- see chat: floor=14.84, amp=20.37, tau=19.96,
# fit to this league's real (0->35.2), (6->29.9), (8->28.5) PPG history).
OPP_FLOOR, OPP_AMP, OPP_TAU = 14.84, 20.37, 19.96


def opportunity_cost(keepers_per_team: float) -> float:
    """What you'd realistically get back with a fresh draft pick instead of
    keeping this roster spot, as a function of how many keepers exist
    league-wide. NOT positional replacement level (VBD) -- this doesn't
    touch position at all, it's purely about how much talent the keeper
    rule holds out of the draft pool overall."""
    return OPP_FLOOR + OPP_AMP * np.exp(-keepers_per_team / OPP_TAU)


def decompose(row: pd.DataFrame):
    """Split the frozen model's score into (non_age_score, age_peak, age_up, age_down)."""
    idx = 0
    non_age = 0.0
    for name, col in RAMP_FEATURES:
        lo_x, span, weight = params[idx], params[idx + 1], params[idx + 2]
        idx += 3
        hi_x = lo_x + span
        x = float(row[col].iloc[0])
        frac = np.clip((x - lo_x) / max(hi_x - lo_x, 1e-6), 0.0, 1.0)
        non_age += weight * frac
    peak_age, slope_up, slope_down = params[idx], params[idx + 1], params[idx + 2]
    idx += 3
    team_changed_w, traded_w = params[idx], params[idx + 1]
    idx += 2
    non_age += team_changed_w * float(row["TEAM_CHANGED_OFFSEASON"].iloc[0])
    non_age += traded_w * float(row["TRADED_MIDSEASON"].iloc[0])
    non_age += params[idx]  # intercept
    return non_age, peak_age, slope_up, slope_down


def trajectory(row: pd.DataFrame, years=12):
    non_age, peak_age, slope_up, slope_down = decompose(row)
    age0 = float(row["AGE"].iloc[0])
    vals = []
    for k in range(years):
        age = age0 + k
        diff = age - peak_age
        age_contrib = slope_up * diff if diff <= 0 else slope_down * diff
        vals.append(non_age + age_contrib)
    return vals


def value_over_opportunity_cost(row: pd.DataFrame, keepers_per_team: float, years=7):
    # years=7: a fixed, honestly-labeled horizon cap (Tommy's call, after the
    # true "integrate until the trajectory naturally crosses below
    # opportunity cost" approach broke down -- the fitted age-decline slope
    # was only ever validated one season ahead and produced absurd 20+ year
    # runouts when extrapolated further. This still counts fewer years for
    # anyone whose trajectory crosses below opportunity cost before 7 years,
    # it just doesn't trust the linear extrapolation past 7 either way.
    opp = opportunity_cost(keepers_per_team)
    traj = trajectory(row, years)
    total = 0.0
    years_counted = 0
    for v in traj:
        if v < opp:
            break
        total += v - opp
        years_counted += 1
    return total, years_counted, opp, traj


if __name__ == "__main__":
    current = pd.read_csv(ROOT / "data" / "current_season_for_prediction_v2.csv")
    for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
        base_col = col.replace("_PREV", "")
        current[col] = current[col].fillna(current[base_col])

    print(f"Opportunity cost at K=3 (this year): {opportunity_cost(3):.2f} PPG")
    print(f"Opportunity cost at K=5 (future plan): {opportunity_cost(5):.2f} PPG\n")

    def normalize_name(n: str) -> str:
        n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
        n = re.sub(r"[^a-z ]", "", n.lower())
        return re.sub(r"\s+", " ", n).strip()

    current["_NORM"] = current["PLAYER_NAME"].apply(normalize_name)

    for name in ["Cooper Flagg", "Nikola Jokic", "Jayson Tatum", "Evan Mobley"]:
        matches = current[current["_NORM"] == normalize_name(name)]
        if matches.empty:
            print(f"{name}: not found")
            continue
        row = matches.iloc[[0]]
        for K, label in [(3, "K=3"), (5, "K=5")]:
            total, yrs, opp, traj = value_over_opportunity_cost(row, K)
            print(f"{name:16s} age={row['AGE'].iloc[0]:.0f}  {label}: value={total:6.1f} over {yrs} years "
                  f"(opp_cost={opp:.1f})  trajectory[:5]={[round(v,1) for v in traj[:5]]}")
        print()
