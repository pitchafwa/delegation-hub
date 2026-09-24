"""Mandatory sanity gate before any refit result goes near production:
simulate a representative young quality player's full BLENDED fantasy
trajectory (all 9 stats combined, exact same math as build_trajectory in
kalman_vor.py / project_prospect_trajectory.py) and eyeball whether it's
plausible in absolute terms. A metric score alone already proved it can
hide a badly miscalibrated result (PTS's Spearman-optimized curve produced
a real points-per-game trajectory crashing from 10.1 to 1.7 by age 27).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]

# representative starting state: a real, decent young quality player's
# rough per-minute rates + minutes at age 20 (ballpark realistic values)
START_STATE = {
    "PTS": 0.42, "REB": 0.18, "AST": 0.09, "BLK": 0.022, "TOV": 0.06,
    "FG3M": 0.04, "FTM": 0.07, "FTA": 0.10, "MIN": 24.0,
}
START_AGE = 20.0
HORIZON_YEARS = 8


def fantasy_ppg(state):
    return (
        state["PTS"] * state["MIN"] + 1.5 * state["REB"] * state["MIN"] + 2 * state["AST"] * state["MIN"]
        + 3 * state["BLK"] * state["MIN"] + state["FG3M"] * state["MIN"]
        + 2 * state["FTM"] * state["MIN"] - state["FTA"] * state["MIN"] - state["TOV"] * state["MIN"]
    )


def load_params(suffix):
    params = {}
    for s in KALMAN_STATS:
        path = ROOT / "data" / f"kalman_fit_{s}_{suffix}.json"
        if not path.exists():
            return None
        params[s] = json.loads(path.read_text())["params"]
    return params


def simulate(params):
    state = dict(START_STATE)
    rows = []
    for k in range(HORIZON_YEARS):
        ppg = fantasy_ppg(state)
        rows.append((START_AGE + k, dict(state), ppg))
        for s in KALMAN_STATS:
            Q, R, peak_age, slope_up, slope_down = params[s]
            diff = (START_AGE + k) - peak_age
            slope = slope_up if diff <= 0 else slope_down
            state[s] = max(state[s] + slope, 0.0)
    return rows


if __name__ == "__main__":
    suffix = sys.argv[1] if len(sys.argv) > 1 else "calibrated"
    params = load_params(suffix)
    if params is None:
        print(f"Missing one or more kalman_fit_<STAT>_{suffix}.json files -- not all 9 stats fit yet.")
        sys.exit(1)

    rows = simulate(params)
    print(f"Blended trajectory sanity check ('{suffix}' params), starting age {START_AGE}:\n")
    print(f"{'age':>5} {'PTS':>7} {'REB':>7} {'AST':>7} {'MIN':>7} {'PPG':>8}")
    for age, state, ppg in rows:
        print(f"{age:5.0f} {state['PTS']:7.3f} {state['REB']:7.3f} {state['AST']:7.3f} {state['MIN']:7.2f} {ppg:8.2f}")

    ppgs = [r[2] for r in rows]
    net_change = ppgs[-1] - ppgs[0]
    peak_idx = int(np.argmax(ppgs))
    print(f"\nNet change over {HORIZON_YEARS} years: {net_change:+.1f} PPG "
          f"({'rises then falls' if 0 < peak_idx < HORIZON_YEARS-1 else 'monotonic'}, peak at age {START_AGE+peak_idx:.0f})")
    if ppgs[3] < ppgs[0] * 0.5:
        print("*** FAILS sanity check: real quality player PPG should not halve within 3 years of debut. ***")
    elif ppgs[1] < ppgs[0]:
        print("*** WARNING: PPG declines in year 1 for a 20-year-old -- check this is intentional/real. ***")
    else:
        print("Passes basic sanity check (no early collapse, no implausible year-1 decline).")
