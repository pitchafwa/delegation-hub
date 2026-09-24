"""The real replacement for the abandoned "beat draft order" ranking:
project every prospect's expected ROOKIE-SEASON stat line (Output B), then
age it forward using the SAME per-stat Kalman aging curves already
validated and used for existing NBA players (kalman_vor.py), producing a
real multi-year trajectory and VOR on the exact same scale Tommy already
uses to judge his current roster.

This doesn't try to out-rank scouts at spotting who's elite (that was
tested exhaustively and failed). It answers a different, more useful
question: "given this prospect's real college/international profile, what
does a data-grounded trajectory actually look like, apples-to-apples
against a real player I could keep instead?"
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from keeper_value_over_replacement import opportunity_cost

HORIZON_YEARS = 10
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]
STATS = KALMAN_STATS + ["STL"]

# --- per-stat aging curves, same fitted files kalman_vor.py uses for real players ---
params_by_stat = {}
for stat in KALMAN_STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        params_by_stat[stat] = json.load(f)["params"]


def fantasy_ppg(state):
    return (
        state["PTS"] * state["MIN"] + 1.5 * state["REB"] * state["MIN"] + 2 * state["AST"] * state["MIN"]
        + 3 * state["STL"] * state["MIN"] + 3 * state["BLK"] * state["MIN"] + state["FG3M"] * state["MIN"]
        + 2 * state["FTM"] * state["MIN"] - state["FTA"] * state["MIN"] - state["TOV"] * state["MIN"]
    )


def build_prospect_trajectory(output_b_projection: dict, age0: float):
    """output_b_projection: {stat: per-minute rate for the 9 counting stats,
    'MIN': per-game minutes}. age0: age at the start of the rookie season.
    TD3 has no real prior for a never-played prospect -- left at 0, a
    disclosed gap (rare stat, mostly relevant for a handful of bigs)."""
    state = {s: float(output_b_projection[s]) for s in STATS}
    state["MIN"] = float(output_b_projection["MIN"])
    traj = []
    for k in range(HORIZON_YEARS):
        ppg = fantasy_ppg(state)
        traj.append(max(ppg, 0.0))
        for s in KALMAN_STATS:
            Q, R, peak_age, slope_up, slope_down = params_by_stat[s]
            diff = (age0 + k) - peak_age
            slope = slope_up if diff <= 0 else slope_down
            state[s] = max(state[s] + slope, 0.0)
        # STL: flat rate, no age curve (same choice as kalman_vor.py -- tested, didn't help)
    return traj


def value_over_opportunity_cost(traj, keepers_per_team, years=HORIZON_YEARS):
    opp = opportunity_cost(keepers_per_team)
    total, years_counted = 0.0, 0
    for v in traj[:years]:
        if v < opp:
            break
        total += v - opp
        years_counted += 1
    return round(total, 1), years_counted


def project_college(feature_row: dict) -> dict:
    with open(ROOT / "data" / "output_b_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    x = np.array([[feature_row.get(f, artifact["medians"][f]) for f in artifact["features"]]], dtype=float)
    x = np.where(np.isnan(x), [artifact["medians"][f] for f in artifact["features"]], x)
    out = {}
    for stat, m in artifact["models"].items():
        if m["kind"] == "ridge":
            out[stat] = max(0.0, float(m["model"].predict(m["scaler"].transform(x))[0]))
        else:
            out[stat] = max(0.0, float(m["model"].predict(x)[0]))
    return out


def project_international(feature_row: dict) -> dict:
    with open(ROOT / "data" / "output_b_model_international.pkl", "rb") as f:
        artifact = pickle.load(f)
    x = np.array([[feature_row.get(f, artifact["medians"][f]) for f in artifact["features"]]], dtype=float)
    x = np.where(np.isnan(x), [artifact["medians"][f] for f in artifact["features"]], x)
    out = {}
    for stat, m in artifact["models"].items():
        out[stat] = max(0.0, float(m["model"].predict(m["scaler"].transform(x))[0]))
    return out


if __name__ == "__main__":
    K_THIS_YEAR = 3  # matches kalman_vor.py's real-player convention
    df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
    df = df[df["real_draft_year"] >= 2008].copy()

    EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
    df["exp_numeric"] = df["exp"].map(EXP_MAP)
    df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
    df["rec_filled"] = df["rec"].fillna(0)
    df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
    df["pick_filled"] = df["real_draft_number"].fillna(61.0)

    COLLEGE_FEATURES = [
        "talent_pctile", "rec_filled", "exp_numeric", "draft_age_filled", "breakout_age_filled",
        "porpag", "usg", "ts", "ortg", "obpm", "dbpm", "bpm", "stops",
        "oreb_rate", "dreb_rate", "ast_to", "ftr", "pfr",
        "WINGSPAN_PCTILE", "STANDING_REACH_PCTILE", "STANDING_VERTICAL_LEAP_PCTILE",
        "MAX_VERTICAL_LEAP_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE",
        "three_pct", "rim_pct", "mid_pct", "pick_filled",
    ]
    COLLEGE_FEATURES = [f for f in COLLEGE_FEATURES if f in df.columns]

    # international raw box-score features (built the same way as fit_output_b_international.py)
    intl_raw = pd.read_csv(ROOT / "data" / "international_features.csv").rename(
        columns={"PERSON_ID": "PLAYER_ID", "game_score": "int_game_score_raw",
                 "TRB": "int_TRB_raw", "AST": "int_AST_raw", "STL": "int_STL_raw",
                 "BLK": "int_BLK_raw", "TOV": "int_TOV_raw", "PTS": "int_PTS_raw",
                 "MP": "int_MP_raw", "FG%": "int_fg_pct", "3P%": "int_three_pct", "FT%": "int_ft_pct"}
    )
    df = df.merge(
        intl_raw[["PLAYER_ID", "int_MP_raw", "int_fg_pct", "int_three_pct", "int_ft_pct",
                   "int_TRB_raw", "int_AST_raw", "int_STL_raw", "int_BLK_raw", "int_TOV_raw",
                   "int_PTS_raw", "int_game_score_raw"]],
        on="PLAYER_ID", how="left",
    )
    for stat, col in [("PTS", "int_PTS_raw"), ("REB", "int_TRB_raw"), ("AST", "int_AST_raw"),
                       ("STL", "int_STL_raw"), ("BLK", "int_BLK_raw"), ("TOV", "int_TOV_raw")]:
        df[f"int_{stat}_per_min"] = df[col] / df["int_MP_raw"].replace(0, np.nan)
    df["int_game_score_per_min"] = df["int_game_score_raw"] / df["int_MP_raw"].replace(0, np.nan)
    INTL_FEATURES = [
        "int_PTS_per_min", "int_REB_per_min", "int_AST_per_min", "int_STL_per_min",
        "int_BLK_per_min", "int_TOV_per_min", "int_fg_pct", "int_three_pct", "int_ft_pct",
        "int_game_score_per_min", "draft_age_filled", "pick_filled",
    ]

    with open(ROOT / "data" / "output_b_model.pkl", "rb") as f:
        college_artifact = pickle.load(f)
    with open(ROOT / "data" / "output_b_model_international.pkl", "rb") as f:
        intl_artifact = pickle.load(f)

    results = []
    for i, row in df.iterrows():
        is_intl = row["data_source"] == "international"
        if is_intl:
            if pd.isna(row.get("int_PTS_per_min")):
                continue  # no real box-score row to project from
            feats = {f: row[f] for f in INTL_FEATURES}
            x = np.array([[feats.get(f, intl_artifact["medians"][f]) for f in intl_artifact["features"]]], dtype=float)
            x = np.where(np.isnan(x), [intl_artifact["medians"][f] for f in intl_artifact["features"]], x)
            proj = {}
            for stat, m in intl_artifact["models"].items():
                proj[stat] = max(0.0, float(m["model"].predict(m["scaler"].transform(x))[0]))
        else:
            feats = {f: row[f] for f in COLLEGE_FEATURES}
            x = np.array([[feats.get(f, college_artifact["medians"][f]) for f in college_artifact["features"]]], dtype=float)
            x = np.where(np.isnan(x), [college_artifact["medians"][f] for f in college_artifact["features"]], x)
            proj = {}
            for stat, m in college_artifact["models"].items():
                if m["kind"] == "ridge":
                    proj[stat] = max(0.0, float(m["model"].predict(m["scaler"].transform(x))[0]))
                else:
                    proj[stat] = max(0.0, float(m["model"].predict(x)[0]))

        age0 = float(row["draft_age_filled"])
        traj = build_prospect_trajectory(proj, age0)
        vor, years_above = value_over_opportunity_cost(traj, K_THIS_YEAR)
        results.append({
            "PLAYER_ID": row["PLAYER_ID"], "player": row["player"], "real_draft_year": row["real_draft_year"],
            "data_source": row["data_source"], "rookie_year_ppg": round(traj[0], 1),
            "peak_ppg": round(max(traj), 1), "peak_year_index": int(np.argmax(traj)),
            "trajectory": [round(v, 1) for v in traj], "vor": vor, "years_above_replacement": years_above,
            "rookie_proj": {k: round(v, 4) for k, v in proj.items()},
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(ROOT / "data" / "prospect_trajectories.csv", index=False)
    print(f"Projected {len(out_df)} prospects (college: {(out_df['data_source']=='college').sum()}, "
          f"international: {(out_df['data_source']=='international').sum()})")

    for name in ["Cooper Flagg", "Victor Wembanyama", "Luka Dončić"]:
        r = out_df[out_df["player"] == name]
        if r.empty:
            print(f"\n{name}: not found")
            continue
        r = r.iloc[0]
        print(f"\n{name}: rookie_ppg={r['rookie_year_ppg']}  peak_ppg={r['peak_ppg']} (year {r['peak_year_index']+1})  "
              f"VOR={r['vor']}  years_above={r['years_above_replacement']}")
        print(f"  trajectory: {r['trajectory']}")
