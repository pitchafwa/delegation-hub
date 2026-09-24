"""Rebuild the multi-year trajectory + value-over-opportunity-cost (VOR)
layer on top of the NEW validated Kalman composite, replacing the old
single-blended-curve version. Each of the 10 stats now has its OWN
separately-fit aging curve (more realistic than one curve standing in for
everything), so the year-by-year trajectory should be more principled.

Opportunity cost by keeper count is unchanged -- that was fit from real
league draft history and doesn't depend on which player-value engine
produces the underlying projections.
"""
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config
from kalman_engine import run_filter_all_players
from aging_shape import annual_slope  # validated late-career decline correction (see aging_shape.py)
from keeper_value_over_replacement import opportunity_cost

ROOT = Path(__file__).resolve().parent
# STL and TD3 are deliberately excluded from the Kalman machinery: both were
# tested with the same rigor as everything else and the fancy model
# genuinely lost to simple persistence (STL: 0.635 vs 0.651 naive; TD3: 0.347
# vs 0.492 naive). STL uses a flat career-average rate (no age evolution,
# since none was found to help); TD3 stays on the season-carried-forward
# placeholder it already had.
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]
STATS = KALMAN_STATS + ["STL"]
HORIZON_YEARS = 7
K_THIS_YEAR = 3

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

TODAY = pd.Timestamp("2026-09-18")
NEXT_SEASON_START = pd.Timestamp("2026-10-20")

player_ids = df["PLAYER_ID"].to_numpy()
days = df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = df["MIN"].to_numpy(dtype=float)

last_game_idx = df.groupby("PLAYER_ID").tail(1).index
last_game_date = df.loc[last_game_idx].set_index("PLAYER_ID")["GAME_DATE"]
last_game_age = df.loc[last_game_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
days_to_next_season = (NEXT_SEASON_START - last_game_date).dt.days.clip(lower=0)

# --- Output B: per-player starting priors (x0), replacing the single
# population-wide average previously used for every player alike. Matters
# most for players with only a handful of real games so far (a rookie's
# early-season posterior is still heavily anchored to x0); for a long-
# tenured veteran the real game history quickly dominates regardless of
# x0, so there's no downside to using their projection too when available.
# Falls back to the original population-average x0 for anyone without a
# resolvable pre-draft profile (pre-2008 draftees, no profile on record).
def _load_output_b_priors():
    try:
        rookie_df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
    except FileNotFoundError:
        return {}
    EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
    rookie_df["exp_numeric"] = rookie_df["exp"].map(EXP_MAP)
    rookie_df["exp_numeric"] = rookie_df["exp_numeric"].fillna(rookie_df["exp_numeric"].median())
    rookie_df["rec_filled"] = rookie_df["rec"].fillna(0)
    rookie_df["draft_age_filled"] = rookie_df["draft_age"].fillna(rookie_df["draft_age"].median())
    rookie_df["pick_filled"] = rookie_df["real_draft_number"].fillna(61.0)

    priors = {}
    try:
        with open(ROOT / "data" / "output_b_model.pkl", "rb") as f:
            college_artifact = pickle.load(f)
        college_rows = rookie_df[rookie_df["data_source"] == "college"]
        for _, row in college_rows.iterrows():
            x = np.array([[row.get(f, college_artifact["medians"][f]) for f in college_artifact["features"]]], dtype=float)
            x = np.where(np.isnan(x), [college_artifact["medians"][f] for f in college_artifact["features"]], x)
            proj = {}
            for stat, m in college_artifact["models"].items():
                pred = m["model"].predict(m["scaler"].transform(x))[0] if m["kind"] == "ridge" else m["model"].predict(x)[0]
                proj[stat] = max(0.0, float(pred))
            priors[row["PLAYER_ID"]] = proj
    except FileNotFoundError:
        pass

    try:
        with open(ROOT / "data" / "output_b_model_international.pkl", "rb") as f:
            intl_artifact = pickle.load(f)
        intl_raw = pd.read_csv(ROOT / "data" / "international_features.csv").rename(
            columns={"PERSON_ID": "PLAYER_ID", "game_score": "int_game_score_raw",
                     "TRB": "int_TRB_raw", "AST": "int_AST_raw", "STL": "int_STL_raw",
                     "BLK": "int_BLK_raw", "TOV": "int_TOV_raw", "PTS": "int_PTS_raw",
                     "MP": "int_MP_raw", "FG%": "int_fg_pct", "3P%": "int_three_pct", "FT%": "int_ft_pct"})
        intl_rows = rookie_df[rookie_df["data_source"] == "international"].merge(
            intl_raw[["PLAYER_ID", "int_MP_raw", "int_fg_pct", "int_three_pct", "int_ft_pct",
                      "int_TRB_raw", "int_AST_raw", "int_STL_raw", "int_BLK_raw", "int_TOV_raw",
                      "int_PTS_raw", "int_game_score_raw"]],
            on="PLAYER_ID", how="left",
        )
        for stat, col in [("PTS", "int_PTS_raw"), ("REB", "int_TRB_raw"), ("AST", "int_AST_raw"),
                           ("STL", "int_STL_raw"), ("BLK", "int_BLK_raw"), ("TOV", "int_TOV_raw")]:
            intl_rows[f"int_{stat}_per_min"] = intl_rows[col] / intl_rows["int_MP_raw"].replace(0, np.nan)
        intl_rows["int_game_score_per_min"] = intl_rows["int_game_score_raw"] / intl_rows["int_MP_raw"].replace(0, np.nan)
        for _, row in intl_rows.iterrows():
            if pd.isna(row.get("int_PTS_per_min")):
                continue
            x = np.array([[row.get(f, intl_artifact["medians"][f]) for f in intl_artifact["features"]]], dtype=float)
            x = np.where(np.isnan(x), [intl_artifact["medians"][f] for f in intl_artifact["features"]], x)
            proj = {}
            for stat, m in intl_artifact["models"].items():
                proj[stat] = max(0.0, float(m["model"].predict(m["scaler"].transform(x))[0]))
            priors[row["PLAYER_ID"]] = proj
    except FileNotFoundError:
        pass

    return priors


output_b_priors = _load_output_b_priors()
n_matched = sum(1 for pid in df["PLAYER_ID"].unique() if pid in output_b_priors)
print(f"  Output B priors available for {n_matched} of {df['PLAYER_ID'].nunique()} players in kalman_input.csv "
      f"(rest fall back to the population-average prior, as before).")

# year0[stat] = posterior projected to next season's start; params[stat] = fitted curve.
year0 = pd.DataFrame(index=last_game_date.index)
params_by_stat = {}
for stat in KALMAN_STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        fit = json.load(f)
    params_by_stat[stat] = fit["params"]
    Q, R, peak_age, slope_up, slope_down = fit["params"]
    is_min = stat == "MIN"
    obs = minutes if is_min else df[stat].to_numpy(dtype=float)
    gain = np.ones_like(minutes) if is_min else minutes
    population_x0 = float(minutes.sum()) / len(minutes) if is_min else float(obs.sum()) / max(float(gain.sum()), 1.0)

    x0_arr = np.full(len(df), population_x0)
    for pid, proj in output_b_priors.items():
        if stat in proj:
            x0_arr[player_ids == pid] = proj[stat]

    posterior = run_filter_all_players(player_ids, days, age, gain, obs, Q, R, peak_age, slope_up, slope_down, x0_arr)
    df["_post"] = posterior
    end_state = df.loc[last_game_idx].set_index("PLAYER_ID")["_post"]
    diff = last_game_age - peak_age
    daily_slope = annual_slope(stat, last_game_age, fit["params"]) / 365.0
    year0[stat] = (end_state + daily_slope * days_to_next_season).clip(lower=0)
    print(f"  {stat} year-0 projected.")

# STL: flat career per-minute rate, no age evolution (tested, no real signal found).
stl_rate = df.groupby("PLAYER_ID").apply(
    lambda d: d["STL"].sum() / max(d["MIN"].sum(), 1.0), include_groups=False
)
year0["STL"] = stl_rate.reindex(year0.index).fillna(0.0)
print("  STL year-0 set to flat career rate (no Kalman filter -- tested, didn't help).")

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
latest = season_base.sort_values("SEASON_YEAR").groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
td3_pg = (latest["TD3"] / latest["GP"].replace(0, np.nan)).reindex(year0.index).fillna(0.0)

# NOTE: an injury-history availability discount was built and tested here
# (real signal: 5x washout-risk gap between quartiles) but Tommy decided
# NOT to bake it into the actual projections -- a smooth population-average
# curve mislabeled Cooper Flagg (one normal 70/82-game rookie season) as
# real injury risk alongside Jayson Tatum (a genuine 16-game Achilles-recovery
# season), and conflated a resolved single incident with chronic risk. Kept
# as a separate, visible flag instead (see injury_risk_flag.py) so Tommy
# applies his own judgment rather than the model silently discounting for him.


def fantasy_ppg(row):
    return (
        row["PTS"] * row["MIN"] + 1.5 * row["REB"] * row["MIN"] + 2 * row["AST"] * row["MIN"]
        + 3 * row["STL"] * row["MIN"] + 3 * row["BLK"] * row["MIN"] + row["FG3M"] * row["MIN"]
        + 2 * row["FTM"] * row["MIN"] - row["FTA"] * row["MIN"] - row["TOV"] * row["MIN"]
    )


def build_trajectory(player_id):
    if player_id not in year0.index:
        return None
    age0 = float(last_game_age.get(player_id, np.nan))
    if np.isnan(age0):
        return None
    state = {s: float(year0.loc[player_id, s]) for s in STATS}
    traj = []
    for k in range(HORIZON_YEARS):
        ppg = fantasy_ppg(state) + 3 * float(td3_pg.get(player_id, 0.0))
        traj.append(max(ppg, 0.0))
        # advance one year for each Kalman-filtered stat using its own fitted
        # annual slope; STL stays flat (no age evolution -- tested, didn't help).
        for s in KALMAN_STATS:
            slope = annual_slope(s, age0 + k, params_by_stat[s])
            floor = 0.0
            state[s] = max(state[s] + slope, floor)
    return traj


def value_over_opportunity_cost(player_id, keepers_per_team, years=HORIZON_YEARS):
    traj = build_trajectory(player_id)
    if traj is None:
        return None, None, None
    opp = opportunity_cost(keepers_per_team)
    total, years_counted = 0.0, 0
    for v in traj[:years]:
        if v < opp:
            break
        total += v - opp
        years_counted += 1
    return round(total, 1), years_counted, traj


if __name__ == "__main__":
    def normalize_name(name: str) -> str:
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
        name = re.sub(r"[^a-z ]", "", name.lower())
        return re.sub(r"\s+", " ", name).strip()

    names = df.groupby("PLAYER_ID")["PLAYER_NAME"].last()
    norm_to_id = {normalize_name(n): pid for pid, n in names.items()}

    print(f"\nOpportunity cost at K={K_THIS_YEAR}: {opportunity_cost(K_THIS_YEAR):.1f} PPG\n")
    for name in ["Cooper Flagg", "Nikola Jokic", "Jayson Tatum", "Evan Mobley"]:
        pid = norm_to_id.get(normalize_name(name))
        total, yrs, traj = value_over_opportunity_cost(pid, K_THIS_YEAR)
        print(f"{name:16s} VOR={total:6.1f}  years_above={yrs}/{HORIZON_YEARS}  trajectory={[round(v,1) for v in traj]}")

    # --- full-roster export for the dashboard: every current player, not
    # just the 4 sanity-check names above ---
    rows = []
    for pid in year0.index:
        total, yrs, traj = value_over_opportunity_cost(pid, K_THIS_YEAR)
        if traj is None:
            continue
        next_season_proj = {s: round(float(year0.loc[pid, s]), 4) for s in STATS}
        rows.append({
            "PLAYER_ID": pid, "player": names.get(pid, ""), "age": round(float(last_game_age.get(pid, np.nan)), 1),
            "current_year_ppg": round(traj[0], 1), "peak_ppg": round(max(traj), 1),
            "peak_year_index": int(np.argmax(traj)), "trajectory": [round(v, 1) for v in traj],
            "vor": total, "years_above_replacement": yrs,
            "had_output_b_prior": bool(pid in output_b_priors),
            "next_season_proj": next_season_proj,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(ROOT / "data" / "current_player_trajectories.csv", index=False)
    print(f"\nWrote {len(out_df)} current players to current_player_trajectories.csv "
          f"({out_df['had_output_b_prior'].sum()} used an Output B individualized prior)")
