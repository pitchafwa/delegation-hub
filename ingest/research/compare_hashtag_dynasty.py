"""Compare our model's dynasty ranking against hashtagbasketball's public
dynasty rankings, for every named player they list. Uses the same final
frozen approach (9 Kalman stats + flat STL rate + TD3 placeholder), summed
over the 7-year trajectory -- opportunity cost is skipped here since it's
specific to Tommy's league's keeper count and barely changes ORDER for
players who stay above it the whole horizon anyway (only relevant at the
bottom of a roster, not for a general top-100 dynasty comparison).
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from kalman_engine import run_filter_all_players

ROOT = Path(__file__).resolve().parent
KALMAN_STATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]
HORIZON_YEARS = 7

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

NEXT_SEASON_START = pd.Timestamp("2026-10-20")

player_ids = df["PLAYER_ID"].to_numpy()
days = df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = df["MIN"].to_numpy(dtype=float)

last_game_idx = df.groupby("PLAYER_ID").tail(1).index
last_game_date = df.loc[last_game_idx].set_index("PLAYER_ID")["GAME_DATE"]
last_game_age = df.loc[last_game_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
days_to_next_season = (NEXT_SEASON_START - last_game_date).dt.days.clip(lower=0)

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
    x0 = float(minutes.sum()) / len(minutes) if is_min else float(obs.sum()) / max(float(gain.sum()), 1.0)
    posterior = run_filter_all_players(player_ids, days, age, gain, obs, Q, R, peak_age, slope_up, slope_down, x0)
    df["_post"] = posterior
    end_state = df.loc[last_game_idx].set_index("PLAYER_ID")["_post"]
    diff = last_game_age - peak_age
    daily_slope = np.where(diff <= 0, slope_up, slope_down) / 365.0
    year0[stat] = (end_state + daily_slope * days_to_next_season).clip(lower=0)
    print(f"  {stat} projected.")

stl_rate = df.groupby("PLAYER_ID").apply(lambda d: d["STL"].sum() / max(d["MIN"].sum(), 1.0), include_groups=False)
year0["STL"] = stl_rate.reindex(year0.index).fillna(0.0)

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
latest = season_base.sort_values("SEASON_YEAR").groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
td3_pg = (latest["TD3"] / latest["GP"].replace(0, np.nan)).reindex(year0.index).fillna(0.0)


def fantasy_ppg(row):
    return (
        row["PTS"] * row["MIN"] + 1.5 * row["REB"] * row["MIN"] + 2 * row["AST"] * row["MIN"]
        + 3 * row["STL"] * row["MIN"] + 3 * row["BLK"] * row["MIN"] + row["FG3M"] * row["MIN"]
        + 2 * row["FTM"] * row["MIN"] - row["FTA"] * row["MIN"] - row["TOV"] * row["MIN"]
    )


def dynasty_score(player_id):
    if player_id not in year0.index:
        return None, None
    age0 = float(last_game_age.get(player_id, np.nan))
    if np.isnan(age0):
        return None, None
    state = {s: float(year0.loc[player_id, s]) for s in KALMAN_STATS + ["STL"]}
    total = 0.0
    year0_ppg = None
    for k in range(HORIZON_YEARS):
        ppg = fantasy_ppg(state) + 3 * float(td3_pg.get(player_id, 0.0))
        ppg = max(ppg, 0.0)
        if k == 0:
            year0_ppg = ppg
        total += ppg
        for s in KALMAN_STATS:
            Q, R, peak_age, slope_up, slope_down = params_by_stat[s]
            diff = (age0 + k) - peak_age
            slope = slope_up if diff <= 0 else slope_down
            state[s] = max(state[s] + slope, 0.0)
    return total, year0_ppg


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


names = df.groupby("PLAYER_ID")["PLAYER_NAME"].last()
norm_to_id = {normalize_name(n): pid for pid, n in names.items()}

HASHTAG_RANKINGS = [
    (1, "Victor Wembanyama"), (2, "Shai Gilgeous-Alexander"), (3, "Luka Doncic"),
    (4, "Nikola Jokic"), (5, "Cade Cunningham"), (6, "Cooper Flagg"),
    (7, "Anthony Edwards"), (8, "Tyrese Maxey"), (9, "Cameron Boozer"),
    (10, "Jalen Johnson"), (11, "Jayson Tatum"), (12, "Scottie Barnes"),
    (13, "Tyrese Haliburton"), (14, "Chet Holmgren"), (15, "Evan Mobley"),
    (16, "Alperen Sengun"), (17, "Dylan Harper"), (18, "Donovan Mitchell"),
    (19, "Jalen Williams"), (20, "Darryn Peterson"), (21, "Giannis Antetokounmpo"),
    (22, "Trey Murphy III"), (23, "Amen Thompson"), (24, "Austin Reaves"),
    (25, "Trae Young"), (26, "Devin Booker"), (27, "Karl-Anthony Towns"),
    (28, "Kon Knueppel"), (29, "Franz Wagner"), (30, "Alex Sarr"),
    (31, "Jamal Murray"), (32, "Josh Giddey"), (33, "Jalen Duren"),
    (34, "Caleb Wilson"), (35, "Jaren Jackson Jr."), (36, "Deni Avdija"),
    (37, "AJ Dybantsa"), (38, "Jalen Brunson"), (40, "LaMelo Ball"),
    (41, "Darius Garland"), (42, "Stephon Castle"), (43, "Matas Buzelis"),
    (45, "Brandon Miller"), (46, "Bam Adebayo"), (47, "Donovan Clingan"),
    (48, "Paolo Banchero"), (49, "VJ Edgecombe"), (50, "Jaylen Brown"),
    (51, "Keyonte George"), (52, "Anthony Davis"), (53, "Tyler Herro"),
    (54, "Lauri Markkanen"), (55, "Zach Edey"), (56, "De'Aaron Fox"),
    (58, "Mikel Brown Jr."), (59, "Darius Acuff Jr."), (60, "Onyeka Okongwu"),
]

rows = []
unmatched = []
for hashtag_rank, name in HASHTAG_RANKINGS:
    pid = norm_to_id.get(normalize_name(name))
    if pid is None:
        unmatched.append(name)
        continue
    total, y0 = dynasty_score(pid)
    if total is None:
        unmatched.append(name)
        continue
    rows.append({"hashtag_rank": hashtag_rank, "player": name, "our_score": round(total, 1), "our_year0_ppg": round(y0, 1)})

result = pd.DataFrame(rows)
result["our_rank"] = result["our_score"].rank(ascending=False).astype(int)
result["rank_diff"] = result["hashtag_rank"] - result["our_rank"]
result = result.sort_values("hashtag_rank")

print(f"\nMatched {len(result)} of {len(HASHTAG_RANKINGS)} named players. Unmatched: {unmatched}\n")
pd.set_option("display.width", 140)
print(result[["hashtag_rank", "our_rank", "rank_diff", "player", "our_score", "our_year0_ppg"]].to_string(index=False))

from scipy.stats import spearmanr
rho, _ = spearmanr(result["hashtag_rank"], result["our_rank"])
print(f"\nSpearman correlation between hashtagbasketball's ranks and ours (within this matched set): {rho:.3f}")

result.to_csv(ROOT / "data" / "hashtag_comparison.csv", index=False)
