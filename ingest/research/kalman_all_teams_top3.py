"""Apply the validated Kalman composite (9 stats + MIN, real career history
through today) to every rostered player in the league, project forward to
the start of next season, and rank teams by top-3 combined value.
"""
import json
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

ROOT = Path(__file__).resolve().parent
STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]

df = pd.read_csv(ROOT / "data" / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

TODAY = pd.Timestamp("2026-09-18")
NEXT_SEASON_START = pd.Timestamp("2026-10-20")  # approx real NBA season-opener timing

player_ids = df["PLAYER_ID"].to_numpy()
days = df["DAYS_SINCE_LAST"].to_numpy(dtype=float)
age = df["AGE_AT_GAME"].to_numpy(dtype=float)
minutes = df["MIN"].to_numpy(dtype=float)

last_game_idx = df.groupby("PLAYER_ID").tail(1).index
last_game_date = df.loc[last_game_idx].set_index("PLAYER_ID")["GAME_DATE"]
last_game_age = df.loc[last_game_idx].set_index("PLAYER_ID")["AGE_AT_GAME"]
days_to_next_season = (NEXT_SEASON_START - last_game_date).dt.days.clip(lower=0)

projections = pd.DataFrame(index=last_game_date.index)
for stat in STATS:
    with open(ROOT / "data" / f"kalman_fit_{stat}.json") as f:
        fit = json.load(f)
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
    projections[stat] = end_state + daily_slope * days_to_next_season
    print(f"  {stat} projected to next season.")

# TD3 placeholder: most recent season's TD3-per-game rate (same known simplification as before).
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
latest = season_base.sort_values("SEASON_YEAR").groupby("PLAYER_ID").tail(1).set_index("PLAYER_ID")
projections["TD3_PG"] = (latest["TD3"] / latest["GP"].replace(0, np.nan)).reindex(projections.index).fillna(0.0)

proj_ppg = (
    projections["PTS"] * projections["MIN"]
    + 1.5 * projections["REB"] * projections["MIN"]
    + 2 * projections["AST"] * projections["MIN"]
    + 3 * projections["STL"] * projections["MIN"]
    + 3 * projections["BLK"] * projections["MIN"]
    + projections["FG3M"] * projections["MIN"]
    + 2 * projections["FTM"] * projections["MIN"]
    - projections["FTA"] * projections["MIN"]
    - projections["TOV"] * projections["MIN"]
    + 3 * projections["TD3_PG"]
).clip(lower=0)  # a raw score floor -- same spirit as flagging negative scores as nonsensical before

player_values = proj_ppg.rename("kalman_value").reset_index().rename(columns={"index": "PLAYER_ID"})
player_values.columns = ["PLAYER_ID", "kalman_value"]

# --- Match every real roster to these player IDs (stable-ID crosswalk first, validated name fallback) ---
crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

names = df.groupby("PLAYER_ID")["PLAYER_NAME"].last()


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


norm_to_id = {normalize_name(n): pid for pid, n in names.items()}
value_by_id = player_values.set_index("PLAYER_ID")["kalman_value"]

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)

rows = []
unmatched = []
for team in league.teams:
    for p in team.roster:
        nba_id = espnid_to_nbaid.get(p.playerId)
        if nba_id is not None and not pd.isna(nba_id) and int(nba_id) in value_by_id.index:
            pid = int(nba_id)
        else:
            pid = norm_to_id.get(normalize_name(p.name))
        if pid is None or pid not in value_by_id.index:
            unmatched.append((team.team_name.strip(), p.name))
            continue
        rows.append({"team_id": team.team_id, "team_name": team.team_name.strip(), "player": p.name,
                     "value": round(float(value_by_id[pid]), 1)})

result = pd.DataFrame(rows)
print(f"\nScored {len(result)} players, {len(unmatched)} unmatched: {unmatched}\n")

top3 = result.sort_values(["team_id", "value"], ascending=[True, False]).groupby("team_id").head(3)
foundation = top3.groupby("team_id")["value"].sum().rename("foundation").reset_index()
foundation = foundation.merge(result[["team_id", "team_name"]].drop_duplicates(), on="team_id")
foundation = foundation.sort_values("foundation", ascending=False).reset_index(drop=True)
foundation.index = foundation.index + 1

print("=== Team foundations, NEW Kalman composite, top-3 players ===\n")
for rank, row in foundation.iterrows():
    tid = row["team_id"]
    players = top3[top3["team_id"] == tid].sort_values("value", ascending=False)
    plist = ", ".join(f"{r.player} ({r.value:.0f})" for r in players.itertuples())
    marker = "  <-- YOUR TEAM" if tid == 12 else ""
    print(f"{rank:2d}. {row['team_name']:32s} foundation={row['foundation']:6.1f}{marker}")
    print(f"      {plist}\n")

foundation.to_csv(ROOT / "data" / "kalman_team_foundations.csv", index=False)
