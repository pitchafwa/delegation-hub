"""v3 of the injury risk flag: uses REAL, reason-specific missed-game counts
(from the raw DNP comment field) for ALL seasons now, including the 2 most
recent (2024-25, 2025-26), pulled directly from BoxScoreTraditionalV2/V3
per-game (see pull_recent_dnp_reasons.py) since nba_api's game-log endpoints
don't expose DNP rows at all. Coach's Decision (healthy scratch) and League
Suspension are explicitly excluded from "injury" -- only genuine
injury/illness-coded DNPs count. No more unadjusted-proxy fallback.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config
from kalman_vor import value_over_opportunity_cost, K_THIS_YEAR, df as game_df

ROOT = Path(__file__).resolve().parent

# --- Real, reason-specific missed games from the raw DNP comment field ---
bulk_frames = [pd.read_csv(ROOT / "data" / "game_logs" / f"regular_season_box_scores_2010_2024_{p}.csv") for p in ["part_1", "part_2", "part_3"]]
bulk = pd.concat(bulk_frames, ignore_index=True)
bulk_std = pd.DataFrame({
    "PLAYER_ID": bulk["personId"], "comment": bulk["comment"],
    "SEASON_YEAR": bulk["season_year"].apply(lambda s: int(str(s).split("-")[0])),
})

recent = pd.read_csv(ROOT / "data" / "recent_dnp_reasons.csv")
recent_std = pd.DataFrame({
    "PLAYER_ID": recent["personId"], "comment": recent["comment"],
    "SEASON_YEAR": recent["season"].apply(lambda s: int(str(s).split("-")[0])),
})

all_dnp = pd.concat([bulk_std, recent_std], ignore_index=True)

INJURY_PATTERN = "INJUR|ILLNESS|SORE|SPRAIN|STRAIN|SURGERY|FRACTURE|TORN|ACL|MCL|CONCUSSION|HEALTH AND SAFETY|BACK SPASM|KNEE|ANKLE|HAMSTRING|GROIN|SHOULDER|WRIST|FOOT|CALF|HIP|ELBOW"
all_dnp["is_injury_dnp"] = all_dnp["comment"].fillna("").str.upper().str.contains(INJURY_PATTERN, regex=True)

injury_dnp_counts = all_dnp[all_dnp["is_injury_dnp"]].groupby(["PLAYER_ID", "SEASON_YEAR"]).size().rename("injury_dnp_games").reset_index()

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_len = {y: (72 if y in (2020, 2021) else 82) for y in range(2010, 2027)}
season_base["season_len"] = season_base["SEASON_YEAR"].map(season_len)

merged = season_base.merge(injury_dnp_counts, on=["PLAYER_ID", "SEASON_YEAR"], how="left")
HAS_REASON_DATA = set(range(2010, 2026))  # now every season 2010-11 through 2025-26 has real reason data
merged["has_reason_data"] = merged["SEASON_YEAR"].isin(HAS_REASON_DATA)
merged["injury_missed_frac"] = merged["injury_dnp_games"].fillna(0) / merged["season_len"]

MAX_YEAR = merged["SEASON_YEAR"].max()
TRAILING_YEARS = [MAX_YEAR - 3, MAX_YEAR - 2, MAX_YEAR - 1, MAX_YEAR]
SIGNIFICANT_THRESHOLD = 0.25

trailing = merged[merged["SEASON_YEAR"].isin(TRAILING_YEARS)].copy()
trailing["significant_injury_miss"] = trailing["injury_missed_frac"] > SIGNIFICANT_THRESHOLD

sig_count = trailing.groupby("PLAYER_ID")["significant_injury_miss"].sum().rename("significant_injury_seasons")
seasons_available = trailing.groupby("PLAYER_ID")["SEASON_YEAR"].nunique().rename("seasons_of_history")


def risk_tier(n):
    if pd.isna(n) or n == 0:
        return "green: none"
    if n == 1:
        return "yellow: watch (1 real injury season)"
    return "red: chronic (2+ real injury seasons)"


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


names = game_df.groupby("PLAYER_ID")["PLAYER_NAME"].last()
norm_to_id = {normalize_name(n): pid for pid, n in names.items()}
crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)
my_team = next(t for t in league.teams if t.team_id == 12)

rows = []
for p in my_team.roster:
    nba_id = espnid_to_nbaid.get(p.playerId)
    pid = int(nba_id) if (nba_id is not None and not pd.isna(nba_id)) else None
    if pid is None or pid not in names.index:
        pid = norm_to_id.get(normalize_name(p.name))
    if pid is None:
        continue
    real_vor, real_years, traj = value_over_opportunity_cost(pid, K_THIS_YEAR)
    n_sig = sig_count.get(pid, 0)
    rows.append({
        "player": p.name, "real_vor": real_vor, "risk_tier": risk_tier(n_sig),
        "significant_injury_seasons": int(n_sig), "seasons_of_history": int(seasons_available.get(pid, 0)),
    })

result = pd.DataFrame(rows).sort_values("real_vor", ascending=False)
pd.set_option("display.width", 160)
print(result.to_string(index=False))
result.to_csv(ROOT / "data" / "injury_risk_flags_roster_v2.csv", index=False)
