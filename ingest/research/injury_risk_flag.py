"""Injury-risk FLAG, not a projection adjustment. Tommy's call: leave the
actual VOR numbers alone, but surface (a) how many real, significantly-
shortened seasons a player has had recently, and (b) what their VOR WOULD be
if the tested-but-rejected availability discount were applied -- so he can
apply his own subjective judgment (e.g. "Tatum's specific Achilles recovery
is a resolved one-off, the model can't know that") rather than the model
silently discounting for him.

Fixes the Cooper Flagg false positive from the smooth-average version: a
single normal 70/82-game rookie season (13% missed) is not a real signal.
Uses a real THRESHOLD instead of a smooth average -- only counts a season as
"significantly shortened" if a player missed more than 25% of it, which
correctly separates one normal rest-driven absence from a real injury year.
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

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_len = {y: (72 if y in (2020, 2021) else 82) for y in range(2010, 2027)}
season_base["season_len"] = season_base["SEASON_YEAR"].map(season_len)
season_base["missed_frac"] = 1 - (season_base["GP"] / season_base["season_len"]).clip(upper=1.0)

MAX_YEAR = season_base["SEASON_YEAR"].max()
TRAILING_YEARS = [MAX_YEAR - 3, MAX_YEAR - 2, MAX_YEAR - 1, MAX_YEAR]  # 4-year window, more context than 3
SIGNIFICANT_THRESHOLD = 0.25  # missed >25% of a season -- a real absence, not normal rest/minor tweaks

trailing = season_base[season_base["SEASON_YEAR"].isin(TRAILING_YEARS)].copy()
trailing["significant_miss"] = trailing["missed_frac"] > SIGNIFICANT_THRESHOLD

sig_count = trailing.groupby("PLAYER_ID")["significant_miss"].sum().rename("significant_missed_seasons")
seasons_available = trailing.groupby("PLAYER_ID")["SEASON_YEAR"].nunique().rename("seasons_of_history")
avg_missed_frac = trailing.groupby("PLAYER_ID")["missed_frac"].mean().rename("avg_missed_frac_4yr")

AVAIL_A, AVAIL_B, AVAIL_C = 0.88420392, 0.71142792, 0.68520518


def hypothetical_availability(injury_history):
    return np.clip(AVAIL_A - AVAIL_B * injury_history ** AVAIL_C, 0.05, 1.0)


def risk_tier(row):
    n = row["significant_missed_seasons"]
    if pd.isna(n) or n == 0:
        return "🟢 none"
    if n == 1:
        return "🟡 watch (1 significant season)"
    return "🔴 chronic (2+ significant seasons)"


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
    n_hist = seasons_available.get(pid, 0)
    avg_missed = avg_missed_frac.get(pid, 0.0)
    hyp_avail = hypothetical_availability(avg_missed)
    hyp_vor = round(real_vor * hyp_avail, 1) if real_vor is not None else None
    pct_drop = round(100 * (1 - hyp_avail), 1)
    tier = risk_tier(pd.Series({"significant_missed_seasons": n_sig}))
    rows.append({
        "player": p.name, "real_vor": real_vor, "risk_tier": tier,
        "significant_missed_seasons": int(n_sig), "seasons_of_history": int(n_hist),
        "avg_missed_frac_4yr": round(avg_missed, 2),
        "hypothetical_vor_if_discounted": hyp_vor, "hypothetical_pct_drop": pct_drop,
    })

result = pd.DataFrame(rows).sort_values("real_vor", ascending=False)
pd.set_option("display.width", 160)
print(result.to_string(index=False))
result.to_csv(ROOT / "data" / "injury_risk_flags_roster.csv", index=False)
