"""Rank all 12 teams by the strength of their top-3 players, using the real
frozen Layer A model + value-over-opportunity-cost (K=3, this league's real
keeper count this year) -- the same metric already applied to Tommy's roster.
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config
from keeper_value_over_replacement import value_over_opportunity_cost, opportunity_cost

ROOT = Path(__file__).resolve().parent
K_THIS_YEAR = 3

crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

current = pd.read_csv(ROOT / "data" / "current_season_for_prediction_v2.csv")
for col in ["FANTASY_PPG_PREV", "USG_PCT_PREV", "TS_PCT_PREV"]:
    current[col] = current[col].fillna(current[col.replace("_PREV", "")])


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


current["NORM_NAME"] = current["PLAYER_NAME"].apply(normalize_name)
name_to_idx = {r.NORM_NAME: i for i, r in enumerate(current.itertuples())}

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)

rows = []
unmatched = []
for team in league.teams:
    for p in team.roster:
        nba_id = espnid_to_nbaid.get(p.playerId)
        idx = None
        if nba_id is not None and not pd.isna(nba_id):
            m = current[current["PLAYER_ID"] == int(nba_id)]
            if not m.empty:
                idx = m.index[0]
        if idx is None:
            idx = name_to_idx.get(normalize_name(p.name))
        if idx is None:
            unmatched.append((team.team_name, p.name))
            continue
        row = current.loc[[idx]]
        total, yrs, opp, traj = value_over_opportunity_cost(row, K_THIS_YEAR, years=7)
        rows.append({
            "team_id": team.team_id, "team_name": team.team_name.strip(), "player": p.name,
            "age": float(row["AGE"].iloc[0]), "value": round(total, 1),
        })

df = pd.DataFrame(rows)
print(f"Opportunity cost (K={K_THIS_YEAR}): {opportunity_cost(K_THIS_YEAR):.1f} PPG")
print(f"Scored {len(df)} players, {len(unmatched)} unmatched (likely injury-thin sample this season): {unmatched}\n")

top3 = (
    df.sort_values(["team_id", "value"], ascending=[True, False])
    .groupby("team_id")
    .head(3)
)
foundation = top3.groupby("team_id")["value"].sum().rename("foundation_value").reset_index()
foundation = foundation.merge(df[["team_id", "team_name"]].drop_duplicates(), on="team_id")
foundation = foundation.sort_values("foundation_value", ascending=False).reset_index(drop=True)
foundation.index = foundation.index + 1

pd.set_option("display.width", 160)
print("=== Team foundations ranked by sum of top-3 players' value-over-opportunity-cost ===\n")
for rank, row in foundation.iterrows():
    tid = row["team_id"]
    players = top3[top3["team_id"] == tid].sort_values("value", ascending=False)
    plist = ", ".join(f"{r.player} ({r.value:.0f}, age {r.age:.0f})" for r in players.itertuples())
    marker = "  <-- YOUR TEAM" if tid == 12 else ""
    print(f"{rank:2d}. {row['team_name']:32s} foundation={row['foundation_value']:6.1f}{marker}")
    print(f"      {plist}\n")

foundation.to_csv(ROOT / "data" / "team_foundations_top3.csv", index=False)
top3.to_csv(ROOT / "data" / "team_top3_players.csv", index=False)
