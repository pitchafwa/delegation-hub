"""Apply the rebuilt VOR (new Kalman engine) to Tommy's real roster and
every other team in the league.
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config
from kalman_vor import value_over_opportunity_cost, K_THIS_YEAR, df as game_df, opportunity_cost

ROOT = Path(__file__).resolve().parent

crosswalk = pd.read_csv(ROOT / "data" / "espn_nba_id_crosswalk.csv", encoding="latin1").dropna(subset=["ESPNID"])
crosswalk["ESPNID"] = crosswalk["ESPNID"].astype(int)
espnid_to_nbaid = dict(zip(crosswalk["ESPNID"], crosswalk["NBAID"]))

names = game_df.groupby("PLAYER_ID")["PLAYER_NAME"].last()


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


norm_to_id = {normalize_name(n): pid for pid, n in names.items()}
valid_ids = set(names.index)

league = League(league_id=config.LEAGUE_ID, year=config.SEASON, espn_s2=config.ESPN_S2, swid=config.SWID)

rows = []
unmatched = []
for team in league.teams:
    for p in team.roster:
        nba_id = espnid_to_nbaid.get(p.playerId)
        if nba_id is not None and not pd.isna(nba_id) and int(nba_id) in valid_ids:
            pid = int(nba_id)
        else:
            pid = norm_to_id.get(normalize_name(p.name))
        if pid is None or pid not in valid_ids:
            unmatched.append((team.team_name.strip(), p.name))
            continue
        total, yrs, traj = value_over_opportunity_cost(pid, K_THIS_YEAR)
        if total is None:
            unmatched.append((team.team_name.strip(), p.name))
            continue
        rows.append({"team_id": team.team_id, "team_name": team.team_name.strip(),
                     "player": p.name, "vor": total, "years_above": yrs})

result = pd.DataFrame(rows)
print(f"Opportunity cost at K={K_THIS_YEAR}: {opportunity_cost(K_THIS_YEAR):.1f} PPG")
print(f"Scored {len(result)} players, {len(unmatched)} unmatched: {unmatched}\n")

pd.set_option("display.width", 140)
print("=== Your roster (team_id 12), ranked by rebuilt VOR ===\n")
mine = result[result["team_id"] == 12].sort_values("vor", ascending=False)
print(mine[["player", "vor", "years_above"]].to_string(index=False))

top3 = result.sort_values(["team_id", "vor"], ascending=[True, False]).groupby("team_id").head(3)
foundation = top3.groupby("team_id")["vor"].sum().rename("foundation").reset_index()
foundation = foundation.merge(result[["team_id", "team_name"]].drop_duplicates(), on="team_id")
foundation = foundation.sort_values("foundation", ascending=False).reset_index(drop=True)
foundation.index = foundation.index + 1

print("\n=== Team foundations, rebuilt VOR (K=3), top-3 players ===\n")
for rank, row in foundation.iterrows():
    tid = row["team_id"]
    players = top3[top3["team_id"] == tid].sort_values("vor", ascending=False)
    plist = ", ".join(f"{r.player} ({r.vor:.0f})" for r in players.itertuples())
    marker = "  <-- YOUR TEAM" if tid == 12 else ""
    print(f"{rank:2d}. {row['team_name']:32s} foundation={row['foundation']:6.1f}{marker}")
    print(f"      {plist}\n")

result.to_csv(ROOT / "data" / "kalman_vor_all_players.csv", index=False)
foundation.to_csv(ROOT / "data" / "kalman_vor_team_foundations.csv", index=False)
