"""Apply the value-over-replacement (7-year horizon, K=3 real keeper count)
framework to Tommy's actual roster -- the real output for the actual keeper
decision.
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
my_team = next(t for t in league.teams if t.team_id == 12)

K_THIS_YEAR = 3
rows = []
for p in my_team.roster:
    nba_id = espnid_to_nbaid.get(p.playerId)
    idx = None
    if nba_id is not None and not pd.isna(nba_id):
        m = current[current["PLAYER_ID"] == int(nba_id)]
        if not m.empty:
            idx = m.index[0]
    if idx is None:
        idx = name_to_idx.get(normalize_name(p.name))
    if idx is None:
        rows.append({"player": p.name, "age": None, "vor_k3": None, "years_above_repl": None})
        continue
    row = current.loc[[idx]]
    total, yrs, opp, traj = value_over_opportunity_cost(row, K_THIS_YEAR, years=7)
    rows.append({
        "player": p.name, "age": float(row["AGE"].iloc[0]),
        "current_ppg": float(row["FANTASY_PPG"].iloc[0]),
        "vor_k3": round(total, 1), "years_above_repl_of_7": yrs,
    })

df = pd.DataFrame(rows).sort_values("vor_k3", ascending=False, na_position="last")
pd.set_option("display.width", 140)
print(f"Opportunity cost at K=3: {opportunity_cost(3):.1f} PPG\n")
print(df.to_string(index=False))
df.to_csv(ROOT / "data" / "roster_vor_k3.csv", index=False)
