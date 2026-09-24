"""Build a real 'value by draft slot' curve from THIS LEAGUE's own actual
draft history (2021-2025, non-keeper picks only -- keeper-flagged picks
aren't fresh open-market value, they're re-slotted assets you already owned).

This is the basketball-keeper analog of WRPI/RUPI's draft-capital curve: it
answers "if I don't keep this player, what value would I realistically get
back by drafting someone else at the pick I'd have instead?" -- the real
opportunity-cost baseline for Layer B's keeper decision.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from espn_api.basketball import League
import config

ROOT = Path(__file__).resolve().parent
TEAM_COUNT = 12

base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
base["FANTASY_PTS"] = (
    base["PTS"] + 1.5 * base["REB"] + 2 * base["AST"] + 3 * base["STL"] + 3 * base["BLK"]
    + base["FG3M"] + 2 * base["FTM"] - base["FTA"] - base["TOV"] + 3 * base["TD3"]
)
base["FANTASY_PPG"] = base["FANTASY_PTS"] / base["GP"].replace(0, np.nan)
base["SEASON_YEAR"] = base["SEASON"].apply(lambda s: int(s.split("-")[0]))


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


base["NORM_NAME"] = base["PLAYER_NAME"].apply(normalize_name)
name_year_to_ppg = {(r.NORM_NAME, r.SEASON_YEAR): r.FANTASY_PPG for r in base.itertuples()}

rows = []
unmatched = []
for draft_year in [2021, 2022, 2023, 2024, 2025]:
    league = League(league_id=config.LEAGUE_ID, year=draft_year, espn_s2=config.ESPN_S2, swid=config.SWID)
    picks = league.draft
    # The keeper block occupies the FIRST several rounds as a placeholder
    # listing (Tommy confirmed: "not actual draft ordered results ... just a
    # list of the keepers ... then the draft commencing as planned after those
    # rounds"). The number of keeper rounds varies by year, so raw round_num
    # is NOT comparable across years -- re-base each year's real draft to
    # start at round 1, or the curve silently averages together different
    # real draft positions across years (confirmed bug, caught before this
    # was frozen anywhere).
    keeper_rounds = [p.round_num for p in picks if p.keeper_status]
    n_keeper_rounds = max(keeper_rounds) if keeper_rounds else 0
    for pick in picks:
        if pick.keeper_status:
            continue  # not fresh open-market value -- excluded from the curve
        real_round = pick.round_num - n_keeper_rounds
        overall = (real_round - 1) * TEAM_COUNT + pick.round_pick
        norm = normalize_name(pick.playerName)
        ppg = name_year_to_ppg.get((norm, draft_year))
        if ppg is None:
            unmatched.append((draft_year, pick.playerName))
            ppg = 0.0  # undrafted-into-relevance / didn't play meaningful minutes that season
        rows.append({"draft_year": draft_year, "n_keeper_rounds": n_keeper_rounds, "real_round": real_round,
                     "round_pick": pick.round_pick, "overall_pick": overall, "player": pick.playerName, "value": ppg})

df = pd.DataFrame(rows)
print(f"{len(df)} non-keeper picks across 5 drafts. {len(unmatched)} unmatched to a real season "
      f"(likely didn't play meaningful NBA minutes that season -- scored as 0, which is honest, not an error).")
if unmatched[:10]:
    print("Sample unmatched (spot check a few):", unmatched[:10])

by_pick = df.groupby("overall_pick")["value"].mean().reset_index()
print("\n=== Mean real value by overall pick number (this league's actual draft history) ===")
print(by_pick.to_string(index=False))


def power_decay(pick, k, c, p, floor):
    return floor + k * (pick + c) ** (-p)


try:
    popt, _ = curve_fit(
        power_decay, by_pick["overall_pick"], by_pick["value"],
        p0=[200, 5, 0.7, 2], bounds=([0, 0.1, 0.1, 0], [2000, 50, 3, 15]), maxfev=10000,
    )
    print(f"\nFitted power-law decay curve: k={popt[0]:.2f}, c={popt[1]:.2f}, p={popt[2]:.2f}, floor={popt[3]:.2f}")
    fitted = power_decay(by_pick["overall_pick"], *popt)
    resid = by_pick["value"] - fitted
    print(f"Residual std: {resid.std():.2f}  (mean actual value std: {by_pick['value'].std():.2f})")
    np.save(ROOT / "data" / "draft_capital_curve_params.npy", popt)
except Exception as e:
    print(f"Curve fit failed: {e}")

df.to_csv(ROOT / "data" / "draft_history_values.csv", index=False)
by_pick.to_csv(ROOT / "data" / "draft_value_by_pick.csv", index=False)
