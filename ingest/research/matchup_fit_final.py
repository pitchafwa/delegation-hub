"""Final production fit for the matchup adjustment (RESEARCH_gamelevel.md), using ALL available history (not held out -- the held-out numbers are
already in the research doc). Rebuilt to use team_defense_shared.py -- the SAME pace/defense math build_matchup_context.py uses live in production
(player-level box scores only; no team-level OREB/DREB split available live), so training and production are on identical footing.
Writes matchup_model.json: a plain linear adjustment (in fantasy points) from opponent defense, positional defense, opponent missing production and
opponent back-to-back, centered on league-average so a neutral matchup adds ~0.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F
import team_defense_shared as TD

sys.stdout.reconfigure(encoding="utf-8")
GL = F.D.parent / "gamelevel"

# ---------------- opponent lookup for every historical game, from the real team box data already pulled (team_box_features.pkl) ----------------
tb = pd.read_pickle(GL / "team_box_features.pkl")
FIX = {"NOH": "NOP", "NJN": "BKN", "CHO": "CHA", "SEA": "OKC"}
tb["team"] = tb.TEAM_ABBREVIATION.map(lambda t: FIX.get(t, t))
tb["date_s"] = tb.GAME_DATE.dt.strftime("%Y-%m-%d")
opp_by_team_date = tb.set_index(["team", "date_s"]).opp.to_dict()
b2b_by_team_date = tb.set_index(["team", "date_s"]).b2b.to_dict()

g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
g = g[g.season != "2025-26"].copy()
g["fp"] = F.fp(g)
g["date_s"] = g.date.dt.strftime("%Y-%m-%d")
g["DATE_D"] = g.date.dt.date
g["opp"] = [opp_by_team_date.get((t, d)) for t, d in zip(g.team, g.date_s)]
g["b2b_own"] = [b2b_by_team_date.get((t, d)) for t, d in zip(g.team, g.date_s)]
g["b2b_opp_flag"] = [b2b_by_team_date.get((o, d)) for o, d in zip(g.opp, g.date_s)]

seasons = sorted(g.season.unique())
g = g.sort_values(["pid", "season", "date"]).reset_index(drop=True)
gk = g.groupby(["pid", "season"])
g["n_prior"] = gk.cumcount()
g["min_td"] = gk["min"].transform(lambda s: s.shift(1).expanding().mean())
g["fp_per_min_td"] = gk.apply(lambda d: (d["fp"] / d["min"]).shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)

# ---------------- per-season, to-date team defense/pace/positional-defense, using ONLY the shared (player-box) math ----------------
def to_date_team_tables(season):
    """returns a dict of DATE (str) -> team_defense_table computed from all games of that season STRICTLY BEFORE that date (expanding, no leakage)."""
    s = g[g.season == season]
    dates = sorted(s.date_s.unique())
    pos_lookup = TD.player_position_mix(s, min_gp=5)     # season-long position mix (role doesn't change mid-season; not a production-relevant leak)
    tables = {}
    for d in dates:
        prior_rows = s[s.date_s < d]
        if len(prior_rows) < 200:                        # too early in the season for a stable table
            continue
        tables[d] = TD.team_defense_table(prior_rows, pos_lookup)
    return tables


all_tables = {}
for sea in seasons:
    print("building to-date defense tables for", sea, "...")
    all_tables[sea] = to_date_team_tables(sea)


def lookup(season, date_s, team, col):
    tbl = all_tables.get(season, {}).get(date_s)
    if tbl is None or team not in tbl.index:
        return np.nan
    return tbl.loc[team, col]


g["drtg_td_opp"] = [lookup(se, d, o, "drtg") for se, d, o in zip(g.season, g.date_s, g.opp)]
g["pace_td_own"] = [lookup(se, d, t, "pace") for se, d, t in zip(g.season, g.date_s, g.team)]
g["pace_td_opp"] = [lookup(se, d, o, "pace") for se, d, o in zip(g.season, g.date_s, g.opp)]


def posdef_for_row(se, d, o, pC, pF, pG):
    tbl = all_tables.get(se, {}).get(d)
    if tbl is None or o not in tbl.index:
        return np.nan
    r = tbl.loc[o]
    return pC * r.fpC_pg + pF * r.fpF_pg + pG * r.fpG_pg


POS = pd.read_pickle(GL / "player_pos.pkl")
g = g.join(POS, on=["pid", "season"])
g["opp_pos_def_td"] = [posdef_for_row(se, d, o, pC, pF, pG) for se, d, o, pC, pF, pG in zip(g.season, g.date_s, g.opp, g.pC, g.pF, g.pG)]
g["exp_pace"] = (g.pace_td_own + g.pace_td_opp) / 2

miss = pd.read_pickle(GL / "team_missing.pkl")
g = g.merge(miss.rename(columns={"team": "opp", "DATE_D": "DATE_D_m"}), left_on=["opp", "season", "DATE_D"], right_on=["opp", "season", "DATE_D_m"], how="left")

ROT = g[(g.n_prior >= 10) & (g.min_td >= 15)].dropna(
    subset=["fp_per_min_td", "pace_td_own", "pace_td_opp", "drtg_td_opp", "opp_pos_def_td", "opp_missing_fp"]).copy()
print(f"\n{len(ROT)} rotation player-games with a full trailing baseline and opponent context (shared-math version)")

# rate basis: fp per 100 "pace-proxy" possessions, using the SAME proxy as production
tb_pace_by_gid = tb.set_index("GAME_ID").poss.to_dict() if "GAME_ID" in tb.columns else {}
ROT["fp_rate100"] = ROT.fp / (ROT.exp_pace * ROT["min"] / 48.0) * 100     # to-date exp_pace as the per-player-possession denominator proxy for the RATE itself too (keeps one consistent proxy throughout)
gk2 = ROT.groupby(["pid", "season"])
ROT = ROT.sort_values(["pid", "season", "date"])
ROT["fp_rate100_td"] = ROT.groupby(["pid", "season"]).apply(lambda d: (d.fp / (d.exp_pace * d["min"] / 48.0) * 100).shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
ROT = ROT.dropna(subset=["fp_rate100_td"])
ROT["exp_player_poss"] = ROT.exp_pace * ROT.min_td / 48.0
ROT["pred_perposs_fc"] = ROT.fp_rate100_td * ROT.exp_player_poss / 100
ROT["dev"] = ROT.fp - ROT.pred_perposs_fc
ROT["b2b_opp"] = ROT.b2b_opp_flag.fillna(0)

print(f"final panel: {len(ROT)} rows")
COLS = ["drtg_td_opp", "opp_pos_def_td", "opp_missing_fp", "b2b_opp"]
LEAGUE_AVG = {c: float(ROT[c].mean()) for c in COLS}
mu, sd = ROT[COLS].mean(), ROT[COLS].std().replace(0, 1)
m = Ridge(alpha=3.0).fit((ROT[COLS] - mu) / sd, ROT.dev)
coef = dict(zip(COLS, (m.coef_ / sd.values).tolist()))
print("league averages:", {k: round(v, 3) for k, v in LEAGUE_AVG.items()})
print("coefficients (fp per raw unit):", {k: round(v, 5) for k, v in coef.items()})
ROT["adj"] = sum(coef[c] * (ROT[c] - LEAGUE_AVG[c]) for c in COLS)
print(ROT.adj.describe())
print("check: mean adjustment should be ~0 by construction:", round(ROT.adj.mean(), 4))

# quick honest re-check: leave-one-season-out RMSE, baseline vs +context, on this rebuilt (shared-math) panel
def loso():
    errs_base, errs_full = [], []
    for s_ in ROT.season.unique():
        tr, te = ROT[ROT.season != s_], ROT[ROT.season == s_]
        if len(te) < 200:
            continue
        mu_, sd_ = tr[COLS].mean(), tr[COLS].std().replace(0, 1)
        mm = Ridge(alpha=3.0).fit((tr[COLS] - mu_) / sd_, tr.dev)
        pred = mm.predict((te[COLS] - mu_) / sd_)
        errs_full.append((te.fp - (te.pred_perposs_fc + pred)).values)
        errs_base.append((te.fp - te.pred_perposs_fc).values)
    return np.sqrt(np.mean(np.concatenate(errs_base) ** 2)), np.sqrt(np.mean(np.concatenate(errs_full) ** 2))


rb, rf = loso()
print(f"\nheld-out (leave-one-season-out) RMSE, rebuilt shared-math panel: baseline {rb:.4f}  +context {rf:.4f}")

model = {
    "note": "fp adjustment = sum(coef[c] * (value[c] - league_avg[c])); a fully neutral matchup (opponent at league-average defense/health, no b2b) "
            "gives 0 by construction. Fit on 2010-11..2024-25 using ONLY player-level box-score math (no team OREB/DREB split), matching what "
            "build_matchup_context.py can compute live. Rides on top of the site's own baseline (ESPN/DELCO) -- NOT a standalone projection.",
    "coef": coef, "league_avg": LEAGUE_AVG, "cap": 2.5,
}
json.dump(model, open(Path(__file__).resolve().parent / "matchup_model.json", "w"), indent=1)
print("\nwrote matchup_model.json")
