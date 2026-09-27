"""Team pace / defensive-rating / positional-defense-allowed, computed ONE way, shared by the offline research (gamelevel_study.py,
matchup_fit_final.py) and the live production builder (build_matchup_context.py) -- so the model is trained on exactly what production feeds it.
Built entirely from PLAYER-level game logs (form_common.load_all's unified table) + an opponent lookup, since that's all production has access to
(no team-level box scores with the OREB/DREB split live). Pace is therefore a proxy (FGA + 0.44*FTA + TOV, no offensive-rebound adjustment) --
consistently applied everywhere, which is what actually matters for a model that only needs relative ranking, not the official NBA scale.
"""
import numpy as np
import pandas as pd

import usage_flow as UF


def team_game_totals(g_with_opp):
    """g_with_opp: player-game rows with an 'opp' column already attached. Returns one row per (team, gid): pace proxy for that game."""
    box = g_with_opp.groupby(["team", "gid"]).agg(fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum")).reset_index()
    box["pace"] = box.fga + 0.44 * box.fta + box.tov
    return box


def player_position_mix(g_slice, min_gp=5):
    """g_slice: player-game rows for one 'profile window' (e.g. one season). Returns {pid: (pC, pF, pG)}."""
    prof = g_slice.groupby("pid").agg(minsum=("min", "sum"), reb=("reb", "sum"), ast=("ast", "sum"), blk=("blk", "sum"), stl=("stl", "sum"), fg3m=("fg3m", "sum"), gp=("min", "size"))
    prof = prof[prof.gp >= min_gp]
    out = {}
    for pid, r in prof.iterrows():
        m = r.minsum or 1
        out[pid] = UF.pos_probs(r.reb / m * 36, r.ast / m * 36, r.blk / m * 36, r.stl / m * 36, r.fg3m / m * 36)
    return out


def team_defense_table(g_with_opp, pos_lookup):
    """g_with_opp: player-game rows (one season, or however much history you want pooled) with 'opp' already attached.
    pos_lookup: {pid: (pC,pF,pG)}, used to split each scorer's fantasy points across the position groups he draws from.
    Returns a DataFrame indexed by team: pace, drtg (pts allowed per 100 pace-proxy-possessions), fpC_pg/fpF_pg/fpG_pg (fp allowed per game to
    each position group), n_games."""
    s = g_with_opp.dropna(subset=["opp"]).copy()
    box = team_game_totals(s)
    pace = box.groupby("team").pace.mean()
    n_games = s.groupby("team").gid.nunique()
    default = (1 / 3, 1 / 3, 1 / 3)
    s["pC"] = [pos_lookup.get(p, default)[0] for p in s.pid]
    s["pF"] = [pos_lookup.get(p, default)[1] for p in s.pid]
    s["pG"] = [pos_lookup.get(p, default)[2] for p in s.pid]
    allowed = s.groupby("opp").agg(pts_allowed=("pts", "sum"), fpC=("pC", lambda x: (x * s.loc[x.index, "fp"]).sum()),
                                    fpF=("pF", lambda x: (x * s.loc[x.index, "fp"]).sum()), fpG=("pG", lambda x: (x * s.loc[x.index, "fp"]).sum()))
    n_g = n_games.reindex(allowed.index).fillna(1).values
    pace_pg = pace.reindex(allowed.index).values
    total_poss = pace_pg * n_g
    return pd.DataFrame({"team": allowed.index, "pace": pace_pg, "drtg": np.where(total_poss > 0, allowed.pts_allowed.values / total_poss * 100, np.nan),
                         "fpC_pg": allowed.fpC.values / n_g, "fpF_pg": allowed.fpF.values / n_g, "fpG_pg": allowed.fpG.values / n_g, "n_games": n_g}).set_index("team")
