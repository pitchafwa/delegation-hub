"""Matchup adjustment: opponent defense (overall + positional), opponent missing rotation production, and opponent back-to-back status. A small
additive points-per-game adjustment, fit and validated in RESEARCH_gamelevel.md's game-level study (241,840 real player-games, 2010-11..2024-25).
Rides on top of the site's own level (ESPN/DELCO) -- it is NOT a standalone projection, and only ever nudges by a capped, modest amount.
"""
import json
from pathlib import Path

_M = json.load(open(Path(__file__).resolve().parent / "matchup_model.json", encoding="utf-8"))
COEF, LEAGUE_AVG, CAP = _M["coef"], _M["league_avg"], _M["cap"]


def adjustment(drtg_opp=None, posdef_opp=None, missing_opp=None, b2b_opp=None):
    """fp adjustment for one player, one game, given the opponent's trailing overall defense (drtg_opp), trailing defense allowed to his position
    (posdef_opp), the opponent's missing rotation production that day (missing_opp), and whether the opponent is on a back-to-back (b2b_opp, bool).
    Any argument can be omitted (None) if unknown; only the terms provided contribute. Capped at +-CAP fp/game."""
    vals = {"drtg_td_opp": drtg_opp, "opp_pos_def_td": posdef_opp, "opp_missing_fp": missing_opp, "b2b_opp": (1.0 if b2b_opp else (0.0 if b2b_opp is not None else None))}
    adj = sum(COEF[k] * (v - LEAGUE_AVG[k]) for k, v in vals.items() if v is not None)
    return max(-CAP, min(CAP, adj))
