"""Build the unified Fantasy Hub dashboard data: current NBA players +
incoming prospects, on the SAME trajectory/VOR scale (the validated
Kalman engine + opportunity-cost framework), replacing the old prospect-only
"beat draft order" ranking view. Trajectories are shipped raw so the
dashboard can recompute VOR live for any keeper-count assumption, not just
the K=3 default baked in here.

Real data-quality fixes in this pass (from Tommy's review of v1):
- Current players filtered to ROSTER_STATUS==1 (real active roster flag) --
  v1 included retired players (Shaquille O'Neal at 1.9 PPG) with no filter.
- Prospects filtered to (a) not already a current player -- v1 had 963 of
  1042 "prospects" duplicating an already-active player with a stale,
  lower pre-debut projection instead of their real current numbers -- and
  (b) a real recency cutoff (draft year >= 2025), since a decade-old
  draftee who never reached the NBA has no real fantasy relevance today.
- Real ESPN multi-position eligibility (pull_espn_positions.py) instead of
  NBA Stats' single-position field, which undersells real flexibility.
- Team logos (ESPN's public CDN) + real, standing injury-risk tier
  (generalized from injury_risk_flag.py's single-team version).
- Calendar-year trajectory labels instead of unlabeled "Y1..Y7".
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT.parent.parent / "dashboard" / "hub_data.json"

OPP_FLOOR, OPP_AMP, OPP_TAU = 14.84, 20.37, 19.96
HORIZON_YEARS = 7
CURRENT_SEASON_END_YEAR = 2027  # matches kalman_vor.py's NEXT_SEASON_START = 2026-10-20 (the 2026-27 season)
PROSPECT_RECENCY_CUTOFF = 2025  # drop draft classes older than this that never became a current player

# ESPN's team-logo CDN slug differs from the NBA API's own abbreviation for
# a handful of teams -- explicit map rather than guessing.
ESPN_LOGO_SLUG = {
    "GSW": "gs", "NOP": "no", "NYK": "ny", "SAS": "sa", "UTA": "utah", "WAS": "wsh",
}


def logo_url(abbr):
    if not abbr or pd.isna(abbr):
        return None
    slug = ESPN_LOGO_SLUG.get(abbr, abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png"


def round_or_none(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), nd)


bio = pd.read_csv(ROOT / "data" / "player_bio.csv")


def parse_height(h):
    if not isinstance(h, str) or "-" not in h:
        return None
    try:
        ft, inch = h.split("-")
        return int(ft) * 12 + int(inch)
    except ValueError:
        return None


bio["height_in"] = bio["HEIGHT"].apply(parse_height)

espn_pos = pd.read_csv(ROOT / "data" / "espn_positions.csv")

# --- real, standing injury-risk tier for EVERY player (generalized from
# injury_risk_flag.py, which only computed this for Tommy's own roster) ---
season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_len = {y: (72 if y in (2020, 2021) else 82) for y in range(2010, 2027)}
season_base["season_len"] = season_base["SEASON_YEAR"].map(season_len)
season_base["missed_frac"] = 1 - (season_base["GP"] / season_base["season_len"]).clip(upper=1.0)
MAX_YEAR = season_base["SEASON_YEAR"].max()
TRAILING_YEARS = [MAX_YEAR - 3, MAX_YEAR - 2, MAX_YEAR - 1, MAX_YEAR]
trailing = season_base[season_base["SEASON_YEAR"].isin(TRAILING_YEARS)].copy()
trailing["significant_miss"] = trailing["missed_frac"] > 0.25
sig_count = trailing.groupby("PLAYER_ID")["significant_miss"].sum()


def risk_tier(pid):
    n = sig_count.get(pid, 0)
    if n == 0:
        return "none"
    if n == 1:
        return "watch"
    return "chronic"


# ---------------------------------------------------------------------------
# real season-by-season history (up to the last 5 real completed seasons),
# same league scoring formula as the projection side, so a viewer can see
# the actual trailing trend a projection continues from -- catches cases
# like a smooth aging curve implying a "future peak" for a player whose
# real recent seasons already show clear decline.
# ---------------------------------------------------------------------------
HISTORY_YEARS = 5
season_base["fantasy_pg"] = (
    season_base["PTS"] + 1.5 * season_base["REB"] + 2 * season_base["AST"] + 3 * season_base["STL"]
    + 3 * season_base["BLK"] + season_base["FG3M"] + 2 * season_base["FTM"] - season_base["FTA"]
    - season_base["TOV"] + 3 * season_base["TD3"]
) / season_base["GP"].replace(0, np.nan)
season_base["season_end_year"] = season_base["SEASON_YEAR"] + 1
history_by_player = {}
for pid, g in season_base.sort_values("SEASON_YEAR").groupby("PLAYER_ID"):
    tail = g.tail(HISTORY_YEARS)
    history_by_player[pid] = {
        "years": tail["season_end_year"].tolist(),
        "values": [round_or_none(v, 1) for v in tail["fantasy_pg"]],
    }
print(f"Real season history built for {len(history_by_player)} players (up to {HISTORY_YEARS} seasons each).", flush=True)

# ---------------------------------------------------------------------------
# current NBA players -- filtered to real, active roster status
# ---------------------------------------------------------------------------
current = pd.read_csv(ROOT / "data" / "current_player_trajectories.csv")
current["trajectory"] = current["trajectory"].apply(ast.literal_eval)
current["next_season_proj"] = current["next_season_proj"].apply(ast.literal_eval)
n_before = len(current)
current = current.merge(
    bio[["PERSON_ID", "POSITION", "TEAM_ABBREVIATION", "ROSTER_STATUS"]].rename(columns={"PERSON_ID": "PLAYER_ID"}),
    on="PLAYER_ID", how="left",
)
current = current[current["ROSTER_STATUS"] == 1.0].copy()
print(f"Current players: {n_before} -> {len(current)} after filtering to real active roster status "
      f"(drops retired/inactive players like Shaquille O'Neal)", flush=True)
current = current.merge(espn_pos[["PLAYER_ID", "espn_position", "injury_status"]], on="PLAYER_ID", how="left")
# breakout-candidate probabilities (breakout_model.py) -- only players below the
# quality line last season are scored, everyone else stays null
import datetime as _dt
LEDGER_CSV = ROOT / "data" / "breakout_ledger" / "2026-27.csv"
LEDGER_META = ROOT / "data" / "breakout_ledger" / "2026-27.json"
OPENER = _dt.date(2026, 10, 20)
frozen = LEDGER_CSV.exists()
breakouts = pd.read_csv(LEDGER_CSV if frozen else ROOT / "data" / "breakout_candidates.csv")[
    ["PLAYER_ID", "tier", "p_break", "p_baseline", "p_bust", "fpg_last", "proj_fpg", "proj_lo", "proj_hi", "proj_gp", "adp",
     "edge_fpg", "edge_total", "team_next", "why"]]
_bval = json.loads((ROOT / "data" / "breakout_validation.json").read_text(encoding="utf-8"))
DELTA_PATH = _bval["delta_path"]  # pts/g a breakout adds vs non-breakers at years +1..+4


def brk_shift(tier, k):
    """persistence of a breakout k years after next season; held at the year+4 level and decayed 15%/yr after"""
    path = DELTA_PATH[tier]
    return path[k] if k < len(path) else path[-1] * (0.85 ** (k - len(path) + 1))
current = current.merge(breakouts, on="PLAYER_ID", how="left")

# keeper-count-aware asset value (asset_value_v2.py): one value per keeper count 0..19, picked live by the slider
_av = pd.read_csv(ROOT / "data" / "asset_value.csv")
_avcols = [f"av{k}" for k in range(20)]
# ONE projection path per player, shared by VOR / trajectory and Asset value (asset_value_v2.py):
# veterans blend the Kalman path with an empirical forecast; prospects blend an empirical draft-slot model with the
# calibrated Output B path. Tested out-of-sample (engine_blend_test.py, prospect_engine_test.py).
_up = pd.read_csv(ROOT / "data" / "unified_paths.csv")
UNIFIED = {int(r.PLAYER_ID): [round(float(getattr(r, f"E{h}")), 1) for h in range(1, 8)] for r in _up.itertuples()}
# 90th-percentile career ("ceiling outcome") per player: path + asset value at each keeper count (asset_value_v2.py)
_cp = pd.read_csv(ROOT / "data" / "ceiling_paths.csv")
CEIL = {int(r.PLAYER_ID): {"path": [round(float(getattr(r, f"C{h}")), 1) for h in range(1, 8)],
                           "asset": [round(float(getattr(r, f"CA{k}")), 1) for k in range(20)]} for r in _cp.itertuples()}
ASSET_BY_PID = {int(r.PLAYER_ID): [round(float(getattr(r, c)), 1) for c in _avcols] for r in _av.itertuples()}
current_out = []
for _, r in current.iterrows():
    pos = r["espn_position"] if pd.notna(r.get("espn_position")) else r.get("POSITION")
    hist = history_by_player.get(r["PLAYER_ID"], {"years": [], "values": []})
    _uni = UNIFIED.get(int(r["PLAYER_ID"]))
    _traj_c = _uni if _uni else [round(float(v), 1) for v in r["trajectory"]]
    _fc = float(np.clip(_traj_c[0] / r["trajectory"][0], 0.6, 1.6)) if (_uni and r["trajectory"][0] > 0) else 1.0
    current_out.append({
        "kind": "current",
        "id": f"c{int(r['PLAYER_ID'])}",
        "asset_k": ASSET_BY_PID.get(int(r["PLAYER_ID"])),
        "ceil_path": (CEIL.get(int(r["PLAYER_ID"])) or {}).get("path"),
        "ceil_asset_k": (CEIL.get(int(r["PLAYER_ID"])) or {}).get("asset"),
        "player": r["player"],
        "pos": pos if pd.notna(pos) else None,
        "team": r["TEAM_ABBREVIATION"] if pd.notna(r.get("TEAM_ABBREVIATION")) else None,
        "team_logo": logo_url(r.get("TEAM_ABBREVIATION")),
        "age": round_or_none(r["age"], 1),
        "data_source": None,
        "anchor_year": CURRENT_SEASON_END_YEAR,
        "history_years": hist["years"],
        "history_values": hist["values"],
        "year0_ppg": _traj_c[0],
        "peak_ppg": max(_traj_c),
        "peak_year_index": int(np.argmax(_traj_c)),
        "trajectory": _traj_c,
        "trajectory_kalman": [round(float(v), 1) for v in r["trajectory"]],
        "unified_projection": bool(_uni),
        "had_output_b_prior": bool(r["had_output_b_prior"]),
        "injury_risk": risk_tier(r["PLAYER_ID"]),
        "espn_injury_status": r["injury_status"] if pd.notna(r.get("injury_status")) else None,
        "next_season_proj": {k: round_or_none(v * (_fc if k != "MIN" else 1.0), 4) for k, v in r["next_season_proj"].items()},
        "p_break": round_or_none(r["p_break"], 3),
        "p_break_base": round_or_none(r["p_baseline"], 3),
        "fpg_last": round_or_none(r["fpg_last"], 1),
        "brk_tier": r["tier"] if pd.notna(r.get("tier")) else None,
        "p_bust": round_or_none(r["p_bust"], 3),
        "proj_fpg": round_or_none(r["proj_fpg"], 1), "proj_lo": round_or_none(r["proj_lo"], 1), "proj_hi": round_or_none(r["proj_hi"], 1),
        "proj_gp": round_or_none(r["proj_gp"], 0), "edge_total": round_or_none(r["edge_total"], 0),
        # expected trajectory once breakout odds BEYOND what age+output already imply are folded in
        # (excess odds x how much a breakout persists); dashboard computes a separate "VOR + breakout" from it
        "trajectory_brk": (
            [round(v + (r["p_break"] - r["p_baseline"]) * brk_shift(r["tier"], i), 2) for i, v in enumerate(_traj_c)]
            if pd.notna(r.get("p_break")) else None),
        "adp": round_or_none(r["adp"], 1),
        "edge_fpg": round_or_none(r["edge_fpg"], 1),
        "break_why": r["why"] if pd.notna(r.get("why")) else None,
    })
print(f"Current players in hub: {len(current_out)}", flush=True)

# ---------------------------------------------------------------------------
# prospects -- only players who (a) haven't already debuted (would show up
# as a duplicate "current" entry otherwise) and (b) are recent enough to
# still be real, actionable fantasy prospects
# ---------------------------------------------------------------------------
prospects = pd.read_csv(ROOT / "data" / "prospect_trajectories.csv")
prospects["trajectory"] = prospects["trajectory"].apply(ast.literal_eval)
prospects["rookie_proj"] = prospects["rookie_proj"].apply(ast.literal_eval)

n_before = len(prospects)
current_ids = set(pd.read_csv(ROOT / "data" / "current_player_trajectories.csv")["PLAYER_ID"])
prospects = prospects[~prospects["PLAYER_ID"].isin(current_ids)]
n_after_dedup = len(prospects)
prospects = prospects[prospects["real_draft_year"] >= PROSPECT_RECENCY_CUTOFF]
print(f"Prospects: {n_before} -> {n_after_dedup} after removing players who already have real "
      f"current-season data (was showing stale pre-debut duplicates) -> {len(prospects)} after "
      f"restricting to draft year >= {PROSPECT_RECENCY_CUTOFF} (older non-debuts have no real "
      f"fantasy relevance today)", flush=True)

profile = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")
profile = profile.merge(
    bio[["PERSON_ID", "height_in", "WEIGHT", "POSITION"]].rename(
        columns={"PERSON_ID": "PLAYER_ID", "WEIGHT": "bio_weight", "POSITION": "bio_position"}
    ), on="PLAYER_ID", how="left",
)
profile["pos_display"] = profile["pos"].fillna(profile["bio_position"])
profile["pick_filled"] = profile["real_draft_number"].fillna(61.0)
profile["draft_age_filled"] = profile["draft_age"].fillna(profile["draft_age"].median())
profile["rec_filled"] = profile["rec"].fillna(0)
profile["talent_pctile"] = profile["talent_pctile"].fillna(profile["talent_pctile"].median())

prospects = prospects.merge(
    profile[["PLAYER_ID", "pos_display", "pick_filled", "draft_age_filled", "rec_filled",
             "talent_pctile", "bpm", "game_score", "breakout_age", "exp", "LANE_AGILITY_TIME_PCTILE"]],
    on="PLAYER_ID", how="left",
)
prospects = prospects.merge(espn_pos[["PLAYER_ID", "espn_position", "pro_team"]], on="PLAYER_ID", how="left")
n_with_team = prospects["pro_team"].notna().sum()
print(f"Prospects with a real ESPN team assignment (already drafted, not yet debuted): "
      f"{n_with_team} of {len(prospects)}", flush=True)

# rookie breakout odds (rookie_breakout_model.py) -- separate model on pre-NBA data
rk_cands = pd.read_csv(ROOT / "data" / "rookie_breakout_candidates.csv")[
    ["PLAYER_ID", "p_break", "p_baseline", "proj_fpg", "proj_lo", "proj_hi", "adp", "adp_implied_fpg", "why"]]
rk_cands = rk_cands.rename(columns={"p_break": "rk_p", "p_baseline": "rk_pbase", "proj_fpg": "rk_proj", "proj_lo": "rk_lo",
                                    "proj_hi": "rk_hi", "adp": "rk_adp", "adp_implied_fpg": "rk_adp_fpg", "why": "rk_why"})
prospects = prospects.merge(rk_cands, on="PLAYER_ID", how="left")

# ceiling calibration for top picks (prospect_calibration.py): past top-15 picks outproduced their
# Output-B-based trajectories by a pick-dependent amount that grows over the first ~3 years
_pc = json.loads((ROOT / "data" / "prospect_calibration.json").read_text(encoding="utf-8"))


def calibrate_prospect(traj, pick):
    if pick is None or pick > 15:
        return list(traj)
    taper = float(np.clip((16 - pick) / 6.0, 0.0, 1.0))  # full strength through pick 10, zero at 16+
    out = []
    for k, v in enumerate(traj):
        a, b, c = _pc["coef"][str(min(k, _pc["max_k"]))]
        out.append(round(max(v + taper * (a + b * float(np.log(pick)) + c * float(pick == 1)), 0.0), 1))
    return out


prospect_out = []
for _, r in prospects.iterrows():
    _pick = int(r["pick_filled"]) if r["pick_filled"] < 61 else None
    _cal_traj = calibrate_prospect(r["trajectory"], _pick)
    _uni_p = UNIFIED.get(int(r["PLAYER_ID"]))
    _traj = _uni_p if _uni_p else _cal_traj
    _f = (_traj[0] / r["trajectory"][0]) if r["trajectory"][0] > 0 else 1.0
    _proj = {k: (round_or_none(v * _f, 4) if k != "MIN" else round_or_none(v, 4)) for k, v in r["rookie_proj"].items()}
    pos = r["espn_position"] if pd.notna(r.get("espn_position")) else r["pos_display"]
    prospect_out.append({
        "kind": "prospect",
        "id": f"p{int(r['PLAYER_ID'])}",
        "asset_k": ASSET_BY_PID.get(int(r["PLAYER_ID"])),
        "ceil_path": (CEIL.get(int(r["PLAYER_ID"])) or {}).get("path"),
        "ceil_asset_k": (CEIL.get(int(r["PLAYER_ID"])) or {}).get("asset"),
        "brk_tier": "rookie" if pd.notna(r.get("rk_p")) else None,
        "p_break": round_or_none(r.get("rk_p"), 3), "p_break_base": round_or_none(r.get("rk_pbase"), 3),
        "proj_fpg": round_or_none(r.get("rk_proj"), 1), "proj_lo": round_or_none(r.get("rk_lo"), 1), "proj_hi": round_or_none(r.get("rk_hi"), 1),
        "adp": round_or_none(r.get("rk_adp"), 1), "adp_implied_fpg": round_or_none(r.get("rk_adp_fpg"), 1),
        "break_why": r["rk_why"] if pd.notna(r.get("rk_why")) else None,
        "player": r["player"],
        "pos": pos if pd.notna(pos) else None,
        "team": r["pro_team"] if pd.notna(r.get("pro_team")) else None,
        "team_logo": logo_url(r.get("pro_team")),
        "age": round_or_none(r["draft_age_filled"], 1),
        "year": int(r["real_draft_year"]) if pd.notna(r["real_draft_year"]) else None,
        "anchor_year": int(r["real_draft_year"]) + 1 if pd.notna(r["real_draft_year"]) else CURRENT_SEASON_END_YEAR,
        "pick": int(r["pick_filled"]) if r["pick_filled"] < 61 else None,
        "data_source": r["data_source"],
        "year0_ppg": _traj[0],
        "peak_ppg": max(_traj),
        "peak_year_index": int(np.argmax(_traj)),
        "trajectory": _traj,
        "trajectory_uncalibrated": [round(float(v), 1) for v in r["trajectory"]],
        "trajectory_kalman": _cal_traj,
        "unified_projection": bool(_uni_p),
        "rookie_proj": _proj,
        "talent_pctile": round_or_none(r["talent_pctile"], 3),
        "bpm": round_or_none(r["bpm"], 2) if r["data_source"] == "college" else None,
        "game_score": round_or_none(r.get("game_score"), 2) if r["data_source"] == "international" else None,
        "draft_age": round_or_none(r["draft_age_filled"], 2),
        "breakout_age": round_or_none(r["breakout_age"], 2) if pd.notna(r["breakout_age"]) else None,
        "rec": round_or_none(r["rec_filled"], 1) if r["rec_filled"] else None,
        "exp": r["exp"] if pd.notna(r["exp"]) else None,
        "agility_pctile": round_or_none(r["LANE_AGILITY_TIME_PCTILE"]) if pd.notna(r["LANE_AGILITY_TIME_PCTILE"]) else None,
        "low_conf": int(r["data_source"] == "international"),
        "low_conf_reason": (
            "international prospect -- rookie projection uses a separate, smaller-sample model "
            "(n=52) trained on real box-score rates rather than the richer college feature set"
            if r["data_source"] == "international" else ""
        ),
    })
print(f"Prospects in hub: {len(prospect_out)} (college: {sum(1 for p in prospect_out if p['data_source']=='college')}, "
      f"international: {sum(1 for p in prospect_out if p['data_source']=='international')})", flush=True)

# --- carry over the existing prospect-profile + stylistic comps ---
try:
    old = json.loads((ROOT.parent.parent / "dashboard" / "rookie_scores.json").read_text(encoding="utf-8"))
    comps_by_name = {row["Player"]: (row.get("similar"), row.get("style_similar"), row.get("style_low_conf"))
                      for row in old["scored"]}
    for p in prospect_out:
        c = comps_by_name.get(p["player"])
        if c:
            p["similar"], p["style_similar"], p["style_low_conf"] = c[0], c[1], c[2]
    print("Merged in existing prospect-profile + stylistic comps.", flush=True)
except FileNotFoundError:
    print("No existing rookie_scores.json found -- comps left empty.", flush=True)

out = {
    "generated": pd.Timestamp.now().isoformat(),
    "opportunity_cost": {"floor": OPP_FLOOR, "amp": OPP_AMP, "tau": OPP_TAU},
    "horizon_years": HORIZON_YEARS,
    "default_keepers": 3,
    "note": "VOR is computed live in the browser from each player's raw trajectory + the "
            "opportunity-cost curve (same real formula fit from this league's own draft "
            "history) -- adjust the keeper-count slider to see value shift.",
    "breakout_meta": {**_bval, "rookie": json.loads((ROOT / "data" / "rookie_breakout_validation.json").read_text(encoding="utf-8")), "opener": str(OPENER), "frozen": frozen,
                      "frozen_at": json.loads(LEDGER_META.read_text())["frozen_at"] if frozen else None},
    "players": current_out + prospect_out,
}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, indent=None, allow_nan=False), encoding="utf-8")
print(f"\nWrote {len(out['players'])} total players to {OUT_PATH}")

# ---------------------------------------------------------------------------
# breakout history (walk-forward predictions + what actually happened), loaded
# lazily by the dashboard's year selector
# ---------------------------------------------------------------------------
hist = pd.concat([pd.read_csv(ROOT / "data" / "breakout_history.csv"), pd.read_csv(ROOT / "data" / "rookie_breakout_history.csv")],
                 ignore_index=True)
hist_years = {}
for yr, g in hist.groupby("yr"):
    g = g.sort_values("p_break", ascending=False)
    hist_years[str(int(yr))] = [{
        "id": f"h{int(yr)}_{int(r.PLAYER_ID)}", "player": r.player, "age": round_or_none(r.age, 1) if pd.notna(r.age) else None,
        "team": r.team if pd.notna(r.team) else None, "team_logo": logo_url(r.team) if pd.notna(r.team) else None,
        "brk_tier": r.tier, "fpg_last": round_or_none(r.fpg_last, 1), "p_break": round_or_none(r.p_break, 3),
        "p_bust": round_or_none(r.p_bust, 3), "proj_fpg": round_or_none(r.proj_fpg, 1), "proj_lo": round_or_none(r.proj_lo, 1),
        "proj_hi": round_or_none(r.proj_hi, 1), "bust_hit": None if pd.isna(r.bust_hit) else int(r.bust_hit),
        "adp": round_or_none(r.adp, 1), "edge_fpg": round_or_none(r.edge_fpg, 1), "break_why": r.why if pd.notna(r.why) else None,
        "fpg_next": round_or_none(r.fpg_next, 1), "gp_next": int(r.GP_next) if pd.notna(r.GP_next) else None,
        "d_next": round_or_none(r.d_next, 1), "hit": None if pd.isna(r.hit) else int(r.hit),
    } for r in g.itertuples()]
(OUT_PATH.parent / "breakout_history.json").write_text(json.dumps(hist_years, allow_nan=False), encoding="utf-8")
print(f"Wrote breakout_history.json: {len(hist_years)} seasons, {len(hist)} rows")
