"""Game-level projection research (2026-09-27, at Tommy's request): can we predict which SPECIFIC games a player will go off in (or underperform)
from opponent context, and does a per-possession production basis beat per-minute? Everything here is trailing/to-date (as of before each game) so
this reflects what could actually be forecast in advance, not hindsight. `gamelevel_study.py`; builds on pull_team_gamelogs_full.py's team box data.

PART 1: per-minute vs per-100-possessions as the rate basis.
PART 2: opponent context -- overall defense, positional defense, both teams' back-to-back status, opponent's missing rotation production.
Rotation players only (rolling 15+ mpg over the trailing 15 games), and only games with 10+ prior games that season (a real trailing baseline).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F
import usage_flow as UF

sys.stdout.reconfigure(encoding="utf-8")
GL = F.D.parent / "gamelevel"
LEAGUE_PACE = 97.0

# ---------------- load everything built so far ----------------
g = pd.read_pickle(GL / "player_games_with_opp.pkl")
POS = pd.read_pickle(GL / "player_pos.pkl")
g = g.join(POS, on=["pid", "season"])
tb = pd.read_pickle(GL / "team_box_features.pkl")
posdef = pd.read_pickle(GL / "team_pos_defense.pkl")
miss = pd.read_pickle(GL / "team_missing.pkl")

g = g.sort_values(["pid", "season", "date"]).reset_index(drop=True)
gk = g.groupby(["pid", "season"])
g["n_prior"] = gk.cumcount()
g["min_td"] = gk["min"].transform(lambda s: s.shift(1).expanding().mean())
g["fp_per_min_td"] = gk.apply(lambda d: (d["fp"] / d["min"]).shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
g["fp_td"] = gk["fp"].transform(lambda s: s.shift(1).expanding().mean())

# player-possession rate: his fp per 100 of HIS OWN team's possessions that game, using the team's pace that game (poss column on the OWN team's row)
own_pace = tb.rename(columns={"team": "TEAM_ABBREVIATION_"}).set_index(["TEAM_ABBREVIATION_", "GAME_DATE"]).poss if False else None
tb_pace_by_gid = tb.set_index("GAME_ID").poss.to_dict()
g["poss_game"] = g.GAME_ID.map(tb_pace_by_gid)
g["player_poss"] = g.poss_game * g["min"] / 48.0
g["fp_rate100"] = np.where(g.player_poss > 0, g.fp / g.player_poss * 100, np.nan)
g["fp_rate100_td"] = gk["fp_rate100"].transform(lambda s: s.shift(1).expanding().mean())

g = g.merge(tb[["team", "season", "GAME_DATE", "b2b", "pace_td"]], left_on=["team", "season", "date"], right_on=["team", "season", "GAME_DATE"], how="left", suffixes=("", "_own"))
g = g.merge(tb[["team", "season", "GAME_DATE", "b2b", "pace_td", "drtg_td"]].rename(columns={"team": "opp"}),
            left_on=["opp", "season", "date"], right_on=["opp", "season", "GAME_DATE"], how="left", suffixes=("", "_opp"))
g = g.merge(posdef.rename(columns={"team": "opp", "DATE_D": "DATE_D_pd"}), left_on=["opp", "season", "DATE_D"], right_on=["opp", "season", "DATE_D_pd"], how="left")
g = g.merge(miss.rename(columns={"team": "opp", "DATE_D": "DATE_D_m"}), left_on=["opp", "season", "DATE_D"], right_on=["opp", "season", "DATE_D_m"], how="left")

g["opp_pos_def_td"] = g.pC * g.fpC_td + g.pF * g.fpF_td + g.pG * g.fpG_td            # opponent's trailing fp allowed to a player with HIS position mix
g["exp_pace"] = (g.pace_td + g.pace_td_opp) / 2                                     # expected pace of THIS matchup, both sides knowable in advance

ROT = g[(g.n_prior >= 10) & (g.min_td >= 15)].dropna(subset=["fp_per_min_td", "fp_rate100_td", "pace_td", "pace_td_opp", "drtg_td_opp", "opp_pos_def_td"]).copy()
print(f"{len(ROT)} rotation player-games with a full trailing baseline and opponent context, {ROT.pid.nunique()} players, seasons {ROT.season.min()}-{ROT.season.max()}")

# ==================== PART 1: per-minute vs per-100-possessions ====================
print("\n" + "=" * 70)
print("PART 1: rate basis -- per minute vs per 100 (team) possessions")
print("=" * 70)
# (a) ORACLE decomposition: given his TRUE minutes and TRUE pace this game, which rate basis better explains the outcome?
ROT["pred_permin_oracle"] = ROT.fp_per_min_td * ROT["min"]
ROT["pred_perposs_oracle"] = ROT.fp_rate100_td * ROT.player_poss / 100
for name, col in [("per-minute (his trailing rate x his real minutes)", "pred_permin_oracle"), ("per-100-poss (his trailing rate x his real minutes AND real pace)", "pred_perposs_oracle")]:
    err = ROT.fp - ROT[col]
    print(f"  oracle, {name:62s} RMSE {np.sqrt((err**2).mean()):.3f}  MAE {err.abs().mean():.3f}")

# (b) REAL FORECAST: both minutes and pace are themselves forecast from trailing averages (what we could actually do in advance)
ROT["pred_permin_fc"] = ROT.fp_per_min_td * ROT.min_td
ROT["exp_player_poss"] = ROT.exp_pace * ROT.min_td / 48.0
ROT["pred_perposs_fc"] = ROT.fp_rate100_td * ROT.exp_player_poss / 100
for name, col in [("per-minute (trailing rate x trailing minutes)", "pred_permin_fc"), ("per-100-poss (trailing rate x trailing minutes x EXPECTED pace)", "pred_perposs_fc")]:
    err = ROT.fp - ROT[col]
    print(f"  forecast, {name:56s} RMSE {np.sqrt((err**2).mean()):.3f}  MAE {err.abs().mean():.3f}")
print(f"  correlation(exp_pace, fp_per_min_td) = {ROT.exp_pace.corr(ROT.fp_per_min_td):+.3f}  (near 0 would mean pace and skill are unrelated, as expected)")
print(f"  spread of exp_pace across games: sd={ROT.exp_pace.std():.2f}, min={ROT.exp_pace.min():.1f}, max={ROT.exp_pace.max():.1f} (vs league mean ~{LEAGUE_PACE})")

# ==================== PART 2: opponent context ====================
print("\n" + "=" * 70)
print("PART 2: opponent context on top of the better rate-basis forecast")
print("=" * 70)
BEST_COL = "pred_perposs_fc"          # filled in after seeing part 1; kept generic here
ROT["dev"] = ROT.fp - ROT[BEST_COL]   # deviation from the best forecast-only baseline -- what context features should explain
ROT["dev_permin"] = ROT.fp - ROT.pred_permin_fc

print("\n=== each candidate alone: correlation with the deviation from the pace-adjusted forecast ===")
cands = {
    "drtg_td_opp": "opponent's overall defensive rating allowed (higher = softer defense)",
    "opp_pos_def_td": "opponent's fp allowed to players with his SAME position mix",
    "b2b": "his own team on a back-to-back",
    "b2b_opp": "opponent on a back-to-back",
    "opp_missing_fp": "opponent's missing rotation production that night",
}
for c, desc in cands.items():
    sub = ROT.dropna(subset=[c, "dev"])
    r = sub[c].corr(sub.dev)
    print(f"  {c:16s} r={r:+.4f}  n={len(sub):6d}   ({desc})")

print("\n=== combined (leave-one-season-out): does opponent context improve on the rate-basis forecast alone? ===")
COLS = list(cands.keys())
PF = ROT.dropna(subset=COLS + ["fp", BEST_COL]).copy()
print(f"n={len(PF)}")


def loso(df, feat_cols, target_col, base_col, model_fn):
    errs_base, errs_full = [], []
    for s in df.season.unique():
        tr, te = df[df.season != s], df[df.season == s]
        if len(te) < 200:
            continue
        mu, sd = tr[feat_cols].mean(), tr[feat_cols].std().replace(0, 1)
        m = model_fn().fit((tr[feat_cols] - mu) / sd, tr[target_col] - tr[base_col])
        pred_delta = m.predict((te[feat_cols] - mu) / sd)
        errs_full.append((te[target_col] - (te[base_col] + pred_delta)).values)
        errs_base.append((te[target_col] - te[base_col]).values)
    return np.sqrt(np.mean(np.concatenate(errs_base) ** 2)), np.sqrt(np.mean(np.concatenate(errs_full) ** 2))


rmse_base, rmse_ridge = loso(PF, COLS, "fp", BEST_COL, lambda: Ridge(alpha=3.0))
print(f"  baseline (rate-basis forecast alone):  RMSE {rmse_base:.4f}")
print(f"  + opponent context (ridge):            RMSE {rmse_ridge:.4f}")
_, rmse_gbm = loso(PF, COLS, "fp", BEST_COL, lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200, l2_regularization=1.0))
print(f"  + opponent context (gradient boosted): RMSE {rmse_gbm:.4f}")

gbm_full = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200, l2_regularization=1.0).fit(PF[COLS], PF.fp - PF[BEST_COL])
from sklearn.inspection import permutation_importance
pi = permutation_importance(gbm_full, PF[COLS], PF.fp - PF[BEST_COL], n_repeats=10, random_state=0)
print("\npermutation importance (predicting the deviation from baseline):")
for c, imp in sorted(zip(COLS, pi.importances_mean), key=lambda x: -x[1]):
    print(f"  {c:16s} {imp:+.5f}")

print("\n=== bucketed view: opponent overall defense (drtg allowed), quintiles ===")
PF["drtg_q"] = pd.qcut(PF.drtg_td_opp, 5, labels=["1 (best D)", "2", "3", "4", "5 (worst D)"])
print(PF.groupby("drtg_q", observed=True).dev.agg(["mean", "count"]))
print("\n=== bucketed view: opponent positional defense (fp allowed to his position), quintiles ===")
PF["posdef_q"] = pd.qcut(PF.opp_pos_def_td, 5, labels=["1 (toughest)", "2", "3", "4", "5 (softest)"])
print(PF.groupby("posdef_q", observed=True).dev.agg(["mean", "count"]))
print("\n=== back-to-back effects (raw, not residualized -- direct check) ===")
for c in ["b2b", "b2b_opp"]:
    print(f"  {c}:")
    print(PF.groupby(c).dev.agg(["mean", "count"]))
print("\n=== opponent missing production, quintiles ===")
PF["miss_q"] = pd.qcut(PF.opp_missing_fp, 5, labels=["1 (healthy)", "2", "3", "4", "5 (banged up)"])
print(PF.groupby("miss_q", observed=True).dev.agg(["mean", "count"]))
