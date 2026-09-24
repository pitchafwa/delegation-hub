"""Rookie breakout model: a SEPARATE model for players with no NBA history, built on the
college / international pre-NBA profile (same feature set as Output B).

Target: rookie-season fantasy pts/g (real, from player_season_base, >=20 GP).
"Breakout" for a rookie = being fantasy-relevant immediately: rookie pts/g within 1 pt of the
32.4 quality line (same relevance rule as the veteran Emerging model).

  P(relevant year 1)  logistic on the pre-NBA profile
  proj rookie pts/g   ridge, with an empirical 10-90% band
  baseline            draft pick alone -- what the draft slot already tells everyone
  edge vs ADP         projected rookie pts/g minus what ESPN's ADP implies (ADP->pts/g relation
                      fit on veterans, applied to rookies; ADP-listed rookies only)

Validation is expanding-window by draft class (train on earlier classes only).
Usage: python rookie_breakout_model.py -> data/rookie_breakout_candidates.csv,
       data/rookie_breakout_history.csv, data/rookie_breakout_validation.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
Q, TOL = 32.4, 1.0

df = pd.read_csv(D / "rookie_model_dataset_unified.csv")
df = df[df["real_draft_year"] >= 2008].copy()
EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["pick_filled"] = df["real_draft_number"].fillna(61.0)
df["is_intl"] = (df["data_source"] == "international").astype(int)
df["log_pick"] = np.log(df["pick_filled"])

FEATURES = [
    "talent_pctile", "rec_filled", "exp_numeric", "draft_age_filled", "breakout_age_filled",
    "porpag", "usg", "ts", "ortg", "obpm", "dbpm", "bpm", "stops",
    "oreb_rate", "dreb_rate", "ast_to", "ftr", "pfr",
    "WINGSPAN_PCTILE", "STANDING_REACH_PCTILE", "STANDING_VERTICAL_LEAP_PCTILE",
    "MAX_VERTICAL_LEAP_PCTILE", "LANE_AGILITY_TIME_PCTILE", "THREE_QUARTER_SPRINT_PCTILE",
    "three_pct", "rim_pct", "mid_pct", "log_pick", "is_intl",
]
FEATURES = [f for f in FEATURES if f in df.columns]
BASE = ["log_pick"]
for f in FEATURES:
    df[f] = pd.to_numeric(df[f], errors="coerce")

# ---- real rookie-season outcome (season starting in the draft year)
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg_rook"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
                  - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
rk = sb[["PLAYER_ID", "yr", "fpg_rook", "GP"]].rename(columns={"yr": "real_draft_year", "GP": "gp_rook"})
df = df.merge(rk, on=["PLAYER_ID", "real_draft_year"], how="left")
df["valid"] = df["gp_rook"] >= 20
df["rel"] = (df["fpg_rook"] >= Q - TOL).astype(int)

def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def rg(alpha=100):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


# ---- ADP: ESPN season_id = draft_year + 1 prices the rookie season; ADP->pts/g relation fit on veterans
adp = pd.read_csv(D / "espn_adp.csv")
adp = adp[adp["adp"].notna() & adp["PLAYER_ID"].notna()].copy()
adp["PLAYER_ID"] = adp["PLAYER_ID"].astype(int)
adp["real_draft_year"] = adp["season_id"] - 1
adp = adp.sort_values("adp").drop_duplicates(["PLAYER_ID", "real_draft_year"])
df = df.merge(adp[["PLAYER_ID", "real_draft_year", "adp"]], on=["PLAYER_ID", "real_draft_year"], how="left")
panel = pd.read_csv(D / "breakout_panel.csv")
vadp = adp.copy()
vadp["yr"] = vadp["season_id"] - 2
vv = panel.merge(vadp[["PLAYER_ID", "yr", "adp"]], on=["PLAYER_ID", "yr"], how="inner")
vv = vv[vv["fpg_next"].notna() & (vv["GP_next"] >= 30) & (vv["yr"].between(2017, 2023))]
vv["ladp"] = np.log(vv["adp"])
vv["l2"], vv["l3"] = vv["ladp"] ** 2, vv["ladp"] ** 3
adp_model = rg(1).fit(vv[["ladp", "l2", "l3"]], vv["fpg_next"])


def adp_fpg(a):
    la = np.log(np.asarray(a, dtype=float))
    return adp_model.predict(pd.DataFrame({"ladp": la, "l2": la ** 2, "l3": la ** 3}))



tr_all = df[df["valid"]].copy()
print(f"rookies with a real >=20 GP rookie season: {len(tr_all)} (classes {int(tr_all.real_draft_year.min())}-{int(tr_all.real_draft_year.max())}); "
      f"relevant in year 1: {tr_all['rel'].sum()} ({tr_all['rel'].mean():.1%})")


rm = lambda a, b: float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))
years = range(2015, 2026)


def walk(feats):
    rows = []
    for y in years:
        tr, te = tr_all[tr_all["real_draft_year"] < y], tr_all[tr_all["real_draft_year"] == y].copy()
        if te.empty:
            continue
        te["p"] = lr().fit(tr[feats], tr["rel"]).predict_proba(te[feats])[:, 1]
        te["proj"] = rg().fit(tr[feats], tr["fpg_rook"]).predict(te[feats])
        rows.append(te)
    return pd.concat(rows)


val = {}
SMALL = [f for f in ["log_pick", "talent_pctile", "draft_age_filled", "is_intl", "exp_numeric", "bpm", "usg", "ts"] if f in FEATURES]
for name, feats in [("pick_only", BASE), ("pick+few", SMALL), ("full", FEATURES)]:
    w = walk(feats)
    o = w.sort_values("p", ascending=False)
    k10 = max(1, int(len(w) * .10))
    val[name] = {"auc": float(roc_auc_score(w["rel"], w["p"])), "top10_hit": float(o.head(k10)["rel"].mean()), "base": float(w["rel"].mean()),
                 "rmse": rm(w["fpg_rook"], w["proj"]), "corr": float(w["proj"].corr(w["fpg_rook"])), "n": int(len(w))}
    print(f"{name:10s} n={len(w)} AUC {val[name]['auc']:.3f} | top-10% hit {val[name]['top10_hit']:.3f} vs base {val[name]['base']:.3f} | "
          f"RMSE {val[name]['rmse']:.2f} corr {val[name]['corr']:.3f}")
PROD = SMALL  # 'pick + a few profile features' tested best; the other ~20 features only add noise
w_full = walk(PROD)
w_base = walk(BASE)
val["rmse_naive_mean"] = rm(w_full["fpg_rook"], tr_all["fpg_rook"].mean())
val["n_intl"] = int(w_full["is_intl"].sum())
ii = w_full[w_full["is_intl"] == 1]
if len(ii) > 15:
    val["intl_auc"] = float(roc_auc_score(ii["rel"], ii["p"])) if ii["rel"].nunique() > 1 else None

# 10-90% band from walk-forward residuals
res = w_full["fpg_rook"] - w_full["proj"]
BAND = (float(res.quantile(.10)), float(res.quantile(.90)))
val["band_10_90"] = BAND
val["features_used"] = PROD
print("projection band (10-90% residual):", tuple(round(x, 1) for x in BAND))

wl = w_full[w_full["adp"].notna()].copy()
if len(wl) >= 10:
    wl["adp_fpg"] = adp_fpg(wl["adp"])
    wl["edge"] = wl["proj"] - wl["adp_fpg"]
    wl["sur"] = wl["fpg_rook"] - wl["adp_fpg"]
    val["adp"] = {"n": int(len(wl)), "corr_edge_vs_surprise": float(wl["edge"].corr(wl["sur"])),
                  "rmse_adp_only": rm(wl["fpg_rook"], wl["adp_fpg"]), "rmse_model_only": rm(wl["fpg_rook"], wl["proj"]),
                  "rmse_blend": rm(wl["fpg_rook"], 0.5 * wl["proj"] + 0.5 * wl["adp_fpg"])}
    print("rookie ADP test (small n):", {k: round(v, 3) for k, v in val["adp"].items()})


def why(r):
    out = []
    pk = r["pick_filled"]
    out.append(f"#{int(pk)} pick" if pk <= 60 else "undrafted")
    if pd.notna(r.get("talent_pctile")) and r["talent_pctile"] >= 0.85:
        out.append(f"elite pre-NBA production ({r['talent_pctile']*100:.0f}th pct)")
    if pd.notna(r.get("draft_age_filled")) and r["draft_age_filled"] <= 19.6:
        out.append("very young")
    if r.get("is_intl") == 1:
        out.append("international")
    return "; ".join(out[:4])


# ---- live: incoming rookies (no rookie season yet)
full_m, base_m = lr().fit(tr_all[PROD], tr_all["rel"]), lr().fit(tr_all[BASE], tr_all["rel"])
proj_m = rg().fit(tr_all[PROD], tr_all["fpg_rook"])
live = df[df["gp_rook"].isna() & (df["real_draft_year"] >= 2025)].copy()
live["p_break"] = full_m.predict_proba(live[PROD])[:, 1]
live["p_baseline"] = base_m.predict_proba(live[BASE])[:, 1]
live["proj_fpg"] = proj_m.predict(live[PROD])
live["proj_lo"], live["proj_hi"] = live["proj_fpg"] + BAND[0], live["proj_fpg"] + BAND[1]
live["edge_fpg"] = np.nan  # not exposed: ADP beat the model at rookies in backtests (see validation json)
live["adp_implied_fpg"] = np.where(live["adp"].notna(), adp_fpg(live["adp"].fillna(100)), np.nan)
live["why"] = live.apply(why, axis=1)
out = live[["PLAYER_ID", "player", "real_draft_year", "pick_filled", "data_source", "p_break", "p_baseline", "proj_fpg", "proj_lo",
            "proj_hi", "adp", "adp_implied_fpg", "edge_fpg", "why"]].sort_values("p_break", ascending=False)
out.to_csv(D / "rookie_breakout_candidates.csv", index=False)
print(f"\nlive incoming rookies scored: {len(out)}")
print(out.head(12)[["player", "pick_filled", "data_source", "p_break", "p_baseline", "proj_fpg", "adp", "edge_fpg"]].round(2).to_string(index=False))

# ---- history rows (walk-forward) for the dashboard's season selector; class Y => panel-year Y-1
h = w_full.copy()
h["adp_fpg"] = np.where(h["adp"].notna(), adp_fpg(h["adp"].fillna(100)), np.nan)
h["edge_fpg"] = np.nan
h["why"] = h.apply(why, axis=1)
hist = pd.DataFrame({
    "yr": h["real_draft_year"].astype(int) - 1, "PLAYER_ID": h["PLAYER_ID"], "player": h["player"], "age": h["draft_age_filled"],
    "team": None, "tier": "rookie", "fpg_last": np.nan, "p_break": h["p"], "p_bust": np.nan, "proj_fpg": h["proj"],
    "proj_lo": h["proj"] + BAND[0], "proj_hi": h["proj"] + BAND[1], "adp": h["adp"], "edge_fpg": h["edge_fpg"], "why": h["why"],
    "fpg_next": h["fpg_rook"], "GP_next": h["gp_rook"], "d_next": np.nan, "hit": h["rel"].astype(float), "bust_hit": np.nan})
hist.to_csv(D / "rookie_breakout_history.csv", index=False)
(D / "rookie_breakout_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
print(f"\nrookie history rows: {len(hist)} across classes {int(h['real_draft_year'].min())}-{int(h['real_draft_year'].max())}")
