"""Breakout-candidate model v3.

ADP-INDEPENDENT models (stats + role + situation only):
  EMERGENCE  pool: below the 32.4 pts/g quality line. label: next season within TOL of the line
             (>= 32.4 - 1.0) AND a real +6 jump (>=30 GP). The line itself is a fitted estimate, so a
             player who lands 0.01 short (Filipowski 32.39) counts.
  ELEVATION  pool: already at/above the line ("good -> great"). label: +6 pts/g jump (>=30 GP).
  BUST       pool: at/above the line. label: a -6 pts/g drop (>=30 GP). Foundation for sell-high.
  PROJECTION ridge model for next-season pts/g with an empirical 10-90% range (walk-forward residuals).
  AVAILABILITY ridge for next-season games played (fraction of the season) from GP history.
ADP-AWARE:
  EDGE       expected next-season pts/g (ADP + our features) minus what ADP alone implies (per game).
  TOTAL EDGE same idea for SEASON-TOTAL points (per-game x games played) -- shown as an additional
             column, it does NOT change any topline output.

Also runs the "should breakout odds change VOR?" accuracy test and estimates the breakout
persistence path used by the dashboard's separate "VOR + breakout" column.

All validation is expanding-window: train on earlier seasons only.
Usage:  python breakout_model.py  -> data/breakout_candidates.csv, breakout_history.csv, breakout_validation.json
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
sys.path.insert(0, str(ROOT))
from build_breakout_context import load_coach_sets, situation_features  # noqa: E402

Q = 32.4
TOL = 1.0          # the relevance line is a fitted estimate: landing within 1 pt of it counts
BASE = ["fpg", "AGE", "young"]
STATS = ["mpg", "late_mpg", "late_min_delta", "d_mpg", "late_fpg", "late_fpg_delta",
         "late_fp_per36", "USG_PCT", "PIE", "draft_pick", "exp"]
SIT = ["moved", "dest_net_usg", "dest_vac_min"]  # head-coach change tested and excluded (no lift)
FULL = BASE + STATS + SIT
GPF = ["gp_frac", "gp_frac_p1", "gp_frac_p2", "AGE", "mpg", "exp", "fpg"]
ADPF = ["ladp", "l2", "l3", "has_adp"]

p = pd.read_csv(D / "breakout_panel_ctx2.csv")
p["young"] = (p["AGE"] <= 24) * (24 - p["AGE"])


def shift_col(df, col, h, name):
    s = df[["PLAYER_ID", "yr", col]].copy()
    s["yr"] -= h
    return df.merge(s.rename(columns={col: name}), on=["PLAYER_ID", "yr"], how="left")


# season-length-normalized availability + longer-horizon outcomes
p["gp_frac"] = p["GP"] / p.groupby("yr")["GP"].transform("max")
for h, nm in [(-1, "gp_frac_p1"), (-2, "gp_frac_p2")]:
    p = shift_col(p, "gp_frac", h, nm)
p = shift_col(p, "gp_frac", 1, "gp_frac_next")
for h in (2, 3, 4):
    p = shift_col(p, "fpg", h, f"fpg_next{h}")
p["fpg_next1"] = p["fpg_next"]
p["tot_next"] = p["fpg_next"] * p["gp_frac_next"] * 82  # season-total pts, 82-game equivalent

# ---- ESPN ADP (seasonId s prices the season AFTER panel year s-2)
adp_all = pd.read_csv(D / "espn_adp.csv")
adp_hist = adp_all[adp_all["adp"].notna() & adp_all["PLAYER_ID"].notna()].copy()
adp_hist["PLAYER_ID"] = adp_hist["PLAYER_ID"].astype(int)
adp_hist["yr"] = adp_hist["season_id"] - 2
adp_hist = adp_hist.sort_values("adp").drop_duplicates(["PLAYER_ID", "yr"])
p = p.merge(adp_hist[["PLAYER_ID", "yr", "adp"]], on=["PLAYER_ID", "yr"], how="left")
p["ladp"] = np.log(p["adp"].fillna(200))
p["has_adp"] = p["adp"].notna().astype(int)
p["l2"], p["l3"] = p["ladp"] ** 2, p["ladp"] ** 3

pool_ok = (p["GP"] >= 20) & (p["mpg"] >= 10)
labeled = p[pool_ok & p["fpg_next"].notna() & (p["GP_next"] >= 30)].copy()
labeled["brk_emerge"] = ((labeled["fpg_next"] >= Q - TOL) & (labeled["d_next"] >= 6)).astype(int)
labeled["brk_elev"] = (labeled["d_next"] >= 6).astype(int)
labeled["bust"] = (labeled["d_next"] <= -6).astype(int)
POOLS = {
    "emergence": (labeled["fpg"] < Q, "brk_emerge"),
    "elevation": (labeled["fpg"] >= Q, "brk_elev"),
}


def lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def hit_rates(y, pr):
    o = np.argsort(-pr)
    return {"top5": float(y[o[: max(1, int(len(y) * .05))]].mean()), "top10": float(y[o[: max(1, int(len(y) * .10))]].mean()),
            "base": float(y.mean())}


def cv_classifier(c, lab, feats, years=range(2014, 2025)):
    pr, yt = [], []
    for ty in years:
        tr, te = c[c["yr"] < ty], c[c["yr"] == ty]
        pr += list(lr().fit(tr[feats], tr[lab]).predict_proba(te[feats])[:, 1]); yt += list(te[lab])
    pr, yt = np.array(pr), np.array(yt)
    return {"auc": float(roc_auc_score(yt, pr)), **hit_rates(yt, pr)}


rm = lambda a, b: float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))
validation = {"pools": {}, "adp": {}, "tol": TOL}
models = {}

# ---------------------------------------------------------------- breakout pools
for name, (mask, lab) in POOLS.items():
    c = labeled[mask]
    res = {"baseline": cv_classifier(c, lab, BASE), "full": cv_classifier(c, lab, FULL),
           "n": int(len(c)), "positives": int(c[lab].sum())}
    validation["pools"][name] = res
    models[name] = lr().fit(c[FULL], c[lab])
    print(f"{name:10s} n={res['n']} pos={res['positives']}  baseline AUC {res['baseline']['auc']:.3f} -> full {res['full']['auc']:.3f} | "
          f"top10% hit {res['full']['top10']:.3f} vs base {res['full']['base']:.3f}")

# ---------------------------------------------------------------- bust model (sell-high foundation)
bc = labeled[labeled["fpg"] >= Q]
validation["bust"] = {"baseline": cv_classifier(bc, "bust", BASE), "full": cv_classifier(bc, "bust", FULL),
                      "n": int(len(bc)), "positives": int(bc["bust"].sum())}
bust_model = lr().fit(bc[FULL], bc["bust"])
print(f"bust       n={len(bc)} pos={bc['bust'].sum()}  baseline AUC {validation['bust']['baseline']['auc']:.3f} -> full "
      f"{validation['bust']['full']['auc']:.3f} | top10% hit {validation['bust']['full']['top10']:.3f} vs base {validation['bust']['full']['base']:.3f}")

# ---------------------------------------------------------------- projected pts/g with a 10-90% range
oof = []
for ty in range(2014, 2025):
    tr, te = labeled[labeled["yr"] < ty], labeled[labeled["yr"] == ty].copy()
    te["proj"] = ridge(10).fit(tr[FULL], tr["fpg_next"]).predict(te[FULL])
    oof.append(te)
oof = pd.concat(oof)
oof["res"] = oof["fpg_next"] - oof["proj"]
QUANT = {}
for name, (mask, _) in POOLS.items():
    r = oof.loc[oof.index.isin(labeled[mask].index), "res"]
    QUANT[name] = (float(r.quantile(.10)), float(r.quantile(.90)))
validation["projection"] = {"rmse": rm(oof["fpg_next"], oof["proj"]), "rmse_naive": rm(oof["fpg_next"], oof["fpg"]),
                            "range_10_90": QUANT}
proj_model = ridge(10).fit(labeled[FULL], labeled["fpg_next"])
print("projection RMSE", round(validation["projection"]["rmse"], 3), "vs naive last-year", round(validation["projection"]["rmse_naive"], 3),
      "| 10-90% residual band", {k: tuple(round(x, 1) for x in v) for k, v in QUANT.items()})

# ---------------------------------------------------------------- availability (games played)
gc = p[pool_ok & p["gp_frac_next"].notna()].copy()
oofg = []
for ty in range(2014, 2025):
    tr, te = gc[gc["yr"] < ty], gc[gc["yr"] == ty].copy()
    te["pg"] = ridge(10).fit(tr[GPF], tr["gp_frac_next"]).predict(te[GPF])
    oofg.append(te)
oofg = pd.concat(oofg)
validation["availability"] = {"rmse_games": rm(oofg["gp_frac_next"], oofg["pg"]) * 82,
                              "rmse_games_naive_last_year": rm(oofg["gp_frac_next"], oofg["gp_frac"]) * 82,
                              "corr": float(oofg["gp_frac_next"].corr(oofg["pg"])), "n": int(len(oofg))}
gp_model = ridge(10).fit(gc[GPF], gc["gp_frac_next"])
print("availability RMSE (games):", round(validation["availability"]["rmse_games"], 1), "vs naive last-year",
      round(validation["availability"]["rmse_games_naive_last_year"], 1), "corr", round(validation["availability"]["corr"], 3))

# ---------------------------------------------------------------- ADP-aware edges (per game + season total)
ac = labeled[labeled["yr"].between(2017, 2023)]
tc = p[pool_ok & p["tot_next"].notna() & p["yr"].between(2017, 2023)].copy()
cv, cvt = [], []
for ty in range(2020, 2024):
    tr, te = ac[ac["yr"] < ty], ac[ac["yr"] == ty].copy()
    te["adp_fpg"] = ridge(1).fit(tr[ADPF], tr["fpg_next"]).predict(te[ADPF])
    te["exp_fpg"] = ridge(30).fit(tr[ADPF + FULL], tr["fpg_next"]).predict(te[ADPF + FULL])
    cv.append(te)
    ttr, tte = tc[tc["yr"] < ty], tc[tc["yr"] == ty].copy()
    tte["adp_tot"] = ridge(1).fit(ttr[ADPF], ttr["tot_next"]).predict(tte[ADPF])
    tte["exp_tot"] = ridge(30).fit(ttr[ADPF + FULL + GPF[:3]], ttr["tot_next"]).predict(tte[ADPF + FULL + GPF[:3]])
    # per-game edge model applied to the same (all-availability) rows, so the two edges are comparable
    tte["adp_fpg"] = ridge(1).fit(tr[ADPF], tr["fpg_next"]).predict(tte[ADPF])
    tte["exp_fpg"] = ridge(30).fit(tr[ADPF + FULL], tr["fpg_next"]).predict(tte[ADPF + FULL])
    cvt.append(tte)
cv, cvt = pd.concat(cv), pd.concat(cvt)
cv["edge"] = cv["exp_fpg"] - cv["adp_fpg"]
cv["surprise"] = cv["fpg_next"] - cv["adp_fpg"]
cvt["edge_tot"] = cvt["exp_tot"] - cvt["adp_tot"]
cvt["edge_pg"] = cvt["exp_fpg"] - cvt["adp_fpg"]
cvt["sur_tot"] = cvt["tot_next"] - cvt["adp_tot"]


def adp_metrics(d):
    top = d.sort_values("edge", ascending=False).head(max(1, int(len(d) * .10)))
    return {"n": int(len(d)), "rmse_adp_only": rm(d["fpg_next"], d["adp_fpg"]), "rmse_adp_plus_model": rm(d["fpg_next"], d["exp_fpg"]),
            "corr_edge_vs_surprise": float(d["edge"].corr(d["surprise"])), "top10_mean_surprise": float(top["surprise"].mean()),
            "top10_hit_5": float((top["surprise"] >= 5).mean()), "base_hit_5": float((d["surprise"] >= 5).mean())}


validation["adp"] = {"listed": adp_metrics(cv[cv["has_adp"] == 1]), "unlisted": adp_metrics(cv[cv["has_adp"] == 0])}
print("ADP-aware per-game (ADP-listed):", {k_: round(v, 3) for k_, v in validation["adp"]["listed"].items()})
lt = cvt[cvt["has_adp"] == 1]
k = max(1, int(len(lt) * .10))
top_tot = lt.sort_values("edge_tot", ascending=False).head(k)
top_pg = lt.sort_values("edge_pg", ascending=False).head(k)
validation["total_edge"] = {
    "n": int(len(lt)), "corr_total_edge_vs_total_surprise": float(lt["edge_tot"].corr(lt["sur_tot"])),
    "corr_pergame_edge_vs_total_surprise": float(lt["edge_pg"].corr(lt["sur_tot"])),
    "top10_total_surprise_by_total_edge": float(top_tot["sur_tot"].mean()),
    "top10_total_surprise_by_pergame_edge": float(top_pg["sur_tot"].mean()),
    "rmse_total_adp_only": rm(lt["tot_next"], lt["adp_tot"]), "rmse_total_adp_plus_model": rm(lt["tot_next"], lt["exp_tot"]),
}
print("TOTAL-points edge (ADP-listed):", {k_: round(v, 3) for k_, v in validation["total_edge"].items()})
adp_only_model = ridge(1).fit(ac[ADPF], ac["fpg_next"])
adp_full_model = ridge(30).fit(ac[ADPF + FULL], ac["fpg_next"])
adp_tot_model = ridge(1).fit(tc[ADPF], tc["tot_next"])
exp_tot_model = ridge(30).fit(tc[ADPF + FULL + GPF[:3]], tc["tot_next"])

# ---------------------------------------------------------------- breakout persistence path + "should odds change VOR?" test
delta_path = {}
for name, (mask, lab) in POOLS.items():
    c = labeled[mask]
    path = []
    for h in (1, 2, 3, 4):
        t = c[c[f"fpg_next{h}"].notna()]
        d = t[f"fpg_next{h}"] - t["fpg"]
        path.append(float(d[t[lab] == 1].mean() - d[t[lab] == 0].mean()))
    delta_path[name] = path
validation["delta_path"] = delta_path
print("breakout jump persistence (pts/g vs non-breakers, years +1..+4):", {k_: [round(x, 1) for x in v] for k_, v in delta_path.items()})

vt = {}
for h in (1, 2, 3):
    tgt = f"fpg_next{h}"
    base_e, adj_e, full_e = [], [], []
    for ty in range(2014, 2026 - h):
        tr = labeled[(labeled["yr"] < ty) & labeled[tgt].notna()]
        te = labeled[(labeled["yr"] == ty) & labeled[tgt].notna()].copy()
        if len(te) < 30:
            continue
        b_pred = ridge(10).fit(tr[BASE], tr[tgt]).predict(te[BASE])
        f_pred = ridge(10).fit(tr[FULL], tr[tgt]).predict(te[FULL])
        excess = np.zeros(len(te))
        for name, (mask, lab) in POOLS.items():
            trp = tr[tr.index.isin(labeled[mask].index)]
            sel = (te["fpg"] < Q).to_numpy() if name == "emergence" else (te["fpg"] >= Q).to_numpy()
            if not sel.any():
                continue
            pf = lr().fit(trp[FULL], trp[lab]).predict_proba(te.loc[sel, FULL])[:, 1]
            pb = lr().fit(trp[BASE], trp[lab]).predict_proba(te.loc[sel, BASE])[:, 1]
            dd = trp[tgt] - trp["fpg"]
            delta = float(dd[trp[lab] == 1].mean() - dd[trp[lab] == 0].mean())
            excess[sel] = (pf - pb) * delta
        base_e += list(te[tgt] - b_pred); adj_e += list(te[tgt] - (b_pred + excess)); full_e += list(te[tgt] - f_pred)
    z = lambda e: float(np.sqrt(np.mean(np.square(e))))
    vt[f"h{h}"] = {"rmse_base": z(base_e), "rmse_base_plus_breakout_odds": z(adj_e), "rmse_full_features": z(full_e), "n": len(base_e)}
    print(f"VOR-fold-in accuracy test, {h}yr ahead: base {z(base_e):.3f} | + breakout odds {z(adj_e):.3f} | full-features ridge {z(full_e):.3f}")
validation["vor_test"] = vt

# ---------------------------------------------------------------- live scoring
live_yr = int(p["yr"].max())
live_all = p[p["yr"] == live_yr].copy()
espn = adp_all[adp_all["season_id"] == live_yr + 2].copy()
espn = espn[espn["PLAYER_ID"].notna()].copy()
espn["PLAYER_ID"] = espn["PLAYER_ID"].astype(int)
ABBR_FIX = {"PHL": "PHI", "PHO": "PHX", "NOR": "NOP", "NO": "NOP"}
espn["abbr"] = espn["pro_team"].replace(ABBR_FIX)
abbr_to_id = p.dropna(subset=["team_id"]).drop_duplicates("team").set_index("team")["team_id"].to_dict()
espn["team_id_next"] = espn["abbr"].map(abbr_to_id)
espn = espn.sort_values("adp").drop_duplicates("PLAYER_ID")
live_all = live_all.drop(columns=["team_id_next", "team_next", "moved", "dest_net_usg", "dest_vac_min", "coach_change",
                                  "adp", "ladp", "l2", "l3", "has_adp"], errors="ignore")
live_all = live_all.merge(espn[["PLAYER_ID", "team_id_next", "abbr", "adp"]], on="PLAYER_ID", how="left")
live_all = live_all.rename(columns={"abbr": "team_next"})
live_all["totmin"] = live_all["totmin"].fillna(0)
live_all = situation_features(live_all, load_coach_sets())
live_all["ladp"] = np.log(live_all["adp"].fillna(200)); live_all["has_adp"] = live_all["adp"].notna().astype(int)
live_all["l2"], live_all["l3"] = live_all["ladp"] ** 2, live_all["ladp"] ** 3
live_all["young"] = (live_all["AGE"] <= 24) * (24 - live_all["AGE"])
live = live_all[(live_all["GP"] >= 20) & (live_all["mpg"] >= 10)].copy()

live["tier"] = np.where(live["fpg"] < Q, "emergence", "elevation")
live["p_break"] = np.nan
live["p_baseline"] = np.nan
for name, (mask, lab) in POOLS.items():
    sel = live["tier"] == name
    live.loc[sel, "p_break"] = models[name].predict_proba(live.loc[sel, FULL])[:, 1]
    tr = labeled[mask]
    live.loc[sel, "p_baseline"] = lr().fit(tr[BASE], tr[lab]).predict_proba(live.loc[sel, BASE])[:, 1]


def score_extras(df):
    """projection + range, bust odds, games, per-game edge and total edge (ADP-listed only)."""
    df = df.copy()
    df["proj_fpg"] = proj_model.predict(df[FULL])
    lo = df["tier"].map({k_: v[0] for k_, v in QUANT.items()}); hi = df["tier"].map({k_: v[1] for k_, v in QUANT.items()})
    df["proj_lo"], df["proj_hi"] = df["proj_fpg"] + lo, df["proj_fpg"] + hi
    df["p_bust"] = np.where(df["tier"] == "elevation", bust_model.predict_proba(df[FULL])[:, 1], np.nan)
    df["proj_gp"] = (gp_model.predict(df[GPF]) * 82).clip(0, 82)
    listed = df["has_adp"] == 1
    df["adp_fpg"] = adp_only_model.predict(df[ADPF])
    df["exp_fpg"] = adp_full_model.predict(df[ADPF + FULL])
    df["edge_fpg"] = (df["exp_fpg"] - df["adp_fpg"]).where(listed)
    df["edge_total"] = pd.Series(exp_tot_model.predict(df[ADPF + FULL + GPF[:3]]) - adp_tot_model.predict(df[ADPF]), index=df.index).where(listed)
    return df


live = score_extras(live)


def why(r):
    out = []
    if r["moved"] == 1 and pd.notna(r.get("team_next")):
        out.append(f"new team ({r['team_next']})")
    if r["dest_net_usg"] >= 0.03:
        out.append("usage opened up on his team")
    if r["late_mpg"] - r["mpg"] >= 2:
        out.append(f"minutes up late ({r['mpg']:.0f}->{r['late_mpg']:.0f})")
    if r["late_fpg"] - r["fpg"] >= 3:
        out.append(f"production up late ({r['fpg']:.0f}->{r['late_fpg']:.0f})")
    if r["fp_per36"] >= 44:
        out.append(f"strong per-36 rate ({r['fp_per36']:.0f})")
    if r["AGE"] <= 22:
        out.append("very young")
    if r["draft_pick"] <= 14:
        out.append(f"lottery pedigree (#{int(r['draft_pick'])})")
    return "; ".join(out[:4])


live["why"] = live.apply(why, axis=1)
bio = pd.read_csv(D / "player_bio.csv")[["PERSON_ID", "ROSTER_STATUS"]].rename(columns={"PERSON_ID": "PLAYER_ID"})
live = live.merge(bio, on="PLAYER_ID", how="left")
live = live[live["ROSTER_STATUS"] == 1]
cols = ["PLAYER_ID", "PLAYER_NAME", "AGE", "tier", "fpg", "p_break", "p_baseline", "p_bust", "proj_fpg", "proj_lo", "proj_hi",
        "proj_gp", "adp", "adp_fpg", "exp_fpg", "edge_fpg", "edge_total", "team_next", "moved", "why"]
out = live.sort_values("p_break", ascending=False)[cols].rename(columns={"PLAYER_NAME": "player", "AGE": "age", "fpg": "fpg_last"})
out.to_csv(D / "breakout_candidates.csv", index=False)
validation["sit_features"] = SIT
validation["live_season"] = f"{live_yr + 1}-{str(live_yr + 2)[-2:]}"
(D / "breakout_validation.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")
print(f"\nlive candidates: {len(out)} ({(out['tier']=='emergence').sum()} emergence, {(out['tier']=='elevation').sum()} elevation)")
show = ["player", "age", "fpg_last", "p_break", "proj_fpg", "proj_lo", "proj_hi", "proj_gp", "adp", "edge_fpg", "edge_total"]
print("\n-- top bust risk (elevation pool)")
print(out.sort_values("p_bust", ascending=False).head(10)[["player", "age", "fpg_last", "p_bust", "proj_fpg", "adp", "edge_fpg"]].round(2).to_string(index=False))
print("\n-- biggest season-total edge vs ADP")
print(out.sort_values("edge_total", ascending=False).head(10)[show].round(1).to_string(index=False))

# ---------------------------------------------------------------- walk-forward history for the dashboard's season selector
hist_rows = []
for ty in range(2014, live_yr):
    hp = p[(p["yr"] == ty) & pool_ok].copy()
    if hp.empty:
        continue
    hp["tier"] = np.where(hp["fpg"] < Q, "emergence", "elevation")
    hp["p_break"] = np.nan
    hp["p_bust"] = np.nan
    for name, (mask, lab) in POOLS.items():
        tr = labeled[mask & (labeled["yr"] < ty)]
        sel = hp["tier"] == name
        if len(tr) < 100 or not sel.any():
            continue
        hp.loc[sel, "p_break"] = lr().fit(tr[FULL], tr[lab]).predict_proba(hp.loc[sel, FULL])[:, 1]
    btr = labeled[(labeled["fpg"] >= Q) & (labeled["yr"] < ty)]
    selb = hp["tier"] == "elevation"
    if len(btr) >= 100 and selb.any():
        hp.loc[selb, "p_bust"] = lr().fit(btr[FULL], btr["bust"]).predict_proba(hp.loc[selb, FULL])[:, 1]
    ptr = labeled[labeled["yr"] < ty]
    hp["proj_fpg"] = ridge(10).fit(ptr[FULL], ptr["fpg_next"]).predict(hp[FULL])
    hp["proj_lo"] = hp["proj_fpg"] + hp["tier"].map({k_: v[0] for k_, v in QUANT.items()})
    hp["proj_hi"] = hp["proj_fpg"] + hp["tier"].map({k_: v[1] for k_, v in QUANT.items()})
    hp["edge_fpg"] = np.nan
    if 2020 <= ty <= 2023:
        atr = ac[ac["yr"] < ty]
        e = (ridge(30).fit(atr[ADPF + FULL], atr["fpg_next"]).predict(hp[ADPF + FULL])
             - ridge(1).fit(atr[ADPF], atr["fpg_next"]).predict(hp[ADPF]))
        hp["edge_fpg"] = np.where(hp["has_adp"] == 1, e, np.nan)
    hp["why"] = hp.apply(why, axis=1)
    valid = hp["fpg_next"].notna() & (hp["GP_next"] >= 30)
    hit_e = (hp["fpg_next"] >= Q - TOL) & (hp["d_next"] >= 6)
    hit_l = hp["d_next"] >= 6
    hp["hit"] = np.where(~valid, np.nan, np.where(hp["tier"] == "emergence", hit_e, hit_l).astype(float))
    hp["bust_hit"] = np.where(~valid | (hp["tier"] != "elevation"), np.nan, (hp["d_next"] <= -6).astype(float))
    hist_rows.append(hp[["yr", "PLAYER_ID", "PLAYER_NAME", "AGE", "team", "tier", "fpg", "p_break", "p_bust", "proj_fpg", "proj_lo",
                         "proj_hi", "adp", "edge_fpg", "why", "fpg_next", "GP_next", "d_next", "hit", "bust_hit"]])
hist = pd.concat(hist_rows).rename(columns={"PLAYER_NAME": "player", "AGE": "age", "fpg": "fpg_last"})
hist = hist[hist["p_break"].notna()]
hist.to_csv(D / "breakout_history.csv", index=False)
print(f"\nhistory: {len(hist)} rows across {hist['yr'].nunique()} prediction seasons")
