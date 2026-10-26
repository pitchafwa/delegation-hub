"""Short-term program, Phase 4 (RESEARCH_short_term_projection_plan.md): model comparison on the same harness.
  A  corrections to DELCO (v1 style) with every signal that survived batches 1-4
  B  structural: minutes x per-minute rate, each corrected separately
  C  gradient-boosted trees on the full feature set (check on what the simple model misses)
  D  stack of A and C by horizon
plus the per-game terms (opponent, home/away) applied on top of each. Fit on TRAIN, judged on TEST.
Run from ingest/:  .venv/Scripts/python research/st_phase4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_harness as H                                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
D, P, T = H.D, H.P, H.T
OUT = {}
COLS = ["game 1", "games 2-3", "games 4-6", "games 7-9", "all (g1-6)"]
SUBS = ("all", "role up", "role down", "teammate out (30+)", "rookies", "back from 5+ days off", "stars (DELCO 40+)", "fringe (DELCO < 20)", "first month")
rate = P["delco"] / P["delco_min"].clip(lower=8)
g = P

# ---------------------------------------------------------------- features kept from batches 1-4
b1 = H.best_level_feats()
stint = (g["new_team"] | (g["season"] != g.groupby("pid")["season"].shift(1))).groupby(g["pid"]).cumsum()
fp_stint = g.groupby([g["pid"], stint])["fp"].transform(lambda s: s.expanding().mean())
post = ((g["games_since_trade"] <= 10) & g["traded_in_season"].groupby(g["pid"]).cumsum().gt(0)).astype(float)
gap_prev = g["days_since_prev"].where(g["gp_season"] > 1, 0)
ret_start = gap_prev >= 6
since_ret = g.groupby([g["pid"], ret_start.groupby(g["pid"]).cumsum()]).cumcount()
ret25 = (ret_start.groupby(g["pid"]).cumsum().gt(0) & since_ret.between(1, 4) & ~ret_start).astype(float)
early = (g["season_day"] <= 30).astype(float)
late = (g["season_day"] >= 150).astype(float)
first = g.groupby("pid")["season"].transform("min")
rook = ((first >= "2015-16") & (g["season"] == first)).astype(float)
raw = pd.read_parquet(D / "st_panel.parquet", columns=["pid", "date", "fg3m", "tov", "info_id"]).sort_values(["pid", "date"])
hot = {}
for c, v in {"threes": raw["fg3m"], "turnovers": -raw["tov"]}.items():
    s5 = v.groupby(raw["pid"]).transform(lambda s: s.rolling(5, min_periods=5).mean())
    s40 = v.groupby(raw["pid"]).transform(lambda s: s.rolling(40, min_periods=15).mean())
    hot["hot_" + c] = pd.Series((s5 - s40).to_numpy(), index=raw["info_id"]).reindex(P.index)
A_FEATS = {**b1, "delco_lvl": P["delco"] - 24.0,
           "post": post, "post_x_newgap": post * (fp_stint - g["delco"]),
           "ret1": ret_start.astype(float), "ret25": ret25,
           "early": early, "early_x_fp": early * b1["fp_gap"], "early_x_min": early * b1["min_gap"], "late": late, "late_star": late * (g["delco"] >= 35),
           "rook": rook, "rook_x_min": rook * b1["min_gap"], **hot}

preds = {"v1 (live)": H.baselines()["v1 (live since 10-26)"], "season DELCO": P["delco"]}
preds["batch 1 winner"] = H.predict_by_k(b1, H.fit_by_k(b1))
coA = H.fit_by_k(A_FEATS)
preds["A corrections, all kept"] = H.predict_by_k(A_FEATS, coA)
OUT["A_coefs"] = {str(b): dict(zip(["const", *A_FEATS], [round(float(x), 4) for x in coA[b]])) for b in coA}

# ---------------------------------------------------------------- B structural: minutes and per-minute rate
TP = H.TP
tr = TP[TP["season"].isin(H.TRAIN) & (TP["min"] > 0)]
mfe = {"min_gap_raw": P["min_ew2"] - P["delco_min"], "min_l10_gap": P["min_l10"] - P["delco_min"], "start_l5": P["start_l5"], "early": early, "late": late,
       "late_star": late * (g["delco"] >= 35), "ret1": ret_start.astype(float), "ret25": ret25, "post": post, "rook": rook}
rfe = {"rate_gap": P["rate_l5"] - rate, "fp_gap_per_min": b1["fp_gap"] / P["delco_min"].clip(lower=8), "lvl": rate - 1.1, "post": post, "rook": rook,
       "hot3": hot["hot_threes"] / P["delco_min"].clip(lower=8), "hottov": hot["hot_turnovers"] / P["delco_min"].clip(lower=8)}
coB = {}
for b in range(len(H.KB)):
    t = tr[tr["kb"] == b]
    for tgt, fe, y, w in [("min", mfe, t["min"].to_numpy() - t["info_id"].map(P["delco_min"]).to_numpy(), None),
                          ("rate", rfe, t["fp"].to_numpy() / t["min"].clip(lower=1).to_numpy() - t["info_id"].map(rate).to_numpy(), t["min"].to_numpy())]:
        X = np.column_stack([np.ones(len(t))] + [t["info_id"].map(f).to_numpy() for f in fe.values()])
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        ww = np.ones(ok.sum()) if w is None else np.sqrt(w[ok])
        coB[(tgt, b)] = np.linalg.lstsq(X[ok] * ww[:, None], y[ok] * ww, rcond=None)[0]
outB = {}
for k in range(1, 11):
    b = H.kb_of(min(k, 9))
    mn = P["delco_min"] + coB[("min", b)][0] + sum(c * f.reindex(P.index).fillna(0) for c, f in zip(coB[("min", b)][1:], mfe.values()))
    rt = rate + coB[("rate", b)][0] + sum(c * f.reindex(P.index).fillna(0) for c, f in zip(coB[("rate", b)][1:], rfe.values()))
    outB[k] = mn.clip(lower=0) * rt
# the simple product is biased (minutes and rate move together); fix the bias per horizon on TRAIN
sB = pd.concat(outB, names=["k"]).swaplevel().sort_index()
sB.index.names = ["info_id", "k"]
trp = TP[TP["season"].isin(H.TRAIN)]
pv = pd.Series(sB.reindex(list(zip(trp["info_id"], trp["k"]))).to_numpy(), index=trp.index)
bias = (trp["fp"] - pv).groupby(trp["kb"]).mean()
sB = sB + pd.Series(sB.index.get_level_values("k").map(lambda k: bias[H.kb_of(min(k, 9))]), index=sB.index)
preds["B minutes x rate"] = sB

# ---------------------------------------------------------------- C gradient-boosted trees (target: points minus DELCO, k as a feature)
import lightgbm as lgb
FEAT = ["delco", "delco_min", "delco_career", "delco_career_min", "age", "fp_l1", "fp_l3", "fp_l5", "fp_l10", "fp_l15", "fp_ew2", "fp_ew4", "fp_ew8",
        "min_l1", "min_l3", "min_l5", "min_l10", "min_l15", "min_ew2", "min_ew4", "min_ew8", "min_sd10", "start_l1", "start_l5", "start_prev10",
        "use_l5", "use_l40", "fg_l5", "fg_l100", "fg3_l5", "fg3_l100", "ft_l5", "ft_l100", "pf_l3", "rate_l5", "days_since_prev", "gp_season",
        "games_with_team", "games_since_trade", "season_day", "out_fp"]
Xp = P[FEAT].astype(float).assign(post=post, ret1=ret_start.astype(float), ret25=ret25, rook=rook, fp_stint=fp_stint, **hot)
trr = TP[TP["season"].isin(H.TRAIN)]
Xtr = Xp.reindex(trr["info_id"]).reset_index(drop=True).assign(k=trr["k"].to_numpy())
ytr = trr["fp"].to_numpy() - trr["info_id"].map(P["delco"]).to_numpy()
gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=400, subsample=0.7, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
gbm.fit(Xtr, ytr)
outC = {}
for k in range(1, 11):
    outC[k] = pd.Series(gbm.predict(Xp.assign(k=k)), index=P.index) + P["delco"]
sC = pd.concat(outC, names=["k"]).swaplevel().sort_index()
sC.index.names = ["info_id", "k"]
preds["C boosted trees"] = sC
imp = pd.Series(gbm.booster_.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
print("boosted trees, most useful inputs:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(10).items()))

# ---------------------------------------------------------------- D stack of A and C (weights per horizon, fit on the last 2 train seasons)
sA = preds["A corrections, all kept"]
st_tr = TP[TP["season"].isin(["2020-21", "2021-22"])]
wD = {}
for b in range(len(H.KB)):
    t = st_tr[st_tr["kb"] == b]
    a = sA.reindex(list(zip(t["info_id"], t["k"]))).to_numpy()
    c = sC.reindex(list(zip(t["info_id"], t["k"]))).to_numpy()
    ok = np.isfinite(a) & np.isfinite(c)
    ws = np.linspace(0, 1, 21)
    err = [np.mean((t["fp"].to_numpy()[ok] - (w * c[ok] + (1 - w) * a[ok])) ** 2) for w in ws]
    wD[b] = float(ws[int(np.argmin(err))])
kk = sA.index.get_level_values("k").map(lambda k: wD[H.kb_of(min(k, 9))])
preds["D stack A+C"] = sA * (1 - kk.to_numpy()) + sC.reindex(sA.index) * kk.to_numpy()
print("stack weight on the trees by horizon (game 1, 2-3, 4-6, 7-9):", [wD[b] for b in range(4)])

lt = H.level_table(preds, subsets=SUBS)
print("\nLEVEL (typical miss on test seasons, points per game he plays)")
print(lt[COLS].round(3).to_string())
print("\nby situation, games 1-6: typical miss / lean")
cols = [c for c in lt.columns if c.endswith("(g1-6)") and not c.startswith("all")]
tab = pd.DataFrame({c.replace(" (g1-6)", ""): lt[c].round(2).astype(str) + " / " + lt[c.replace("(g1-6)", "lean")].map(lambda v: f"{v:+.2f}") for c in cols})
print(tab.to_string())
OUT["level"] = lt.round(4).reset_index().to_dict("records")
OUT["stack_w"] = wD

# ---------------------------------------------------------------- per-game terms (opponent, home/away), fit on TRAIN residuals of each model
raw2 = pd.read_parquet(D / "st_panel.parquet", columns=["pid", "season", "gid", "date", "team", "fp", "pts", "info_id"])
tp = raw2.groupby(["gid", "team"]).agg(tfp=("fp", "sum"), tpts=("pts", "sum")).reset_index()
o = tp.merge(tp.rename(columns={"team": "opp", "tfp": "ofp", "tpts": "opts"}), on="gid")
o = o[o["team"] != o["opp"]]
tgm = raw2.drop_duplicates(["team", "gid"])[["team", "gid", "date", "season"]].sort_values(["team", "date"]).reset_index(drop=True)
tgm["i"] = tgm.groupby("team").cumcount()
tgm = tgm.merge(o[["gid", "team", "opp", "ofp"]], on=["gid", "team"], how="left")
tgm["fpa_r"] = tgm.groupby(["team", "season"])["ofp"].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())
tgm["fpa_rel"] = tgm["fpa_r"] / tgm["season"].map(tgm.groupby("season")["ofp"].mean()) - 1
gl = D / "game_logs"
hm = [pd.read_csv(f, usecols=["gameId", "matchup", "teamTricode"], dtype={"gameId": str}).rename(columns={"gameId": "gid", "teamTricode": "team"})
      for f in sorted(gl.glob("regular_season_box_scores_2010_2024_part_*.csv"))]
hm += [pd.read_csv(f, usecols=["GAME_ID", "MATCHUP", "TEAM_ABBREVIATION"], dtype={"GAME_ID": str}).rename(columns={"GAME_ID": "gid", "MATCHUP": "matchup", "TEAM_ABBREVIATION": "team"})
       for f in sorted(gl.glob("nba_api_*.csv"))]
hm = pd.concat(hm).drop_duplicates(["gid", "team"])
hm["gid"] = hm["gid"].astype(int)
home = hm.assign(h=hm["matchup"].str.contains("vs.", regex=False).astype(float)).set_index(["gid", "team"])["h"]
tgm["home"] = pd.MultiIndex.from_arrays([tgm["gid"].astype(int), tgm["team"]]).map(home)
pos = pd.Series(pd.MultiIndex.from_arrays([P["team"], P["gid"]]).map(tgm.set_index(["team", "gid"])["i"]), index=P.index)
tt = T.copy()
tt["team"] = tt["info_id"].map(P["team"])
j = pd.MultiIndex.from_arrays([tt["team"], tt["info_id"].map(pos) + tt["k"]])
gm = tgm.set_index(["team", "i"])
oj = pd.MultiIndex.from_arrays([j.map(gm["opp"]), j.map(gm["gid"])])
tt["opp_fpa"] = oj.map(tgm.set_index(["team", "gid"])["fpa_rel"])
tt["home"] = j.map(gm["home"])
GF = np.column_stack([tt["opp_fpa"].fillna(0).to_numpy(), (tt["home"].fillna(0.5) - 0.5).to_numpy()])
idx2 = list(zip(tt["info_id"], tt["k"]))
full = {}
for name in ["A corrections, all kept", "B minutes x rate", "C boosted trees", "D stack A+C"]:
    pv = preds[name].reindex(idx2).to_numpy()
    res = tt["fp"].to_numpy() - pv
    trm = (tt["played"] & tt["season"].isin(H.TRAIN)).to_numpy() & np.isfinite(res) & np.isfinite(pv)
    bet = np.linalg.lstsq(GF[trm] * pv[trm, None], res[trm], rcond=None)[0]
    full[name + " + opponent, home"] = pd.Series(pv * (1 + GF @ bet), index=pd.MultiIndex.from_arrays([tt["info_id"], tt["k"]], names=["info_id", "k"]))
    OUT.setdefault("game_terms", {})[name] = [float(x) for x in bet]
lt2 = H.level_table({"v1 (live)": preds["v1 (live)"], **full}, subsets=SUBS)
print("\nwith opponent and home/away for each upcoming game:")
print(lt2[COLS + ["stars (DELCO 40+) lean", "rookies lean", "role up lean", "first month lean"]].round(3).to_string())
print("game terms (opponent per +100% allowed, home vs away):", {k: [round(v, 3) for v in b] for k, b in OUT["game_terms"].items()})

# ---------------------------------------------------------------- WEEK with the chance-to-play model from batch 1
import lightgbm as lgb  # noqa: F811
tg = T.merge(P[["team", "date"]].reset_index(), on="info_id").sort_values(["info_id", "k"])
tg["b2b"] = (tg.groupby("info_id")["days_ahead"].diff().fillna(tg["days_ahead"]) <= 1).astype(int)
P["avail_l10"] = (P.groupby("pid")["date"].transform(lambda s: s.diff().dt.days.rolling(10, min_periods=3).mean())).rpow(-1).mul(2.0).clip(upper=1.0)
XA = pd.DataFrame({"k": tg["k"], "days_ahead": tg["days_ahead"], "b2b": tg["b2b"], "delco": tg["info_id"].map(P["delco"]), "age": tg["info_id"].map(P["age"]),
                   "min_l5": tg["info_id"].map(P["min_l5"]), "min_sd10": tg["info_id"].map(P["min_sd10"]), "days_since_prev": tg["info_id"].map(P["days_since_prev"]),
                   "season_day": tg["info_id"].map(P["season_day"]), "start_l5": tg["info_id"].map(P["start_l5"]), "avail": tg["info_id"].map(P["avail_l10"])}, index=tg.index)
ya = tg["played"].astype(int)
am = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
trA = tg["season"].isin(H.TRAIN).to_numpy()
am.fit(XA[trA], ya[trA])
pA = pd.Series(am.predict_proba(XA)[:, 1], index=tg.index)
fa = H.flat_avail()
best = "A corrections, all kept + opponent, home"     # simplest within noise of the trees, and without their leans (stars +1.0, role down -0.8)
wk = H.week_table({"v1 + flat (live today)": preds["v1 (live)"], "season DELCO + flat": P["delco"], f"{best} + flat": full[best], f"{best} + chance model": full[best], "C trees + chance model": full["C boosted trees + opponent, home"]},
                  {"v1 + flat (live today)": fa, "season DELCO + flat": fa, f"{best} + flat": fa, f"{best} + chance model": pA, "C trees + chance model": pA})
print("\nWEEK (expected points over the next 7 days vs actual):")
print(wk.round(2).to_string())
OUT["week"] = wk.round(3).reset_index().to_dict("records")
OUT["best"] = best
json.dump(OUT, open(D / "st_phase4.json", "w"), indent=1, default=str)

# diagnostics: which part moves the 7-day lean
pd.DataFrame({"pA": pA}).to_parquet(D / "st_phase4_avail.parquet")
diag = {"batch 1 winner": preds["batch 1 winner"], "A (no game terms)": preds["A corrections, all kept"], "A + opponent, home": full[best]}
print("\n7-day lean with the chance model, by level:")
print(H.week_table(diag, {k: pA for k in diag}).round(2).to_string())
print("chance model on test: mean", round(float(pA[tg['season'].isin(H.TEST)].mean()), 3), "actual", round(float(ya[tg['season'].isin(H.TEST).to_numpy()].mean()), 3))
for k_ in [1, 3, 5]:
    m_ = tg["season"].isin(H.TEST) & (tg["k"] == k_)
    print(f"   game {k_}: predicted {pA[m_].mean():.3f}, actual {ya[m_].mean():.3f}")
