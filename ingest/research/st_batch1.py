"""Short-term program, signal batch 1 (RESEARCH_short_term_projection_plan.md): tests 1 (minutes trend length and how long a change lasts),
6+7 (teammates listed out / returning, from the official injury report), 19 (chance to play each upcoming game).
All judged with st_harness.py on held-out seasons. Run from ingest/:  .venv/Scripts/python research/st_batch1.py   (writes data/st_batch1.json)
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import st_harness as H                                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
D = H.D
P, T = H.P, H.T
OUT = {}


def nk(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s))


rate = P["delco"] / P["delco_min"].clip(lower=8)
Tp = T[T["played"]].copy()
KB = [(1, 1), (2, 3), (4, 6), (7, 9)]
kb_of = lambda k: next(i for i, (lo, hi) in enumerate(KB) if lo <= k <= hi)
Tp["kb"] = Tp["k"].map(kb_of)


def fit_by_k(feats, train_seasons, data=None):
    """least squares of (points - DELCO) on the features, separately for each horizon bucket. feats: {name: Series by info_id}"""
    tp = Tp if data is None else data
    tr = tp[tp["season"].isin(train_seasons)]
    coefs = {}
    for b in range(len(KB)):
        t = tr[tr["kb"] == b]
        X = np.column_stack([np.ones(len(t))] + [t["info_id"].map(f).to_numpy() for f in feats.values()])
        y = t["fp"].to_numpy() - t["info_id"].map(P["delco"]).to_numpy()
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        coefs[b] = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return coefs


def predict_by_k(feats, coefs):
    idx = P.index
    out = {}
    for k in range(1, 10):
        c = coefs[kb_of(k)]
        v = P["delco"] + c[0] + sum(c[i + 1] * f.reindex(idx).fillna(0.0) for i, f in enumerate(feats.values()))
        out[k] = v
    s = pd.concat(out, names=["k"]).swaplevel().sort_index()
    s.index.names = ["info_id", "k"]
    return s


# ======================================================================= TEST 1: minutes trend length, and how long a change lasts
print("TEST 1: which recent-minutes window, and how much of a change carries to each distance ahead\n")
cands = {}
for mw in ["min_l1", "min_l3", "min_l5", "min_l10", "min_ew2", "min_ew4", "min_ew8"]:
    for fw in ["fp_l3", "fp_ew4", "fp_l10"]:
        feats = {"fp_gap": P[fw] - P["delco"], "min_gap": (P[mw] - P["delco_min"]) * rate}
        cands[(mw, fw)] = (feats, fit_by_k(feats, H.TRAIN))
lt_rows = {}
for (mw, fw), (feats, co) in cands.items():
    lt_rows[f"minutes {mw} + points {fw}"] = predict_by_k(feats, co)
v1 = H.baselines()["v1 (live since 10-26)"]
lt = H.level_table({"v1 (live)": v1, **lt_rows})
best1 = lt["games 2-3"].idxmin()
print(lt[["game 1", "games 2-3", "games 4-6", "games 7-9", "role up (g1-6)", "stars (DELCO 40+) (g1-6)", "stars (DELCO 40+) lean"]].sort_values("games 2-3").head(8).round(2).to_string())
mw_b, fw_b = best1.replace("minutes ", "").split(" + points ")
feats_b, co_b = cands[(mw_b, fw_b)]
print(f"\nbest: {best1}. How much of a gap carries, by distance ahead (constant / points gap / minutes gap in points):")
for b, (lo, hi) in enumerate(KB):
    print(f"   games {lo}-{hi}: {co_b[b][0]:+.2f} / {co_b[b][1]:.3f} / {co_b[b][2]:.3f}")
# level-dependent: do stars need less of the adjustment?
star = (P["delco"] >= 40).astype(float)
feats_s = {**feats_b, "fp_gap_star": feats_b["fp_gap"] * star, "min_gap_star": feats_b["min_gap"] * star, "star": star}
co_s = fit_by_k(feats_s, H.TRAIN)
lev_s = predict_by_k(feats_s, co_s)
lt2 = H.level_table({"best window": predict_by_k(feats_b, co_b), "best window + star terms": lev_s})
print("\nstars treated separately:")
print(lt2[["game 1", "games 2-3", "games 4-6", "stars (DELCO 40+) (g1-6)", "stars (DELCO 40+) lean", "rookies (g1-6)", "rookies lean"]].round(2).to_string())
OUT["test1"] = {"best": best1, "coefs": {str(b): [float(x) for x in co_b[b]] for b in co_b}, "star_coefs": {str(b): [float(x) for x in co_s[b]] for b in co_s}}

# ======================================================================= injury-report features (2021-22 on)
print("\nbuilding injury-report features ...", flush=True)
rep = pd.read_csv(D / "injury_reports_hourly.csv", usecols=["report_date", "slot", "game_date", "player_key", "status"])
rep["report_date"] = pd.to_datetime(rep["report_date"])
rep["game_date"] = pd.to_datetime(rep["game_date"])
rep["ord"] = rep["report_date"] + pd.to_timedelta(np.where(rep["slot"] == "05PM", 17, 11), unit="h")
rep = rep.sort_values("ord")
# what was known the morning after a player's game (date d0): the latest listing per player from reports up to the evening of d0, for games on d0 or later
key_by_pid = P.drop_duplicates("pid").set_index("pid")["name"].map(nk)
P["key"] = P["pid"].map(key_by_pid)
dates = sorted(P.loc[P["season"] >= "2021-22", "date"].unique())
snap = {}
rep_by_day = {d: g for d, g in rep.groupby("report_date")}
for d0 in dates:
    d0 = pd.Timestamp(d0)
    parts = [rep_by_day.get(d0 - pd.Timedelta(days=i)) for i in (2, 1, 0)]
    parts = [p for p in parts if p is not None]
    if not parts:
        continue
    r = pd.concat(parts)
    r = r[r["game_date"] >= d0]
    snap[d0] = r.drop_duplicates("player_key", keep="last").set_index("player_key")["status"].to_dict()
P["status_now"] = [snap.get(pd.Timestamp(d), {}).get(k, "none") if s >= "2021-22" else "no data" for d, k, s in zip(P["date"], P["key"], P["season"])]
# rotation teammates at each team game (DELCO minutes 22+, played within 10 days), then who is listed out now / who sat last game but is back
Ps = P.sort_values("date")
team_state, mates = {}, {}
for (team, gid), grp in Ps.groupby(["team", "gid"], sort=False):
    stt = team_state.setdefault(team, {})
    d = grp["date"].iloc[0]
    for r in grp.itertuples():
        stt[r.pid] = (r.date, r.delco, r.delco_min, r.key)
    mates[(team, gid)] = [(p_, v[1], v[3], v[0]) for p_, v in stt.items() if v[2] >= 22 and (d - v[0]).days <= 10]
out_now, back_now = [], []
for r in P.itertuples():
    if r.season < "2021-22":
        out_now.append(np.nan); back_now.append(np.nan); continue
    sn = snap.get(pd.Timestamp(r.date), {})
    o = b = 0.0
    for p_, lvl, key, last in mates.get((r.team, r.gid), []):
        if p_ == r.pid or not np.isfinite(lvl):
            continue
        listed_out = sn.get(key) in ("Out", "Doubtful")
        if listed_out:
            o += lvl
        elif last < r.date:                       # sat the last game but is not listed out now: coming back
            b += lvl
    out_now.append(o); back_now.append(b)
P["out_now"] = out_now
P["back_now"] = back_now
share = P["delco_min"] / 240.0
print(f"  status known for {(P['status_now'] != 'no data').mean():.0%} of info points; teammates listed out (any) {np.mean(np.array(out_now)[np.isfinite(out_now)] > 0):.0%}", flush=True)

# ======================================================================= TEST 6+7: teammates listed out / returning
print("\nTEST 6+7: teammates listed out for upcoming games, and teammates returning (fit 2021-22..2022-23, judged 2023-24..2025-26)")
TR2, TE2 = ["2021-22", "2022-23"], ["2023-24", "2024-25", "2025-26"]
f_base = feats_b
f_inj = {**feats_b, "out_now": P["out_now"] * share, "back_now": P["back_now"] * share}
co_a, co_b2 = fit_by_k(f_base, TR2), fit_by_k(f_inj, TR2)
lt3 = H.level_table({"best window (same split)": predict_by_k(f_base, co_a), "+ teammates out / returning": predict_by_k(f_inj, co_b2)}, split=TE2)
print(lt3[["game 1", "games 2-3", "games 4-6", "teammate out (30+) (g1-6)", "teammate out (30+) lean", "role up (g1-6)", "role up lean"]].round(2).to_string())
print("coefficients by distance ahead (const / pts gap / min gap / out now x share / back now x share):")
for b in co_b2:
    print("   ", [round(float(x), 3) for x in co_b2[b]])
OUT["test67"] = {str(b): [float(x) for x in co_b2[b]] for b in co_b2}

# ======================================================================= TEST 19: chance to play each upcoming game
print("\nTEST 19: chance he plays each upcoming team game (gradient-boosted model vs the flat rule)")
import lightgbm as lgb
tg = H.T.merge(P[["team", "date"]].reset_index(), on="info_id")
# back-to-back for that game: the team's previous game was the day before (from the schedule of team games)
tg = tg.sort_values(["info_id", "k"])
tg["b2b"] = (tg.groupby("info_id")["days_ahead"].diff().fillna(tg["days_ahead"]) <= 1).astype(int)
# his recent availability: share of his team's last 10 games he played (from earlier info points' k=1 outcome is unknown at the time; use games played / days)
P["avail_l10"] = (P.groupby("pid")["date"].transform(lambda s: s.diff().dt.days.rolling(10, min_periods=3).mean())).rpow(-1) * 2.0
P["avail_l10"] = P["avail_l10"].clip(upper=1.0)
cat = {"none": 0, "Available": 0, "Probable": 1, "Questionable": 2, "Doubtful": 3, "Out": 4, "no data": -1}
X = pd.DataFrame({
    "k": tg["k"], "days_ahead": tg["days_ahead"], "b2b": tg["b2b"],
    "status": tg["info_id"].map(P["status_now"]).map(cat).fillna(-1), "delco": tg["info_id"].map(P["delco"]), "age": tg["info_id"].map(P["age"]),
    "min_l5": tg["info_id"].map(P["min_l5"]), "min_sd10": tg["info_id"].map(P["min_sd10"]), "days_since_prev": tg["info_id"].map(P["days_since_prev"]),
    "season_day": tg["info_id"].map(P["season_day"]), "start_l5": tg["info_id"].map(P["start_l5"]), "avail": tg["info_id"].map(P["avail_l10"]),
}, index=tg.index)
y = tg["played"].astype(int)
tr = tg["season"].isin(H.TRAIN).to_numpy()
te = tg["season"].isin(H.TEST).to_numpy()
mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
mdl.fit(X[tr], y[tr])
p_gb = pd.Series(mdl.predict_proba(X)[:, 1], index=tg.index)
flat = H.flat_avail().reindex(tg.index)
at = H.avail_table({"flat 85/81/61%": flat, "model": p_gb})
print(at.round(3).to_string())
# calibration by decile
dfc = pd.DataFrame({"p": p_gb[te], "y": y[te]})
dfc["bin"] = pd.qcut(dfc["p"], 10, duplicates="drop")
print("calibration (predicted vs actual, test):", [(round(a, 2), round(b, 2)) for a, b in dfc.groupby("bin", observed=True)[["p", "y"]].mean().to_numpy()])
imp = pd.Series(mdl.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print("most useful signals:", ", ".join(f"{k} {v / imp.sum():.0%}" for k, v in imp.head(6).items()))

# same model on the injury-report years only (fit 2021-22..2022-23, judged 2023-24..2025-26), with and without report status
P[["status_now", "out_now", "back_now", "avail_l10"]].to_parquet(D / "st_inj_features.parquet")
tr2 = tg["season"].isin(TR2).to_numpy()
p_rep = {}
for lab, cols in {"model, report years, no status": [c for c in X.columns if c != "status"], "model, report years, with status": list(X.columns)}.items():
    m2 = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m2.fit(X.loc[tr2, cols], y[tr2])
    p_rep[lab] = pd.Series(m2.predict_proba(X[cols])[:, 1], index=tg.index)
print("\njudged 2023-24..2025-26:")
print(H.avail_table({"flat 85/81/61%": flat, "model (all train seasons)": p_gb, **p_rep}, split=TE2).round(3).to_string())
for st_lab, code in [("listed Out/Doubtful", [3, 4]), ("Questionable", [2]), ("not listed", [0])]:
    mm = X["status"].isin(code).to_numpy() & tg["season"].isin(TE2).to_numpy()
    print(f"   {st_lab}: n {mm.sum()}, actually played {y[mm].mean():.2f} | game 1 {y[mm & (X['k'] == 1).to_numpy()].mean():.2f}, games 2-3 {y[mm & X['k'].between(2, 3).to_numpy()].mean():.2f}, games 4+ {y[mm & (X['k'] >= 4).to_numpy()].mean():.2f}")

# ======================================================================= WEEK: best level x best availability
lev_best = predict_by_k(feats_b, co_b)
print("\nWEEK (expected points over the next 7 days):")
wt = H.week_table({"v1 + flat (live today)": v1, "best level + flat": lev_best, "best level + model": lev_best, "best level + model with status": lev_best},
                  {"v1 + flat (live today)": H.flat_avail(), "best level + flat": H.flat_avail(), "best level + model": p_gb, "best level + model with status": p_rep["model, report years, with status"]}, split=TE2)
print(wt.round(2).to_string())
OUT["week"] = wt.round(3).reset_index().to_dict("records")
json.dump(OUT, open(D / "st_batch1.json", "w"), indent=1, default=str)
