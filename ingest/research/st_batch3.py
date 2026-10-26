"""Short-term program, signal batch 3 (RESEARCH_short_term_projection_plan.md): tests 4 (games cut short: injury exits, blowouts, foul trouble ->
partial weight in the recent windows), 8 (trades), 12 + 14 (opponent strength, blowout risk per upcoming game), 17 (returns from an absence),
18 (season phase). Base = batch 1 winner + the DELCO slope from batch 2.  Run from ingest/:  .venv/Scripts/python research/st_batch3.py
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
TR2, TE2 = ["2021-22", "2022-23"], ["2023-24", "2024-25", "2025-26"]
rate = P["delco"] / P["delco_min"].clip(lower=8)
slope = {"delco_lvl": P["delco"] - 24.0}


def feats_from(fp_ew, min_ew):
    return {"fp_gap": fp_ew.reindex(P.index) - P["delco"], "min_gap": (min_ew.reindex(P.index) - P["delco_min"]) * rate, **slope}


def show(preds, split=H.TEST, extra=()):
    lt = H.level_table(preds, split=split)
    d = lt[COLS + list(extra)]
    first = d.index[0]
    out = d.round(3).astype(str)
    for c in COLS:
        out[c] = d[c].round(3).astype(str) + " (" + (d[c] - d.loc[first, c]).round(3).map(lambda v: f"{v:+.3f}") + ")"
    print(out.to_string())
    return d


# ---------------------------------------------------------------- full panel in game order, with game context
raw = pd.read_parquet(D / "st_panel.parquet", columns=["pid", "season", "gid", "date", "team", "min", "fp", "pts", "pf", "started", "min_l10", "info_id"])
raw = raw.sort_values(["pid", "date"]).reset_index(drop=True)
tp = raw.groupby(["gid", "team"])["pts"].sum().rename("tpts").reset_index()
opp = tp.merge(tp, on="gid", suffixes=("", "_o"))
opp = opp[opp["team"] != opp["team_o"]]
margin = opp.set_index(["gid", "team"])["tpts"] - opp.set_index(["gid", "team"])["tpts_o"]
raw["margin"] = pd.MultiIndex.from_arrays([raw["gid"], raw["team"]]).map(margin)
prev_min10 = raw.groupby("pid")["min_l10"].shift(1)
short = (raw["min"] < 0.6 * prev_min10) & (prev_min10 >= 20)
inj = pd.read_parquet(D / "st_inj_features.parquet")["status_now"]
listed = raw["info_id"].map(inj).isin(["Out", "Doubtful", "Questionable"])
cls = pd.Series("normal", index=raw.index)
cls[short] = "other short"
cls[short & (raw["pf"] >= 5)] = "foul trouble"
cls[short & (raw["margin"].abs() >= 20)] = "blowout"
cls[short & listed] = "injury exit (listed next report)"
raw["cls"] = cls
print("games cut short (prev 10-game minutes 20+, played under 60% of it):")
print(raw.loc[short, "cls"].value_counts().to_string(), "\n")


def wewm(weights, hl):
    """EWMA of fp and min per player (same as the panel's pandas ewm, adjust=True, min 3 games) with a weight per game"""
    a = 0.5 ** (1.0 / hl)
    pid, x1, x2, w = raw["pid"].to_numpy(), raw["fp"].to_numpy(), raw["min"].to_numpy(), weights
    o1, o2 = np.full(len(raw), np.nan), np.full(len(raw), np.nan)
    n1 = n2 = dn = 0.0
    cnt, last = 0, None
    for i in range(len(raw)):
        if pid[i] != last:
            n1 = n2 = dn = 0.0; cnt = 0; last = pid[i]
        n1 = a * n1 + w[i] * x1[i]; n2 = a * n2 + w[i] * x2[i]; dn = a * dn + w[i]; cnt += 1
        if cnt >= 3 and dn > 0:
            o1[i], o2[i] = n1 / dn, n2 / dn
    return o1, o2


def windows(wmap):
    w = raw["cls"].map(lambda c: wmap.get(c, 1.0)).to_numpy(dtype=float)
    fp4, _ = wewm(w, 4)
    _, mn2 = wewm(w, 2)
    idx = raw["info_id"]
    return pd.Series(fp4, index=idx), pd.Series(mn2, index=idx)


# ---------------------------------------------------------------- TEST 4: weight of games cut short
print("TEST 4: how much a game cut short should count in the recent windows")
fp0, mn0 = windows({})
base = feats_from(fp0, mn0)
res4 = {}
for c, split, tr in [("blowout", H.TEST, H.TRAIN), ("foul trouble", H.TEST, H.TRAIN), ("other short", H.TEST, H.TRAIN), ("injury exit (listed next report)", TE2, TR2)]:
    preds = {}
    for w in (1.0, 0.5, 0.25, 0.0):
        f = feats_from(*windows({c: w})) if w != 1.0 else base
        preds[f"{c}: weight {w:g}"] = H.predict_by_k(f, H.fit_by_k(f, tr))
    print(f"\n{c} (judged {split[0]}..{split[-1]}):")
    d = show(preds, split=split)
    # the players it matters for: info points within 5 games after such a game
    res4[c] = d.round(4).reset_index().to_dict("records")
OUT["test4"] = res4
# effect on the affected players only: info points whose last 5 games include a cut-short game of that class
for c in ["blowout", "injury exit (listed next report)", "other short"]:
    flag = raw.assign(f=(raw["cls"] == c).astype(float)).groupby("pid")["f"].transform(lambda s: s.rolling(5, min_periods=1).max())
    aff = pd.Index(raw.loc[flag > 0, "info_id"]).intersection(P.index)
    H.SUB["after " + c] = aff
    split, tr = (TE2, TR2) if "injury" in c else (H.TEST, H.TRAIN)
    pr = {f"weight {w:g}": H.predict_by_k(f, H.fit_by_k(f, tr)) for w, f in [(1.0, base), (0.5, feats_from(*windows({c: 0.5}))), (0.0, feats_from(*windows({c: 0.0})))]}
    lt = H.level_table(pr, split=split, subsets=("after " + c,))
    print(f"   players within 5 games of a '{c}' game ({len(aff)} info points): typical miss / lean,",
          " | ".join(f"{k} {lt.loc[k, 'after ' + c + ' (g1-6)']:.3f} / {lt.loc[k, 'after ' + c + ' lean']:+.2f}" for k in lt.index))

# ---------------------------------------------------------------- per-info-point signals: trades, returns, season phase
print("\nTESTS 8, 17, 18 (added one at a time on top of the base)")
g = P
stint = (g["new_team"] | (g["season"] != g.groupby("pid")["season"].shift(1))).groupby(g["pid"]).cumsum()
fp_stint = g.groupby([g["pid"], stint])["fp"].transform(lambda s: s.expanding().mean())
post = ((g["games_since_trade"] <= 10) & g["traded_in_season"].groupby(g["pid"]).cumsum().gt(0)).astype(float)
gap_prev = g["days_since_prev"].where(g["gp_season"] > 1, 0)
ret_start = (gap_prev >= 6)
since_ret = g.groupby([g["pid"], ret_start.groupby(g["pid"]).cumsum()]).cumcount()
has_ret = ret_start.groupby(g["pid"]).cumsum().gt(0)
ret1 = (ret_start).astype(float)
ret25 = (has_ret & since_ret.between(1, 4) & ~ret_start).astype(float)
early = (g["season_day"] <= 30).astype(float)
late = (g["season_day"] >= 150).astype(float)
cand = {
    "trade: first 10 games with new team": {"post": post, "post_x_newgap": post * (fp_stint - g["delco"]), "post_x_min": post * base["min_gap"]},
    "back from 5+ days off": {"ret1": ret1, "ret1_x_min": ret1 * base["min_gap"], "ret25": ret25, "ret25_x_min": ret25 * base["min_gap"]},
    "season phase": {"early": early, "early_x_fp": early * base["fp_gap"], "early_x_min": early * base["min_gap"], "late": late,
                     "late_x_min": late * base["min_gap"], "late_star": late * (g["delco"] >= 35)},
}
preds = {"base (batch 1 + slope)": H.predict_by_k(base, H.fit_by_k(base))}
coefs = {}
for n, extra in cand.items():
    f = {**base, **extra}
    co = H.fit_by_k(f)
    coefs[n] = {str(b): dict(zip(["const", *f], [round(float(x), 3) for x in co[b]])) for b in co}
    preds["+ " + n] = H.predict_by_k(f, co)
H.SUB["back 5+ days, next 5 games"] = P.index[(ret1 + ret25) > 0]
H.SUB["after trade (10 games)"] = P.index[post > 0]
lt = H.level_table(preds, subsets=("all", "after trade (10 games)", "back 5+ days, next 5 games", "first month", "stars (DELCO 40+)"))
c2 = ["after trade (10 games) (g1-6)", "after trade (10 games) lean", "back 5+ days, next 5 games (g1-6)", "back 5+ days, next 5 games lean", "first month (g1-6)", "first month lean"]
print(lt[COLS + c2].round(3).to_string())
for n in cand:
    print(f"  {n}: games 2-3", {k: v for k, v in coefs[n]["1"].items() if k not in ("fp_gap", "min_gap", "delco_lvl")})
OUT["tests_8_17_18"] = {"table": lt.round(4).reset_index().to_dict("records"), "coefs": coefs}

# ---------------------------------------------------------------- TESTS 12 + 14: opponent and blowout risk, per upcoming game
print("\nTESTS 12 + 14: opponent strength and blowout risk for each upcoming game")
tgm = raw.drop_duplicates(["team", "gid"])[["team", "gid", "date", "season"]].sort_values(["team", "date"]).reset_index(drop=True)
tgm["i"] = tgm.groupby("team").cumcount()
fpa = raw.groupby(["gid", "team"])["fp"].sum().rename("tfp").reset_index().merge(tp, on=["gid", "team"])
fpa = fpa.merge(fpa[["gid", "team", "tfp", "tpts"]].rename(columns={"team": "opp", "tfp": "ofp", "tpts": "opts"}), on="gid")
fpa = fpa[fpa["team"] != fpa["opp"]]
tgm = tgm.merge(fpa[["gid", "team", "opp", "ofp", "tpts", "opts"]], on=["gid", "team"], how="left")
# strength known before each game: rolling 15 previous games, same season
grp = tgm.groupby(["team", "season"])
tgm["fpa_r"] = grp["ofp"].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())       # fantasy points his team allows (opponents' total)
tgm["net_r"] = grp["tpts"].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean()) - grp["opts"].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())
lg = tgm.groupby("season")["ofp"].mean()
tgm["fpa_rel"] = tgm["fpa_r"] / tgm["season"].map(lg) - 1
key = tgm.set_index(["team", "gid"])
st = key[["fpa_rel", "net_r"]]
# the k-th team game after each info point, its opponent, and the strength of both sides
pos = pd.MultiIndex.from_arrays([P["team"], P["gid"]]).map(key["i"])
tt = T.copy()
tt["team"] = tt["info_id"].map(P["team"])
tt["i"] = tt["info_id"].map(pd.Series(pos, index=P.index)) + tt["k"]
gm = tgm.set_index(["team", "i"])
j = pd.MultiIndex.from_arrays([tt["team"], tt["i"]])
tt["opp"] = j.map(gm["opp"])
tt["gid_k"] = j.map(gm["gid"])
tt["my_net"] = j.map(gm["net_r"])
oj = pd.MultiIndex.from_arrays([tt["opp"], tt["gid_k"]])
tt["opp_fpa"] = oj.map(st["fpa_rel"])
tt["opp_net"] = oj.map(st["net_r"])
tt["gapabs"] = (tt["my_net"] - tt["opp_net"]).abs() / 10.0
starter = (tt["info_id"].map(P["start_l5"]) >= 0.6).astype(float)
fB = {**base, **cand["season phase"]} if False else base
levB = H.predict_by_k(base, H.fit_by_k(base))
tt["pred"] = levB.reindex(list(zip(tt["info_id"], tt["k"]))).to_numpy()
tt["res"] = tt["fp"] - tt["pred"]
F = {"opp allows more": tt["opp_fpa"].fillna(0), "mismatch x starter": tt["gapabs"].fillna(0) * starter, "mismatch x bench": tt["gapabs"].fillna(0) * (1 - starter)}
Xf = np.column_stack([v.to_numpy() for v in F.values()]) * tt["pred"].to_numpy()[:, None]
trm = (tt["played"] & tt["season"].isin(H.TRAIN) & tt["res"].notna() & np.isfinite(Xf).all(1)).to_numpy()
bet = np.linalg.lstsq(Xf[trm], tt.loc[trm, "res"].to_numpy(), rcond=None)[0]
print("effects (share of his level per unit; opponent: +10% points allowed; mismatch: per 10 points of net-rating gap):",
      {k: round(float(b), 4) for k, b in zip(F, bet)})
adj = tt["pred"] * (1 + np.nan_to_num(np.column_stack([v.to_numpy() for v in F.values()])) @ bet)
levO = pd.Series(adj.to_numpy(), index=pd.MultiIndex.from_arrays([tt["info_id"], tt["k"]], names=["info_id", "k"]))
show({"base": levB, "+ opponent and mismatch": levO}, extra=("stars (DELCO 40+) lean", "fringe (DELCO < 20) lean"))
te = tt["played"] & tt["season"].isin(H.TEST)
q = pd.qcut(tt.loc[te, "opp_fpa"], 5, duplicates="drop")
print("   lean by opponent (easiest 20% last):", [f"{v:+.2f}" for v in tt.loc[te].groupby(q, observed=True)["res"].mean()])
OUT["tests_12_14"] = {k: float(b) for k, b in zip(F, bet)}
json.dump(OUT, open(D / "st_batch3.json", "w"), indent=1, default=str)
