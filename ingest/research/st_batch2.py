"""Short-term program, signal batch 2 (RESEARCH_short_term_projection_plan.md): tests 3 (minutes stability, rotation size), 5 (usage change),
9 (per-minute rate change vs minutes change), 10 (shooting luck), 13 (rest / back-to-backs / age, per upcoming game), and the stars lean left by batch 1.
Each signal is added on top of the batch 1 winner, fit on TRAIN, judged on TEST.  Run from ingest/:  .venv/Scripts/python research/st_batch2.py
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
rate = P["delco"] / P["delco_min"].clip(lower=8)
base = H.best_level_feats()
COLS = ["game 1", "games 2-3", "games 4-6", "games 7-9"]

# ---- extra features from the full panel (rotation size uses every player, not only DELCO 5+)
raw = pd.read_parquet(D / "st_panel.parquet", columns=["pid", "season", "gid", "date", "team", "min", "fga", "fg3a", "fta", "info_id"]).sort_values(["date", "gid"])
rot = raw[raw["min"] >= 12].groupby(["team", "gid"]).agg(date=("date", "first"), n=("pid", "size")).reset_index().sort_values("date")
rot["rot_l5"] = rot.groupby("team")["n"].transform(lambda s: s.rolling(5, min_periods=1).mean())
P["rot_l5"] = P.set_index(["team", "gid"]).index.map(rot.set_index(["team", "gid"])["rot_l5"])
for c in ["fga", "fg3a", "fta"]:
    raw[c + "5"] = raw.groupby(["pid", "season"])[c].transform(lambda s: s.rolling(5, min_periods=1).mean())
raw = raw.set_index("info_id")
for c in ["fga5", "fg3a5", "fta5"]:
    P[c] = raw[c].reindex(P.index)

cand = {
    "minutes stability": {"min_sd": P["min_sd10"] - 5.7, "min_gap_x_sd": base["min_gap"] * (P["min_sd10"] - 5.7)},
    "rotation size": {"rot": P["rot_l5"] - 9.5, "min_gap_x_rot": base["min_gap"] * (P["rot_l5"] - 9.5)},
    "usage change": {"use_gap": (P["use_l5"] - P["use_l40"]) * P["min_ew2"]},
    "per-minute rate vs minutes": {"rate_gap": (P["rate_l5"] - rate) * P["delco_min"]},
    "shooting luck": {"fg_luck": (P["fg_l5"] - P["fg_l100"]) * P["fga5"], "fg3_luck": (P["fg3_l5"] - P["fg3_l100"]) * P["fg3a5"],
                      "ft_luck": (P["ft_l5"] - P["ft_l100"]) * P["fta5"]},
    "stars: DELCO slope": {"delco_lvl": P["delco"] - 24.0},
    "stars: career gap": {"career_gap": P["delco_career"] - P["delco"]},
}
preds = {"batch 1 winner": H.predict_by_k(base, H.fit_by_k(base))}
coefs = {}
for name, extra in cand.items():
    f = {**base, **extra}
    co = H.fit_by_k(f)
    coefs[name] = {str(b): dict(zip(["const", *f], [round(float(x), 4) for x in co[b]])) for b in co}
    preds["+ " + name] = H.predict_by_k(f, co)
lt = H.level_table(preds)
show = ["all (g1-6)", "role up (g1-6)", "role down (g1-6)", "stars (DELCO 40+) (g1-6)", "stars (DELCO 40+) lean", "fringe (DELCO < 20) (g1-6)", "rookies lean"]
d = lt[COLS + show].copy()
print("LEVEL (typical miss, test seasons); first row is the batch 1 winner, the rest add one signal each")
print(d.round(3).to_string())
print("\nchange vs batch 1 winner (negative = better):")
print((d[COLS + ["all (g1-6)", "role up (g1-6)", "stars (DELCO 40+) (g1-6)"]] - d.loc["batch 1 winner", COLS + ["all (g1-6)", "role up (g1-6)", "stars (DELCO 40+) (g1-6)"]]).iloc[1:].round(3).to_string())
for name in cand:
    print(f"  {name}: games 2-3 coefficients", {k: v for k, v in coefs[name]["1"].items() if k not in ("fp_gap", "min_gap")})
OUT["level"] = d.round(4).reset_index().to_dict("records")
OUT["coefs"] = coefs

# ---- test 13: rest per upcoming game (game-specific, on top of the best level with the signals that survive)
keep = [n for n in cand if (d.loc["+ " + n, "all (g1-6)"] - d.loc["batch 1 winner", "all (g1-6)"]) <= -0.01]
fK = {**base, **{k: v for n in keep for k, v in cand[n].items()}}
coK = H.fit_by_k(fK)
levK = H.predict_by_k(fK, coK)
print(f"\nkept (gain 0.01+ on games 1-6): {keep}")
tg = T.sort_values(["info_id", "k"]).copy()
tg["gap"] = tg.groupby("info_id")["days_ahead"].diff().fillna(tg["days_ahead"])
tg["b2b"] = (tg["gap"] <= 1).astype(float)
tg["rest3"] = (tg["gap"] >= 3).astype(float)
age = tg["info_id"].map(P["age"])
lvl = tg["info_id"].map(P["delco"])
tg["b2b_old"] = tg["b2b"] * (age >= 32)
tg["b2b_star"] = tg["b2b"] * (lvl >= 40)
tg["pred"] = levK.reindex(list(zip(tg["info_id"], tg["k"]))).to_numpy()
tg["res"] = tg["fp"] - tg["pred"]
F = ["b2b", "rest3", "b2b_old", "b2b_star"]
trm = tg["played"] & tg["season"].isin(H.TRAIN) & tg["res"].notna()
X = np.column_stack([tg.loc[trm, F].to_numpy() * tg.loc[trm, "pred"].to_numpy()[:, None]])
bet = np.linalg.lstsq(X, tg.loc[trm, "res"].to_numpy(), rcond=None)[0]
print("rest effects (share of his level):", dict(zip(F, np.round(bet, 4))))
adj = tg["pred"] * (1 + tg[F].to_numpy() @ bet)
levR = pd.Series(adj.to_numpy(), index=pd.MultiIndex.from_arrays([tg["info_id"], tg["k"]], names=["info_id", "k"]))
lt2 = H.level_table({"batch 1 winner": preds["batch 1 winner"], "kept signals": levK, "kept + rest": levR})
print(lt2[COLS + ["all (g1-6)", "stars (DELCO 40+) (g1-6)", "stars (DELCO 40+) lean"]].round(3).to_string())
OUT["rest"] = dict(zip(F, [float(x) for x in bet]))
OUT["kept"] = keep
# how often each rest type happens, and what the raw data says (test seasons)
te = tg["played"] & tg["season"].isin(H.TEST)
for lab, m in [("back-to-back", tg["b2b"] == 1), ("back-to-back, age 32+", tg["b2b_old"] == 1), ("3+ days rest", tg["rest3"] == 1)]:
    mm = te & m
    print(f"   {lab}: {mm.mean() / te.mean():.0%} of games, scored {tg.loc[mm, 'res'].mean():+.2f} vs prediction before rest terms")
json.dump(OUT, open(D / "st_batch2.json", "w"), indent=1, default=str)
