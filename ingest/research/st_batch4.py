"""Short-term program, signal batch 4 (RESEARCH_short_term_projection_plan.md): tests 11 (hot/cold streaks by stat category), 15 (home/away and travel,
per upcoming game), 16 (rookies and second-year players). Base = everything kept so far (batch 1 winner + DELCO slope + trade, return and season-phase
terms from batch 3).  Run from ingest/:  .venv/Scripts/python research/st_batch4.py
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
rate = P["delco"] / P["delco_min"].clip(lower=8)

# ---------------------------------------------------------------- base: everything kept so far
b1 = H.best_level_feats()
g = P
stint = (g["new_team"] | (g["season"] != g.groupby("pid")["season"].shift(1))).groupby(g["pid"]).cumsum()
fp_stint = g.groupby([g["pid"], stint])["fp"].transform(lambda s: s.expanding().mean())
post = ((g["games_since_trade"] <= 10) & g["traded_in_season"].groupby(g["pid"]).cumsum().gt(0)).astype(float)
gap_prev = g["days_since_prev"].where(g["gp_season"] > 1, 0)
ret_start = gap_prev >= 6
since_ret = g.groupby([g["pid"], ret_start.groupby(g["pid"]).cumsum()]).cumcount()
ret25 = (ret_start.groupby(g["pid"]).cumsum().gt(0) & since_ret.between(1, 4) & ~ret_start).astype(float)
early = (g["season_day"] <= 30).astype(float)
late = (g["season_day"] >= 150).astype(float)
base = {**b1, "delco_lvl": P["delco"] - 24.0,
        "post": post, "post_x_newgap": post * (fp_stint - g["delco"]),
        "ret1": ret_start.astype(float), "ret25": ret25,
        "early": early, "early_x_fp": early * b1["fp_gap"], "early_x_min": early * b1["min_gap"], "late": late, "late_star": late * (g["delco"] >= 35)}


def show(preds, split=H.TEST, extra=()):
    lt = H.level_table(preds, split=split)
    d = lt[COLS + list(extra)]
    first = d.index[0]
    out = d.round(3).astype(str)
    for c in COLS:
        out[c] = d[c].round(3).astype(str) + " (" + (d[c] - d.loc[first, c]).round(3).map(lambda v: f"{v:+.3f}") + ")"
    print(out.to_string())
    return d


raw = pd.read_parquet(D / "st_panel.parquet", columns=["pid", "season", "gid", "date", "team", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "ftm", "fta", "info_id"])
raw = raw.sort_values(["pid", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------- TEST 11: streaks by stat category
print("TEST 11: does a hot streak in one category carry more than in another? (last 5 games vs last 40, in fantasy points)")
cat_fp = {"points": raw["pts"], "rebounds": 1.5 * raw["reb"], "assists": 2 * raw["ast"], "steals+blocks": 3 * (raw["stl"] + raw["blk"]),
          "threes": raw["fg3m"], "free throws": 2 * raw["ftm"] - raw["fta"], "turnovers": -raw["tov"]}
streak = {}
for c, v in cat_fp.items():
    s5 = v.groupby(raw["pid"]).transform(lambda s: s.rolling(5, min_periods=5).mean())
    s40 = v.groupby(raw["pid"]).transform(lambda s: s.rolling(40, min_periods=15).mean())
    streak["hot_" + c] = pd.Series((s5 - s40).to_numpy(), index=raw["info_id"]).reindex(P.index)
fS = {**base, **streak}
coS = H.fit_by_k(fS)
show({"base (all kept so far)": H.predict_by_k(base, H.fit_by_k(base)), "+ streaks by category": H.predict_by_k(fS, coS)})
print("share of a 5-game streak (vs his last 40) that carries into the next games, by category (games 2-3 / 7-9):")
names = ["const", *fS]
for c in cat_fp:
    i = names.index("hot_" + c)
    print(f"   {c:14s} {coS[1][i]:+.2f} / {coS[3][i]:+.2f}")
OUT["test11"] = {c: [float(coS[b][names.index("hot_" + c)]) for b in coS] for c in cat_fp}

# ---------------------------------------------------------------- TEST 16: rookies and second-year players
print("\nTEST 16: rookies and second-year players (first season in the panel from 2015-16 on)")
first = g.groupby("pid")["season"].transform("min")
yr = g["season"].str[:4].astype(int) - first.str[:4].astype(int)
known = first >= "2015-16"
rook = (known & (yr == 0)).astype(float)
soph = (known & (yr == 1)).astype(float)
H.SUB["rookies (true)"] = P.index[rook > 0]
H.SUB["second-year"] = P.index[soph > 0]
H.SUB["rookies, first 20 games"] = P.index[(rook > 0) & (g["gp_season"] <= 20)]
cand16 = {
    "rookie terms": {"rook": rook, "rook_x_fp": rook * b1["fp_gap"], "rook_x_min": rook * b1["min_gap"], "rook_early": rook * (g["gp_season"] <= 20)},
    "rookie + second-year terms": {"rook": rook, "rook_x_fp": rook * b1["fp_gap"], "rook_x_min": rook * b1["min_gap"], "rook_early": rook * (g["gp_season"] <= 20),
                                   "soph": soph, "soph_x_fp": soph * b1["fp_gap"]},
}
pr = {"base (all kept so far)": H.predict_by_k(base, H.fit_by_k(base))}
co16 = {}
for n, ex in cand16.items():
    f = {**base, **ex}
    co16[n] = H.fit_by_k(f)
    pr["+ " + n] = H.predict_by_k(f, co16[n])
lt = H.level_table(pr, subsets=("all", "rookies (true)", "rookies, first 20 games", "second-year"))
print(lt[COLS + ["rookies (true) (g1-6)", "rookies (true) lean", "rookies, first 20 games (g1-6)", "rookies, first 20 games lean", "second-year (g1-6)", "second-year lean"]].round(3).to_string())
nm = ["const", *base, *cand16["rookie + second-year terms"]]
print("games 2-3 terms:", {k: round(float(v), 3) for k, v in zip(nm, co16["rookie + second-year terms"][1]) if k.startswith(("rook", "soph"))})
OUT["test16"] = lt.round(4).reset_index().to_dict("records")

# ---------------------------------------------------------------- TEST 15: home/away and travel, per upcoming game
print("\nTEST 15: home/away and travel for each upcoming game")
gl = D / "game_logs"
hm = []
for f in sorted(gl.glob("regular_season_box_scores_2010_2024_part_*.csv")):
    x = pd.read_csv(f, usecols=["gameId", "matchup", "teamTricode"], dtype={"gameId": str}).drop_duplicates(["gameId", "teamTricode"])
    hm.append(x.rename(columns={"gameId": "gid", "teamTricode": "team"}))
for f in sorted(gl.glob("nba_api_*.csv")):
    x = pd.read_csv(f, usecols=["GAME_ID", "MATCHUP", "TEAM_ABBREVIATION"], dtype={"GAME_ID": str}).drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"])
    hm.append(x.rename(columns={"GAME_ID": "gid", "MATCHUP": "matchup", "TEAM_ABBREVIATION": "team"}))
hm = pd.concat(hm)
hm["gid"] = hm["gid"].astype(int)
hm = hm.drop_duplicates(["gid", "team"])
hm["home"] = hm["matchup"].str.contains("vs.", regex=False).astype(float)
home = hm.set_index(["gid", "team"])["home"]
# k-th team game after each info point
tgm = raw.drop_duplicates(["team", "gid"])[["team", "gid", "date"]].sort_values(["team", "date"]).reset_index(drop=True)
tgm["i"] = tgm.groupby("team").cumcount()
tgm["gid_i"] = tgm["gid"].astype(int)
tgm["home"] = pd.MultiIndex.from_arrays([tgm["gid_i"], tgm["team"]]).map(home)
tgm["prev_home"] = tgm.groupby("team")["home"].shift(1)
key = tgm.set_index(["team", "gid"])["i"]
pos = pd.Series(pd.MultiIndex.from_arrays([P["team"], P["gid"]]).map(key), index=P.index)
tt = T.copy()
tt["team"] = tt["info_id"].map(P["team"])
j = pd.MultiIndex.from_arrays([tt["team"], tt["info_id"].map(pos) + tt["k"]])
gm = tgm.set_index(["team", "i"])
tt["home"] = j.map(gm["home"])
tt["road_trip"] = ((j.map(gm["home"]) == 0) & (j.map(gm["prev_home"]) == 0)).astype(float)      # second+ straight road game
print(f"   home flag found for {tt['home'].notna().mean():.0%} of upcoming games")
levB = pr["base (all kept so far)"]
tt["pred"] = levB.reindex(list(zip(tt["info_id"], tt["k"]))).to_numpy()
tt["res"] = tt["fp"] - tt["pred"]
F = {"home": tt["home"].fillna(0.5) - 0.5, "road trip": tt["road_trip"]}
Xf = np.column_stack([v.to_numpy() for v in F.values()]) * tt["pred"].to_numpy()[:, None]
trm = (tt["played"] & tt["season"].isin(H.TRAIN) & tt["res"].notna()).to_numpy()
bet = np.linalg.lstsq(Xf[trm], tt.loc[trm, "res"].to_numpy(), rcond=None)[0]
print("   effects (share of his level): home vs away", f"{bet[0]:+.4f}", "| second straight road game", f"{bet[1]:+.4f}")
adj = tt["pred"] * (1 + Xf / tt["pred"].to_numpy()[:, None] @ bet)
levH = pd.Series(adj.to_numpy(), index=pd.MultiIndex.from_arrays([tt["info_id"], tt["k"]], names=["info_id", "k"]))
show({"base": levB, "+ home/away, road trip": levH})
te = tt["played"] & tt["season"].isin(H.TEST)
print("   raw lean (test): home", f"{tt.loc[te & (tt['home'] == 1), 'res'].mean():+.2f}", "away", f"{tt.loc[te & (tt['home'] == 0), 'res'].mean():+.2f}")
OUT["test15"] = {k: float(b) for k, b in zip(F, bet)}
json.dump(OUT, open(D / "st_batch4.json", "w"), indent=1, default=str)
