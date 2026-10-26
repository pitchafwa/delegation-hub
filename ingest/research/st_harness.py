"""Short-term projection program, Phase 1-2: the test harness and the baselines (RESEARCH_short_term_projection_plan.md).

Every candidate is scored the same way on held-out seasons (TEST: 2022-23 .. 2025-26; candidates may only be fit on TRAIN: 2015-16 .. 2021-22):
  LEVEL         fantasy points per game in upcoming games he PLAYS, by how far ahead (game 1, games 2-3, 4-6, 7-9): typical miss (RMSE) and lean (bias)
  AVAILABILITY  chance he plays each upcoming team game: Brier score and calibration
  WEEK          expected points over the next 7 days (sum over his team's games of chance x level) vs what he actually scored
Subsets: role up / down (minutes over 3 games vs DELCO's minutes, 5+), new teammate absence, recent trade, rookies (first season), returning from 5+ days off,
stars (DELCO 40+), fringe (DELCO under 20), first month of the season.
Usage:  import st_harness as H; H.report({"name": level_series}, avail={"name": p_series})      (series indexed like the panel / targets)
        .venv/Scripts/python research/st_harness.py      -> the baselines
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
TRAIN = ["2015-16", "2016-17", "2017-18", "2018-19", "2020-21", "2021-22"]          # 2019-20 left out (bubble season)
TEST = ["2022-23", "2023-24", "2024-25", "2025-26"]

P = pd.read_parquet(D / "st_panel.parquet")
T = pd.read_parquet(D / "st_targets.parquet")
P = P[P["delco"].notna() & (P["delco"] >= 5)].copy()
T = T[T["info_id"].isin(P["info_id"])].merge(P[["info_id", "season"]], on="info_id")
P = P.set_index("info_id", drop=False)
first_season = P.groupby("pid")["season"].transform("min")
SUB = {
    "all": P.index,
    "role up": P.index[(P["min_l3"] - P["delco_min"]) >= 5],
    "role down": P.index[(P["min_l3"] - P["delco_min"]) <= -5],
    "teammate out (30+)": P.index[P["out_fp"] >= 30],
    "traded, first 10 games": P.index[P["traded_in_season"].groupby(P["pid"]).cumsum().gt(0) & (P["games_since_trade"] <= 10) & (P["games_since_trade"] >= 0)],
    "rookies": P.index[P["season"] == first_season],
    "back from 5+ days off": P.index[P["days_since_prev"] >= 6],
    "stars (DELCO 40+)": P.index[P["delco"] >= 40],
    "fringe (DELCO < 20)": P.index[P["delco"] < 20],
    "first month": P.index[P["season_day"] <= 30],
}
KB = [(1, 1, "game 1"), (2, 3, "games 2-3"), (4, 6, "games 4-6"), (7, 9, "games 7-9")]


def _lev(pred, split):
    t = T[T["season"].isin(split) & T["played"]]
    t = t.assign(pred=t["info_id"].map(pred))
    if "k" in getattr(pred, "index", pd.Index([])).names if isinstance(pred.index, pd.MultiIndex) else False:
        pass
    return t.dropna(subset=["pred"])


def level_table(preds, split=TEST, subsets=("all", "role up", "role down", "teammate out (30+)", "rookies", "back from 5+ days off", "stars (DELCO 40+)", "fringe (DELCO < 20)", "first month")):
    """preds: {name: Series indexed by info_id (same level for every game ahead) or by (info_id, k)}"""
    out = []
    t0 = T[T["season"].isin(split) & T["played"]]
    for name, pr in preds.items():
        if isinstance(pr.index, pd.MultiIndex):
            pv = pd.Series(pr.reindex(list(zip(t0["info_id"], t0["k"]))).to_numpy(), index=t0.index)
        else:
            pv = t0["info_id"].map(pr)
        e = (t0["fp"] - pv)
        row = {"model": name}
        for lo, hi, lab in KB:
            m = (t0["k"] >= lo) & (t0["k"] <= hi) & e.notna()
            row[lab] = np.sqrt(np.mean(e[m] ** 2))
        for s in subsets:
            m = t0["info_id"].isin(SUB[s]) & (t0["k"] <= 6) & e.notna()
            row[f"{s} (g1-6)"] = np.sqrt(np.mean(e[m] ** 2))
            row[f"{s} lean"] = np.mean(e[m])
        out.append(row)
    return pd.DataFrame(out).set_index("model")


def avail_table(preds, split=TEST):
    """preds: {name: Series indexed like T (row index)} of the chance he plays each upcoming game"""
    t0 = T[T["season"].isin(split)]
    out = []
    for name, pr in preds.items():
        p = pr.reindex(t0.index)
        y = t0["played"].astype(float)
        m = p.notna()
        row = {"model": name, "Brier": np.mean((p[m] - y[m]) ** 2), "mean pred": p[m].mean(), "actual": y[m].mean()}
        for lo, hi, lab in KB:
            mk = m & (t0["k"] >= lo) & (t0["k"] <= hi)
            row[f"Brier {lab}"] = np.mean((p[mk] - y[mk]) ** 2)
        out.append(row)
    return pd.DataFrame(out).set_index("model")


def week_table(levels, avails, split=TEST):
    """expected points over the next 7 days: sum over his team's games of chance x level, vs actual points scored in those games"""
    t0 = T[T["season"].isin(split) & (T["days_ahead"] <= 7)]
    act = t0.assign(fp0=t0["fp"].fillna(0.0)).groupby("info_id")["fp0"].sum()
    out = []
    for name, (lv, av) in {k: (levels[k], avails[k]) for k in levels}.items():
        lvv = t0["info_id"].map(lv) if not isinstance(lv.index, pd.MultiIndex) else pd.Series(lv.reindex(list(zip(t0["info_id"], t0["k"]))).to_numpy(), index=t0.index)
        ev = (lvv * av.reindex(t0.index)).groupby(t0["info_id"]).sum(min_count=1)
        j = pd.concat([act, ev], axis=1, keys=["act", "ev"]).dropna()
        out.append({"model": name, "RMSE (7-day pts)": np.sqrt(np.mean((j["act"] - j["ev"]) ** 2)), "lean": np.mean(j["act"] - j["ev"]), "n": len(j)})
    return pd.DataFrame(out).set_index("model")


# ---------------- baselines
def baselines():
    rate = P["delco"] / P["delco_min"].clip(lower=8)
    b = {
        "season DELCO": P["delco"],
        "career DELCO": P["delco_career"],
        "v1 (live since 10-26)": P["delco"] + 1.254 + 0.113 * (P["fp_l3"] - P["delco"]) + 0.363 * (P["min_l3"] - P["delco_min"]) * rate,
        "last 3 avg": P["fp_l3"], "last 5 avg": P["fp_l5"], "last 10 avg": P["fp_l10"], "last 15 avg": P["fp_l15"],
        "weighted recent (half-life 4)": P["fp_ew4"], "weighted recent (half-life 8)": P["fp_ew8"],
        "old site rule (55% DELCO + 45% last 15)": 0.55 * P["delco"] + 0.45 * P["fp_l15"].fillna(P["delco"]),
    }
    return {k: v.dropna() for k, v in b.items()}


def flat_avail():
    lvl = T["info_id"].map(P["delco"])
    first = T["k"] == 1
    p = np.where(lvl < 15, 0.61, np.where(lvl < 20, 0.81, 0.85))
    return pd.Series(p, index=T.index)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    B = baselines()
    print(f"panel {len(P)} info points; targets {len(T)} (test seasons {', '.join(TEST)})\n")
    lt = level_table(B)
    print("LEVEL: typical miss (RMSE, fantasy pts/g) in games he plays, by how far ahead")
    print(lt[[c for _, _, c in KB]].round(2).to_string())
    print("\nLEVEL by situation (games 1-6): typical miss / lean (+ = he scored more than predicted)")
    cols = [c for c in lt.columns if c.endswith("(g1-6)")]
    show = pd.DataFrame({c.replace(" (g1-6)", ""): lt[c].round(2).astype(str) + " / " + lt[c.replace("(g1-6)", "lean")].round(1).map(lambda v: f"{v:+.1f}") for c in cols})
    print(show.to_string())
    fa = flat_avail()
    print("\nAVAILABILITY (chance he plays each upcoming team game): today's flat rule")
    print(avail_table({"flat 85/81/61%": fa}).round(3).to_string())
    print("\nWEEK (expected points over the next 7 days, flat availability):")
    print(week_table({k: B[k] for k in ["season DELCO", "v1 (live since 10-26)", "last 10 avg", "old site rule (55% DELCO + 45% last 15)"]},
                     {k: fa for k in ["season DELCO", "v1 (live since 10-26)", "last 10 avg", "old site rule (55% DELCO + 45% last 15)"]}).round(2).to_string())
