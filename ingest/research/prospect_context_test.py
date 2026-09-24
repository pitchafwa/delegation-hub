"""Does the team a rookie lands on (how much production is walking out vs. returning) predict his output?

For every past draftee (classes 2010-2025) find the team he played for at the START of his rookie season, and measure that
team's situation from the previous season's roster:
   returning_fp  fantasy pts per team-game produced last season by players who are STILL on the team
   vacated_fp    ...by players who LEFT (traded/FA/retired)
   arriving_fp   ...last season's production of veterans who ARRIVED from other teams (new competition)
   open_fp       vacated - arriving  (net production up for grabs)
   star_ret      best returning player's fpg (a true alpha ahead of him)
Then: does adding them improve leave-one-draft-class-out predictions of his rookie-year and later pts/g?
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"

panel = pd.read_csv(D / "breakout_panel_ctx2.csv")
panel["maxgp"] = panel.groupby("yr")["GP"].transform("max")
panel["fpc"] = panel["fpg"] * panel["GP"] / panel["maxgp"]  # fantasy pts contributed per TEAM game last season
panel["stays"] = (panel["team_id"] == panel["team_id_next"])
panel["arrives"] = panel["team_id_next"].notna() & ~panel["stays"]


def team_context(yr_prev):
    p = panel[panel["yr"] == yr_prev]
    rows = {}
    for T in set(p["team_id"].dropna()).union(set(p["team_id_next"].dropna())):
        prior = p[p["team_id"] == T]
        ret = prior[prior["team_id_next"] == T]
        dep = prior[prior["team_id_next"] != T]
        arr = p[(p["team_id_next"] == T) & (p["team_id"] != T)]
        rows[int(T)] = dict(returning_fp=ret["fpc"].sum(), vacated_fp=dep["fpc"].sum(), arriving_fp=arr["fpc"].sum(),
                            star_ret=ret["fpg"].max() if len(ret) else 0.0, n_ret_good=int((ret["fpg"] >= 35).sum()))
    out = pd.DataFrame(rows).T
    out["open_fp"] = out["vacated_fp"] - out["arriving_fp"]
    out["crowd_fp"] = out["returning_fp"] + out["arriving_fp"]  # everything he must beat for minutes
    return out


# rookie -> first team of rookie season
gl = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "TEAM"])
gl["sy"] = gl["SEASON"].str[:4].astype(int)
first = gl.sort_values("GAME_DATE").groupby(["PLAYER_ID", "sy"]).first().reset_index()
abbr_id = panel.dropna(subset=["team_id"]).drop_duplicates(["team", "yr"]).set_index(["team", "yr"])["team_id"].to_dict()

E = pd.read_pickle(D / "_prospect_engine_rows.pkl")   # pid, cls (draft year), pick, dage, talent, k, y, B, lp ...
info = {}
for cls in sorted(E["cls"].unique()):
    if cls - 1 < panel["yr"].min():
        continue
    ctx = team_context(cls - 1)
    f = first[first["sy"] == cls]
    for r in f.itertuples():
        tid = abbr_id.get((r.TEAM, cls))
        if tid is not None and int(tid) in ctx.index:
            info[(r.PLAYER_ID, cls)] = ctx.loc[int(tid)].to_dict()
CT = pd.DataFrame([{**{"pid": k[0], "cls": k[1]}, **v} for k, v in info.items()])
E = E.merge(CT, on=["pid", "cls"], how="inner")
print(f"prospect-seasons with a known team context: {len(E)} ({E['pid'].nunique()} players, classes {E['cls'].min()}-{E['cls'].max()})")
E["is1"] = (E["pick"] == 1).astype(float)
E["open_pct"] = E["vacated_fp"] / (E["vacated_fp"] + E["returning_fp"]).replace(0, np.nan)
print(f"team context spread: returning_fp mean {E['returning_fp'].mean():.0f} (sd {E['returning_fp'].std():.0f}); vacated_fp mean {E['vacated_fp'].mean():.0f} (sd {E['vacated_fp'].std():.0f}); "
      f"crowd_fp mean {E['crowd_fp'].mean():.0f} (sd {E['crowd_fp'].std():.0f})")
BASE = ["lp", "dage", "talent", "is1"]
CTXS = {"none": [], "+returning": ["returning_fp"], "+vacated": ["vacated_fp"], "+crowd (ret+arriving)": ["crowd_fp"],
        "+open (vac-arr)": ["open_fp"], "+star_ret": ["star_ret"], "+crowd+star": ["crowd_fp", "star_ret"], "+ret+vac+arr": ["returning_fp", "vacated_fp", "arriving_fp"]}
rm = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))


def loco(feats, sub):
    out = pd.Series(np.nan, index=sub.index)
    for cls in sub["cls"].unique():
        for k in range(5):
            te = (sub["cls"] == cls) & (sub["k"] == k)
            tr = (sub["cls"] != cls) & (sub["k"] == k)
            if te.sum() == 0 or tr.sum() < 30:
                continue
            X = sub.loc[tr, feats].fillna(sub[feats].median())
            sc = StandardScaler().fit(X)
            m = Ridge(alpha=5).fit(sc.transform(X), sub.loc[tr, "y"])
            out[te] = m.predict(sc.transform(sub.loc[te, feats].fillna(sub[feats].median())))
    return out


for label, sub in [("PICKS 1-15", E[E["pick"] <= 15]), ("ALL PICKS", E)]:
    print(f"\n{label}: held-out RMSE (pts/g) by seasons since draft, and coefficient of the context term (per +1 sd) at k=0")
    print(f"{'context':24s} " + " ".join(f"k={k:<5d}" for k in range(4)) + " | coef k=0")
    for name, cf in CTXS.items():
        pr = loco(BASE + cf, sub)
        line = f"{name:24s} " + " ".join(f"{rm(sub[sub.k==k]['y'], pr[sub.k==k]):6.2f} " for k in range(4))
        if cf:
            t = sub[sub.k == 0]
            X = t[BASE + cf].fillna(t[BASE + cf].median()); sc = StandardScaler().fit(X)
            m = Ridge(alpha=5).fit(sc.transform(X), t["y"])
            line += " | " + ", ".join(f"{c}:{v:+.2f}" for c, v in zip(cf, m.coef_[len(BASE):]))
        print(line)


# ---------------------------------------------------------------- position-specific competition
bio = pd.read_csv(D / "player_bio.csv")[["PERSON_ID", "POSITION"]].rename(columns={"PERSON_ID": "PLAYER_ID"})
def grp(pos):
    if not isinstance(pos, str) or not pos: return "F"
    return {"G": "G", "F": "F", "C": "C"}.get(pos.strip()[0].upper(), "F")
bio["pg"] = bio["POSITION"].apply(grp)
pg = bio.set_index("PLAYER_ID")["pg"]
panel["pg"] = panel["PLAYER_ID"].map(pg).fillna("F")
# 3 groups; a big is 'C' or 'F' listed with C? keep simple: front-court = F or C
panel["front"] = panel["pg"].isin(["F", "C"])


def pos_ctx(yr_prev, is_front):
    p = panel[(panel["yr"] == yr_prev) & (panel["front"] == is_front)]
    rows = {}
    for T in set(p["team_id"].dropna()).union(set(p["team_id_next"].dropna())):
        prior = p[p["team_id"] == T]
        ret = prior[prior["team_id_next"] == T]
        dep = prior[prior["team_id_next"] != T]
        arr = p[(p["team_id_next"] == T) & (p["team_id"] != T)]
        rows[int(T)] = dict(pos_ret=ret["fpc"].sum(), pos_vac=dep["fpc"].sum(), pos_arr=arr["fpc"].sum())
    o = pd.DataFrame(rows).T
    o["pos_open"] = o["pos_vac"] - o["pos_arr"]
    o["pos_crowd"] = o["pos_ret"] + o["pos_arr"]
    return o


# rookie's own position group from the dataset (guard vs front-court)
u = pd.read_csv(D / "rookie_model_dataset_unified.csv")[["PLAYER_ID", "pos"]].drop_duplicates("PLAYER_ID")
u["front"] = ~u["pos"].astype(str).str.upper().str.contains("G") | u["pos"].astype(str).str.upper().str.contains("F|C")
tid_by = {}
rows = []
for cls in sorted(E["cls"].unique()):
    for is_front in (True, False):
        pc = pos_ctx(cls - 1, is_front)
        for pid_ in E[E["cls"] == cls]["pid"].unique():
            fr = bool(u.loc[u["PLAYER_ID"] == pid_, "front"].iloc[0]) if (u["PLAYER_ID"] == pid_).any() else True
            if fr != is_front:
                continue
            f = first[(first["PLAYER_ID"] == pid_) & (first["sy"] == cls)]
            if f.empty: continue
            tid = abbr_id.get((f["TEAM"].iloc[0], cls))
            if tid is not None and int(tid) in pc.index:
                rows.append({"pid": pid_, "cls": cls, **pc.loc[int(tid)].to_dict()})
PC = pd.DataFrame(rows).drop_duplicates(["pid", "cls"])
E2 = E.merge(PC, on=["pid", "cls"], how="inner")
print(f"\nwith position-specific context: {len(E2)} prospect-seasons")
CT2 = {"team open_fp": ["open_fp"], "same-position open (vac-arr)": ["pos_open"], "same-position crowd (ret+arr)": ["pos_crowd"], "both open + pos_open": ["open_fp", "pos_open"]}
for label, sub in [("PICKS 1-15", E2[E2["pick"] <= 15]), ("ALL PICKS", E2)]:
    print(f"\n{label}: held-out RMSE by seasons since draft")
    print(f"{'context':32s} " + " ".join(f"k={k:<5d}" for k in range(4)))
    pr0 = loco(BASE, sub)
    print(f"{'none':32s} " + " ".join(f"{rm(sub[sub.k==k]['y'], pr0[sub.k==k]):6.2f} " for k in range(4)))
    for name, cf in CT2.items():
        pr = loco(BASE + cf, sub)
        print(f"{name:32s} " + " ".join(f"{rm(sub[sub.k==k]['y'], pr[sub.k==k]):6.2f} " for k in range(4)))
