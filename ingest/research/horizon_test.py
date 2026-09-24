"""Is our 6-season asset horizon the reason young prospects read low?

Dynasty valuation tools value a player over ~10 seasons (RotoWire/others: 'a decade-plus of growth'; draftvalueanalytics:
10-year discounted sum). Ours stops at season 6, which for a 19-year-old is age 24-25 -- before his prime (26-30).
Extends the asset value to a 10-season horizon (seasons 7-10 continue each player's path with the empirical yearly change
by age and career-survival odds by age, both estimated from real data) and compares rankings at horizon 6 vs 10, under
different yearly discounts, against two independent expert anchors:
  Hashtag Basketball crowdsourced dynasty rankings (2026-09-24, 205k votes)
  RotoWire 2026-27 dynasty top-30 + listed prospects (2026)
"""
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.argv = ["x"]
src = open(ROOT / "asset_value_v2.py", encoding="utf-8").read()
part = src.split("def total_value(k, delta=DELTA):")[0]
part = part.replace("for ty in (2015, 2016, 2017, 2018, 2019):", "for ty in ():").replace(
    'print(pd.DataFrame(rows).groupby(["h", "c"]).mean().round(3).to_string())', "pass")
exec(part)  # fc, live, E_v, E_p, pros, pr_*, cut(), prospect_value, ...

# ---- empirical annual change and survival by age, for seasons beyond the 6-season model
pp = panel[(panel["GP"] >= 20) & (panel["mpg"] >= 10)].copy()
pp["age1"] = pp["AGE"] + 1
nxt = pp[["PLAYER_ID", "yr", "fpg"]].copy(); nxt["yr"] -= 1
pp = pp.merge(nxt.rename(columns={"fpg": "fpg_n"}), on=["PLAYER_ID", "yr"], how="left")
good = pp[pp["fpg"] >= 30]
pres = good.assign(p=good["fpg_n"].notna().astype(float)).groupby(good["age1"].round().clip(20, 40))["p"].mean()
chg = good[good["fpg_n"].notna()].assign(d=lambda d: d["fpg_n"] - d["fpg"]).groupby(good[good["fpg_n"].notna()]["age1"].round().clip(20, 40))["d"].mean()
pres = pres.rolling(3, center=True, min_periods=1).mean()
chg = chg.rolling(3, center=True, min_periods=1).mean()
print("annual change in pts/g for a 30+ pts/g player who stays, and P(still a rotation player next year), by age:")
print({int(a): (round(float(chg.get(a, np.nan)), 1), round(float(pres.get(a, np.nan)), 2)) for a in (23, 25, 27, 29, 31, 33, 35, 37)})
H10 = 10


def extend(E6, ages):  # E6: (n, >=6) path; returns (n, 10) and per-year survival factor for h=7..10
    n = len(E6)
    E = np.zeros((n, H10)); E[:, :6] = E6[:, :6]
    S = np.ones((n, H10))
    for h in range(7, H10 + 1):
        a = np.clip(np.round(ages + h - 1), 20, 40)
        E[:, h - 1] = E[:, h - 2] + np.array([float(chg.get(x, chg.iloc[-1])) for x in a])
        S[:, h - 1] = S[:, h - 2] * np.array([float(pres.get(x - 1, pres.iloc[-1])) for x in a])
    return E, S


Ev10, Sv10 = extend(E_v, live_age_next)
Ep10, Sp10 = extend(E_p, pr_age)


def total_value_h(k, delta, horizon):
    c_future = max(cut(k), REPL)
    v_vet, v_pro = np.zeros(len(live)), np.zeros(len(pros))
    for h in range(1, horizon + 1):
        if not np.isfinite(cut(k)) and h > 1:
            continue
        c = REPL if h == 1 else c_future
        hm = min(h, 6)  # residual spread / presence model of season 6 reused beyond 6
        w = delta ** (h - 1)
        v_vet += w * fc.value(live, live_age_next, hm, c, mu=Ev10[:, h - 1]) * (Sv10[:, h - 1] if h > 6 else 1)
        v_pro += w * prospect_value(pr_pick, pr_age, hm, c, mu=Ep10[:, h - 1]) * (Sp10[:, h - 1] if h > 6 else 1)
    return np.concatenate([v_vet, v_pro])


names = list(live["PLAYER_NAME"]) + [q["player"] for q in pros]
res = pd.DataFrame({"player": names, "age": np.concatenate([live_age_next, pr_age])})


def nn_(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


res["norm"] = res["player"].apply(nn_)
dyn = pd.read_csv(D / "hashtag_dynasty_2026-09-24.csv"); dyn["norm"] = dyn["player"].apply(nn_)
ROTO = {"Victor Wembanyama": 1, "Nikola Jokic": 2, "Shai Gilgeous-Alexander": 3, "Luka Doncic": 4, "Cade Cunningham": 5, "Cooper Flagg": 6, "Jayson Tatum": 7,
        "Jalen Johnson": 8, "Anthony Edwards": 9, "Tyrese Maxey": 10, "Scottie Barnes": 11, "Tyrese Haliburton": 12, "Chet Holmgren": 13, "Josh Giddey": 14,
        "Cameron Boozer": 15, "Amen Thompson": 16, "Evan Mobley": 17, "Jalen Williams": 18, "Giannis Antetokounmpo": 19, "Donovan Mitchell": 20,
        "Karl-Anthony Towns": 21, "Deni Avdija": 22, "Devin Booker": 23, "Trey Murphy": 24, "Brandon Miller": 25, "Jalen Brunson": 26, "Alperen Sengun": 27,
        "Bam Adebayo": 28, "LaMelo Ball": 29, "Trae Young": 30, "Caleb Wilson": 37, "Darryn Peterson": 38, "Dylan Harper": 40, "Kon Knueppel": 41,
        "AJ Dybantsa": 42, "VJ Edgecombe": 46, "Ace Bailey": 97}
rw = pd.DataFrame({"norm": [nn_(k) for k in ROTO], "rw": list(ROTO.values())})
res = res.drop_duplicates("norm").merge(dyn[["norm", "rank"]].rename(columns={"rank": "hash"}), on="norm", how="left").merge(rw, on="norm", how="left")
Ks = (1, 3, 5, 19)
print("\nrank of the top prospects (1 = most valuable asset) under different horizons / yearly discounts  [Hashtag rank | RotoWire rank]")
print(f"{'':16s} " + "  ".join(f"{'H='+str(hz)+' d='+str(d):>12s}" for hz, d in [(6, 0.92), (10, 0.92), (10, 0.96), (10, 1.0)]) + "   | anchors")
vals = {}
for hz, d in [(6, 0.92), (10, 0.92), (10, 0.96), (10, 1.0)]:
    for k in Ks:
        vals[(hz, d, k)] = pd.Series(total_value_h(k, d, hz)[: len(names)], index=names).groupby(level=0).first()
res_idx = res.set_index("player")
for k in (5, 19):
    print(f"\n-- {k} keepers per team --")
    for who in ["Cooper Flagg", "Cameron Boozer", "AJ Dybantsa", "Darryn Peterson", "Caleb Wilson", "Dylan Harper", "Ace Bailey", "Nikola Jokić", "Victor Wembanyama"]:
        if who not in vals[(6, 0.92, k)].index:
            continue
        row = f"{who:16s} "
        for hz, d in [(6, 0.92), (10, 0.92), (10, 0.96), (10, 1.0)]:
            s = vals[(hz, d, k)]
            row += f"{int((s > s[who]).sum() + 1):12d}  "
        rr = res_idx.loc[who] if who in res_idx.index else None
        row += f" | {'' if rr is None or pd.isna(rr.hash) else int(rr.hash)} | {'' if rr is None or pd.isna(rr.rw) else int(rr.rw)}"
        print(row)
print("\nagreement with the expert anchors (Spearman; higher = closer)")
for k in (5, 19):
    for hz, d in [(6, 0.92), (10, 0.92), (10, 0.96), (10, 1.0)]:
        s = vals[(hz, d, k)].rename("v").rename_axis("player").reset_index().drop_duplicates("player").merge(res[["player", "hash", "rw"]], on="player")
        h_ = s.dropna(subset=["hash"]); r_ = s.dropna(subset=["rw"])
        print(f"  K={k:2d} H={hz:2d} d={d}: vs Hashtag crowd {-spearmanr(h_.v, h_.hash)[0]:.3f} (n={len(h_)}) | vs RotoWire {-spearmanr(r_.v, r_.rw)[0]:.3f} (n={len(r_)})")
