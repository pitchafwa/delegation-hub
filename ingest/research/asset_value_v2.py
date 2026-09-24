"""Keeper-count-aware ASSET VALUE, v2: empirical forecast + survival + real outcome spread.

For a player x and horizon h (seasons out):
   A_h(c) = P(still a rotation player | x) * E[ (pts/g_h - c)+ | present ]
 P(present)   logistic on age / production / minutes / pedigree (from real career outcomes, incl. retirements)
 pts/g | present   ridge forecast + the EMPIRICAL residual spread for that age group (keeps the skew, so
                   young high-ceiling players carry real upside instead of a symmetric bell)
   V(K) = sum_{h=1..6} delta^(h-1) * A_h(c_h(K))
   c_1 = replacement level (you always play next season)
   c_h(K), h>=2 = keep cutoff = pts/g of the (K x 12 teams)-th best player; K=0 -> no future value.
   Keepers cost nothing in this league (12 teams), only a roster slot.
Incoming prospects: same structure keyed on draft slot, draft age and pre-NBA talent percentile (small but consistent held-out gain), outcomes = past draftees.

Out-of-sample test vs "current output persists", then compared with the two anchors.
Usage: python asset_value_v2.py -> data/asset_value.csv
"""
import json
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
NTEAMS, H, REPL, LAST_YR, DELTA = 12, 6, 22.0, 2025, 0.92

# ------------------------------------------------------------------ vet panel with outcomes at +1..+6
panel = pd.read_csv(D / "breakout_panel_ctx2.csv")
panel["logpick"] = np.log(panel["draft_pick"].clip(1, 61))
panel["young"] = (panel["AGE"] <= 24) * (24 - panel["AGE"])
fp = panel.set_index(["PLAYER_ID", "yr"])
for h in range(1, H + 1):
    idx = pd.MultiIndex.from_arrays([panel["PLAYER_ID"], panel["yr"] + h])
    f, g, m = (fp[c].reindex(idx).to_numpy() for c in ("fpg", "GP", "mpg"))
    present = ((g >= 10) & (m >= 8)).astype(float)
    obs = panel["yr"].to_numpy() + h <= LAST_YR
    panel[f"pres{h}"] = np.where(obs, present, np.nan)
    panel[f"v{h}"] = np.where(obs & (present == 1), f, np.nan)  # pts/g given present

F = ["fpg", "AGE", "young", "mpg", "late_mpg", "late_fpg", "late_fp_per36", "USG_PCT", "PIE", "logpick", "exp", "d_fpg", "d_mpg"]
POOL = panel[(panel["GP"] >= 20) & (panel["mpg"] >= 10)].copy()
AGEB = lambda a: np.digitize(a, [22.5, 25.5, 29.5])  # 4 age groups


def rg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))


def lg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


class Forecaster:
    """fit on vet rows; predict A_h(c) for any rows. `asof` masks outcomes not yet observable."""

    def __init__(self, train, asof=None, hmax=H):
        self.m, self.s, self.res = {}, {}, {}
        for h in range(1, hmax + 1):
            t = train.copy()
            if asof is not None:
                ok = t["yr"] + h <= asof
                t = t[ok]
            pres = t[t[f"pres{h}"].notna()]
            if len(pres) < 200:
                continue
            self.s[h] = lg().fit(pres[F], pres[f"pres{h}"].astype(int))
            pv = pres[pres[f"pres{h}"] == 1]
            self.m[h] = rg().fit(pv[F], pv[f"v{h}"])
            r = pv[f"v{h}"].to_numpy() - self.m[h].predict(pv[F])
            b = AGEB(pv["AGE"].to_numpy())
            self.res[h] = {g: r[b == g] for g in range(4)}

    def value(self, X, age, h, c):
        """A_h(c) per row; c scalar"""
        if h not in self.m:
            return np.zeros(len(X))
        mu = self.m[h].predict(X[F]); ps = self.s[h].predict_proba(X[F])[:, 1]
        b = AGEB(age)
        out = np.zeros(len(X))
        for g in range(4):
            sel = b == g
            if not sel.any():
                continue
            rr = self.res[h][g] if len(self.res[h][g]) > 30 else np.concatenate(list(self.res[h].values()))
            rr = rr[:: max(1, len(rr) // 300)]
            out[sel] = ps[sel] * np.maximum(mu[sel, None] + rr[None, :] - c, 0).mean(axis=1)
        return out


# ------------------------------------------------------------------ out-of-sample test
print("== analog-free empirical forecast vs 'current output persists'  (test base years 2015-2019, trained only on outcomes observable then)")
rows = []
for ty in (2015, 2016, 2017, 2018, 2019):
    test = POOL[POOL["yr"] == ty]
    fc = Forecaster(POOL[POOL["yr"] < ty], asof=ty, hmax=4)
    for h in (1, 2, 3, 4):
        real = np.where(np.isnan(test[f"pres{h}"]), np.nan, np.where(test[f"pres{h}"] == 1, test[f"v{h}"], 0.0))
        ok = ~np.isnan(real)
        for c in (22.0, 35.0, 45.0):
            pa = fc.value(test, test["AGE"].to_numpy(), h, c)
            pn = np.maximum(test["fpg"].to_numpy() - c, 0)
            rr = np.maximum(real - c, 0)
            rows.append(dict(h=h, c=c, rho_model=spearmanr(pa[ok], rr[ok])[0], rho_naive=spearmanr(pn[ok], rr[ok])[0],
                             rmse_model=float(np.sqrt(np.mean((pa[ok] - rr[ok]) ** 2))), rmse_naive=float(np.sqrt(np.mean((pn[ok] - rr[ok]) ** 2)))))
print(pd.DataFrame(rows).groupby(["h", "c"]).mean().round(3).to_string())

# ------------------------------------------------------------------ live vets
fc = Forecaster(POOL[POOL["yr"] < LAST_YR])
live = POOL[POOL["yr"] == LAST_YR].reset_index(drop=True)
live_age_next = live["AGE"].to_numpy() + 1

# ------------------------------------------------------------------ prospects: draft-slot model
u = pd.read_csv(D / "rookie_model_dataset_unified.csv")[["PLAYER_ID", "player", "real_draft_year", "real_draft_number", "draft_age", "talent_pctile"]].drop_duplicates("PLAYER_ID")
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
             - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
sb["mpg"] = sb["MIN"] / sb["GP"].replace(0, np.nan)
sf = sb.set_index(["PLAYER_ID", "yr"])
hd = u[(u["real_draft_year"] >= 2010)].copy()
hd["logpick"] = np.log(hd["real_draft_number"].fillna(61).clip(1, 61))
hd["dage"] = hd["draft_age"].fillna(hd["draft_age"].median())
hd["talent"] = hd["talent_pctile"].fillna(hd["talent_pctile"].median())
for h in range(1, H + 1):  # h=1 is the rookie season
    idx = pd.MultiIndex.from_arrays([hd["PLAYER_ID"], hd["real_draft_year"].astype(int) + h - 1])
    f, g, m = (sf[c].reindex(idx).to_numpy() for c in ("fpg", "GP", "mpg"))
    present = ((g >= 10) & (m >= 8)).astype(float)
    obs = hd["real_draft_year"].to_numpy() + h - 1 <= LAST_YR
    hd[f"pres{h}"] = np.where(obs, present, np.nan)
    hd[f"v{h}"] = np.where(obs & (present == 1), f, np.nan)
PF = ["logpick", "dage", "talent"]
pm, ps_, pres_res = {}, {}, {}
for h in range(1, H + 1):
    t = hd[hd[f"pres{h}"].notna()]
    ps_[h] = lg().fit(t[PF], t[f"pres{h}"].astype(int))
    pv = t[t[f"pres{h}"] == 1]
    pm[h] = rg().fit(pv[PF], pv[f"v{h}"])
    r = pv[f"v{h}"].to_numpy() - pm[h].predict(pv[PF])
    tier = np.digitize(np.exp(pv["logpick"].to_numpy()), [4, 11, 31])
    pres_res[h] = {g_: r[tier == g_] for g_ in range(4)}


def prospect_value(picks, ages, h, c):
    X = pd.DataFrame({"logpick": np.log(np.clip(picks, 1, 61)), "dage": ages, "talent": pr_talent})
    mu, p = pm[h].predict(X[PF]), ps_[h].predict_proba(X[PF])[:, 1]
    tier = np.digitize(picks, [4, 11, 31])
    out = np.zeros(len(X))
    for g_ in range(4):
        sel = tier == g_
        if sel.any():
            rr = pres_res[h][g_] if len(pres_res[h][g_]) > 20 else np.concatenate(list(pres_res[h].values()))
            out[sel] = p[sel] * np.maximum(mu[sel, None] + rr[None, :] - c, 0).mean(axis=1)
    return out


# bring in the hub's prospects
hub = json.loads((ROOT.parent.parent / "dashboard" / "hub_data.json").read_text(encoding="utf-8"))["players"]
pros = [q for q in hub if q["kind"] == "prospect"]
pr_pick = np.array([q["pick"] if q.get("pick") else 61 for q in pros], dtype=float)
pr_age = np.array([q["age"] if q.get("age") else 20.5 for q in pros], dtype=float)
pr_talent = np.array([q["talent_pctile"] if q.get("talent_pctile") is not None else hd["talent"].median() for q in pros], dtype=float)


def expected_y1(vet, pro):
    """expected next-season pts/g (present-weighted) to set the league keep cutoffs"""
    return None


# expected next-season value per player (for keep cutoffs): E[pts/g_1] incl. absent as 0
def mean_y1_vet():
    mu = fc.m[1].predict(live[F]); p = fc.s[1].predict_proba(live[F])[:, 1]
    return mu * p


def mean_y1_pro():
    X = pd.DataFrame({"logpick": np.log(np.clip(pr_pick, 1, 61)), "dage": pr_age, "talent": pr_talent})
    return pm[1].predict(X[PF]) * ps_[1].predict_proba(X[PF])[:, 1]


y1 = np.concatenate([mean_y1_vet(), mean_y1_pro()])
y1s = np.sort(y1)[::-1]
cut = lambda k: np.inf if k <= 0 else float(y1s[min(int(k * NTEAMS) - 1, len(y1s) - 1)])


def total_value(k, delta=DELTA):
    c_future = max(cut(k), REPL)
    v_vet = np.zeros(len(live)); v_pro = np.zeros(len(pros))
    for h in range(1, H + 1):
        c = REPL if h == 1 else c_future
        if not np.isfinite(cut(k)) and h > 1:
            continue
        wgt = delta ** (h - 1)
        v_vet += wgt * fc.value(live, live_age_next, h, c)
        v_pro += wgt * prospect_value(pr_pick, pr_age, h, c)
    return np.concatenate([v_vet, v_pro])


res = pd.DataFrame({"player": list(live["PLAYER_NAME"]) + [q["player"] for q in pros],
                    "PLAYER_ID": list(live["PLAYER_ID"]) + [int(q["id"][1:]) for q in pros],
                    "age": np.concatenate([live_age_next, pr_age]),
                    "kind": ["current"] * len(live) + ["prospect"] * len(pros)})
Ks = (0, 1, 3, 5, 8, 19)  # shown in the printed diagnostics
for k in range(20):       # all keeper counts the dashboard slider can pick
    res[f"av{k}"] = total_value(k)
res["exp_y1"] = y1
res.to_csv(D / "asset_value.csv", index=False)


def nn_(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


res["norm"] = res["player"].apply(nn_)
dyn = pd.read_csv(D / "hashtag_dynasty_2026-09-24.csv"); dyn["norm"] = dyn["player"].apply(nn_)
adp = pd.read_csv(D / "espn_adp.csv"); adp = adp[(adp["season_id"] == 2027) & adp["adp"].notna() & adp["PLAYER_ID"].notna()]
adp["PLAYER_ID"] = adp["PLAYER_ID"].astype(int)
res = res.drop_duplicates("norm").merge(dyn[["norm", "rank"]].rename(columns={"rank": "dyn"}), on="norm", how="left")
res = res.merge(adp.sort_values("adp").drop_duplicates("PLAYER_ID")[["PLAYER_ID", "adp"]], on="PLAYER_ID", how="left")
dm, am = res[res["dyn"].notna()], res[res["adp"].notna()]
print(f"\n== agreement with anchors (Spearman); dynasty n={len(dm)}, ESPN ADP n={len(am)}   [old approach: VOR@K=19 0.786 / y1 0.853]")
for k in Ks:
    print(f"  K={k:2d} (keep cutoff {cut(k):5.1f} pts/g): vs dynasty {-spearmanr(dm[f'av{k}'], dm['dyn'])[0]:.3f} | vs ESPN ADP {-spearmanr(am[f'av{k}'], am['adp'])[0]:.3f}")
young, old = dm[dm["age"] <= 22], dm[dm["age"] >= 31]
for k in (0, 5, 19):
    print(f"  K={k:2d}: within young(<=22, n={len(young)}) vs dynasty {-spearmanr(young[f'av{k}'], young['dyn'])[0]:.3f} | within old(>=31, n={len(old)}) {-spearmanr(old[f'av{k}'], old['dyn'])[0]:.3f}")
print(f"\n  {'':20s}" + "".join(f"K={k:<4d}" for k in Ks) + "  | real: ADP  dynasty")
for w in ["Cameron Boozer", "Cooper Flagg", "Dylan Harper", "AJ Dybantsa", "Jalen Johnson", "Victor Wembanyama", "Nikola Jokić", "Kevin Durant", "Stephen Curry", "LeBron James"]:
    r = res[res["player"] == w]
    if r.empty:
        continue
    ranks = [int((res[f"av{k}"] > r[f"av{k}"].iloc[0]).sum() + 1) for k in Ks]
    print(f"  {w:20s}" + "".join(f"{x:<6d}" for x in ranks) + f"  | {r['adp'].iloc[0] if pd.notna(r['adp'].iloc[0]) else float('nan'):6.0f} {r['dyn'].iloc[0] if pd.notna(r['dyn'].iloc[0]) else float('nan'):7.0f}")
print("\n== top 20 by asset value at K=0 / K=5 / K=19")
tops = {k: res.sort_values(f"av{k}", ascending=False).head(20)["player"].tolist() for k in (0, 5, 19)}
for i in range(20):
    print(f"  {i+1:2d}  " + " | ".join(f"{tops[k][i]:24s}" for k in (0, 5, 19)))
