"""Keeper-count-aware ASSET VALUE from historical analogs.

For a player today, find similar past players (same age, similar production/minutes/pedigree) and
look at what actually happened to them 1-6 seasons later -- including the ones who faded out or
retired (counted as zero value). This bakes in real ceilings, age decline, and survival without
Gaussian assumptions or Kalman aging shapes.

  V(K) = sum_{t=1..6} delta^(t-1) * mean_over_analogs[ (fpg_{t} - c_t(K))+ ]
    c_1      = replacement level for next season (you always play year 1)
    c_t(K)   t>=2: the KEEP cutoff = pts/g of the (K x 12 teams)-th best player (K=0 -> no future value;
             keepers cost nothing in this league, only a slot)
Incoming prospects use analogs = past draftees at similar draft slots (k seasons after being drafted).

Out-of-sample test: analogs restricted to outcomes already observable at the test date.
Usage: python asset_value_analog.py  -> data/asset_value.csv (+ printed diagnostics)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
NTEAMS, H = 12, 6
REPL = 22.0
LAST_YR = 2025
KNN = 60

panel = pd.read_csv(D / "breakout_panel_ctx2.csv")
panel["logpick"] = np.log(panel["draft_pick"].clip(1, 61))
# outcome at yr+h: fpg if he was a real rotation-ish player that season, else 0 value (faded/retired/absent)
key = panel.set_index(["PLAYER_ID", "yr"])
fp = panel.set_index(["PLAYER_ID", "yr"])["fpg"]
gp = panel.set_index(["PLAYER_ID", "yr"])["GP"]
mp = panel.set_index(["PLAYER_ID", "yr"])["mpg"]
for h in range(1, H + 1):
    idx = pd.MultiIndex.from_arrays([panel["PLAYER_ID"], panel["yr"] + h])
    f = fp.reindex(idx).to_numpy(); g = gp.reindex(idx).to_numpy(); m = mp.reindex(idx).to_numpy()
    ok = (g >= 10) & (m >= 8)
    y = np.where(ok, f, 0.0)  # absent / barely played -> 0 fantasy value
    y = np.where(panel["yr"].to_numpy() + h > LAST_YR, np.nan, y)  # future not yet observed
    panel[f"y{h}"] = y

POOL = panel[(panel["GP"] >= 20) & (panel["mpg"] >= 10)].copy()
FEATS = ["AGE", "fpg", "mpg", "d_fpg", "logpick"]
W = np.array([3.0, 1.5, 0.7, 0.4, 0.6])  # age matters most, then production; pedigree helps youngsters


def matrix(df):
    X = df[FEATS].copy()
    X["d_fpg"] = X["d_fpg"].fillna(0)
    return X.to_numpy(dtype=float)


mu, sd = np.nanmean(matrix(POOL), axis=0), np.nanstd(matrix(POOL), axis=0)


def scaled(df):
    return (np.nan_to_num(matrix(df), nan=0.0) - mu) / sd * W


def analog_paths(query, train, need_by=None):
    """for each query row -> array (n, H, KNN) of analog outcomes y_h (nan where unobserved)."""
    from sklearn.neighbors import NearestNeighbors
    Xt, Xq = scaled(train), scaled(query)
    out = np.full((len(query), H, KNN), np.nan)
    nn = NearestNeighbors(n_neighbors=min(KNN * 3, len(train))).fit(Xt)
    dist, ind = nn.kneighbors(Xq)
    Y = train[[f"y{h}" for h in range(1, H + 1)]].to_numpy()
    for i in range(len(query)):
        for h in range(H):
            vals = Y[ind[i], h]
            vals = vals[~np.isnan(vals)][:KNN]  # nearest analogs whose outcome at +h is observable
            out[i, h, : len(vals)] = vals
    return out


def exp_surplus(paths, c):
    """mean over analogs of (y - c)+ at each horizon; c is a scalar or per-horizon array"""
    c = np.broadcast_to(np.asarray(c, dtype=float), (paths.shape[1],))
    v = np.nanmean(np.maximum(paths - c[None, :, None], 0.0), axis=2)
    return np.where(np.isfinite(c)[None, :], v, 0.0)


# ------------------------------------------------------------------ out-of-sample validation
print("== does the analog forecast beat 'current output persists'? (held-out base years; analogs use only outcomes observable then)")
rows = []
for ty in (2015, 2016, 2017, 2018, 2019):
    test = POOL[POOL["yr"] == ty]
    train = POOL[POOL["yr"] + 6 <= ty]  # fully observable by ty (use h<=... only where yr+h<=ty below)
    train = POOL[POOL["yr"] < ty].copy()
    for h in range(1, H + 1):  # mask outcomes not yet observable at ty
        train[f"y{h}"] = np.where(train["yr"] + h > ty, np.nan, train[f"y{h}"])
    paths = analog_paths(test, train)
    for h in (1, 2, 3, 4):
        if ty + h > LAST_YR:
            continue
        real = test[f"y{h}"].to_numpy()
        for c in (22.0, 35.0, 45.0):
            pa = np.nanmean(np.maximum(paths[:, h - 1, :] - c, 0), axis=1)
            pn = np.maximum(test["fpg"].to_numpy() - c, 0)
            rr = np.maximum(real - c, 0)
            ok = ~np.isnan(real)
            rows.append(dict(ty=ty, h=h, c=c, rho_analog=spearmanr(pa[ok], rr[ok])[0], rho_naive=spearmanr(pn[ok], rr[ok])[0],
                             rmse_analog=float(np.sqrt(np.mean((pa[ok] - rr[ok]) ** 2))), rmse_naive=float(np.sqrt(np.mean((pn[ok] - rr[ok]) ** 2)))))
v = pd.DataFrame(rows).groupby(["h", "c"]).mean(numeric_only=True).drop(columns="ty")
print("   (value = (pts/g - c)+ realized; averaged over test years 2015-2019)")
print(v.round(3).to_string())

# ------------------------------------------------------------------ live: current players
live_all = POOL[POOL["yr"] == LAST_YR].copy()
train_all = POOL[POOL["yr"] < LAST_YR]
live_paths = analog_paths(live_all, train_all)
live_all = live_all.reset_index(drop=True)
mean_y1 = np.nanmean(live_paths[:, 0, :], axis=1)

# ------------------------------------------------------------------ live: incoming prospects (analogs = past draftees by slot)
u = pd.read_csv(D / "rookie_model_dataset_unified.csv")[["PLAYER_ID", "player", "real_draft_year", "real_draft_number", "draft_age"]]
u = u.drop_duplicates("PLAYER_ID")
sbase = pd.read_csv(D / "player_season_base.csv")
sbase["yr"] = sbase["SEASON"].str[:4].astype(int)
sbase = sbase.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sbase["fpg"] = (sbase["PTS"] + 1.5 * sbase["REB"] + 2 * sbase["AST"] + 3 * sbase["STL"] + 3 * sbase["BLK"] + sbase["FG3M"]
                + 2 * sbase["FTM"] - sbase["FTA"] - sbase["TOV"] + 3 * sbase["TD3"]) / sbase["GP"].replace(0, np.nan)
sf = sbase.set_index(["PLAYER_ID", "yr"])
hist_d = u[(u["real_draft_year"] >= 2010) & (u["real_draft_year"] <= LAST_YR)].copy()
hist_d["pick"] = hist_d["real_draft_number"].fillna(61).clip(1, 61)
for k in range(H):  # k = seasons after draft year (0 = rookie season); table stored as y{k+1} to reuse code
    idx = pd.MultiIndex.from_arrays([hist_d["PLAYER_ID"], hist_d["real_draft_year"].astype(int) + k])
    f = sf["fpg"].reindex(idx).to_numpy(); g = sf["GP"].reindex(idx).to_numpy(); m = (sf["MIN"] / sf["GP"]).reindex(idx).to_numpy()
    y = np.where((g >= 10) & (m >= 8), f, 0.0)
    hist_d[f"y{k+1}"] = np.where(hist_d["real_draft_year"].to_numpy() + k > LAST_YR, np.nan, y)


def prospect_paths(picks, ages):
    out = np.full((len(picks), H, KNN), np.nan)
    hp = np.log(hist_d["pick"].to_numpy())
    ha = hist_d["draft_age"].fillna(hist_d["draft_age"].median()).to_numpy()
    Y = hist_d[[f"y{h}" for h in range(1, H + 1)]].to_numpy()
    for i, (pk, ag) in enumerate(zip(picks, ages)):
        d = np.abs(hp - np.log(max(pk, 1))) * 3.0 + np.abs(ha - ag) * 0.5
        order = np.argsort(d)
        for h in range(H):
            vals = Y[order, h]
            vals = vals[~np.isnan(vals)][:KNN // 2]
            out[i, h, : len(vals)] = vals
    return out


hub = json.loads((ROOT.parent.parent / "dashboard" / "hub_data.json").read_text(encoding="utf-8"))["players"]
pros = [p for p in hub if p["kind"] == "prospect"]
pr_pick = np.array([p["pick"] if p.get("pick") else 61 for p in pros], dtype=float)
pr_age = np.array([p["age"] if p.get("age") else 20.5 for p in pros], dtype=float)
pr_paths = prospect_paths(pr_pick, pr_age)

# keep cutoffs from the combined expected next-season distribution
y1 = np.concatenate([mean_y1, np.nanmean(pr_paths[:, 0, :], axis=1)])
y1s = np.sort(y1)[::-1]
cut = lambda k: np.inf if k <= 0 else float(y1s[min(int(k * NTEAMS) - 1, len(y1s) - 1)])


def asset(paths, k, delta):
    c = np.array([REPL] + [max(cut(k), REPL)] * (H - 1))
    s = exp_surplus(paths, c)  # (n, H)
    return (s * (delta ** np.arange(H))[None, :]).sum(axis=1)


names_live = live_all["PLAYER_NAME"].tolist()
allnames = names_live + [p["player"] for p in pros]
allpaths = np.concatenate([live_paths, pr_paths], axis=0)
allage = np.concatenate([live_all["AGE"].to_numpy() + 1, pr_age])
res = pd.DataFrame({"player": allnames, "age": allage, "kind": ["current"] * len(live_all) + ["prospect"] * len(pros),
                    "PLAYER_ID": list(live_all["PLAYER_ID"]) + [int(p["id"][1:]) for p in pros]})
DELTA = 0.92
for k in (0, 1, 3, 5, 8, 19):
    res[f"av{k}"] = asset(allpaths, k, DELTA)
res["exp_y1"] = y1
res["exp_y3"] = np.nanmean(allpaths[:, 2, :], axis=1)
res["exp_y5"] = np.nanmean(allpaths[:, 4, :], axis=1)
res.to_csv(D / "asset_value.csv", index=False)

# ------------------------------------------------------------------ compare to the two anchors
import re, unicodedata


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
print(f"\n== agreement with anchors (Spearman), delta={DELTA}; dynasty n={len(dm)}, ADP n={len(am)}")
for k in (0, 1, 3, 5, 8, 19):
    print(f"  K={k:2d} (keep cutoff {cut(k):5.1f} pts/g): vs dynasty {-spearmanr(dm[f'av{k}'], dm['dyn'])[0]:.3f} | vs ESPN ADP {-spearmanr(am[f'av{k}'], am['adp'])[0]:.3f}")
young, old = dm[dm["age"] <= 22], dm[dm["age"] >= 31]
for k in (0, 5, 19):
    print(f"  K={k:2d}: within young(<=22, n={len(young)}) vs dynasty {-spearmanr(young[f'av{k}'], young['dyn'])[0]:.3f} | within old(>=31, n={len(old)}) {-spearmanr(old[f'av{k}'], old['dyn'])[0]:.3f}")
Ks = (0, 1, 3, 5, 8, 19)
print(f"\n  {'':20s}" + "".join(f"K={k:<4d}" for k in Ks) + "  | real: ADP  dynasty")
for w in ["Cameron Boozer", "Cooper Flagg", "Dylan Harper", "AJ Dybantsa", "Jalen Johnson", "Victor Wembanyama", "Kevin Durant", "Stephen Curry", "LeBron James"]:
    r = res[res["player"] == w]
    if r.empty:
        continue
    ranks = [int((res[f"av{k}"] > r[f"av{k}"].iloc[0]).sum() + 1) for k in Ks]
    print(f"  {w:20s}" + "".join(f"{x:<6d}" for x in ranks) + f"  | {r['adp'].iloc[0] if pd.notna(r['adp'].iloc[0]) else float('nan'):6.0f} {r['dyn'].iloc[0] if pd.notna(r['dyn'].iloc[0]) else float('nan'):7.0f}")
print("\n== top 20 by asset value at K=0 / K=5 / K=19")
tops = {k: res.sort_values(f"av{k}", ascending=False).head(20)["player"].tolist() for k in (0, 5, 19)}
for i in range(20):
    print(f"  {i+1:2d}  " + " | ".join(f"{tops[k][i]:24s}" for k in (0, 5, 19)))
