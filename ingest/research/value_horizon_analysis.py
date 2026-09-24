"""Diagnostic: does our VOR distinguish redraft value from dynasty value?

Compares our current VOR(K) rankings against two external "anchors":
  redraft anchor: ESPN's live 2026-27 ADP (what the redraft market pays for next-season output)
  dynasty anchor: Hashtag Basketball crowdsourced dynasty rankings (205k votes, 2026-09-24)
then prototypes a horizon-weighted value family V(delta) = sum_t delta^t * E[(v_t - r)+] (with
forecast uncertainty growing by year, so young high-variance players carry option value) and
reports which delta best matches each anchor.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
OPP_FLOOR, OPP_AMP, OPP_TAU = 14.84, 20.37, 19.96
opp = lambda k: OPP_FLOOR + OPP_AMP * np.exp(-k / OPP_TAU)


def normn(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


hub = json.loads((ROOT.parent.parent / "dashboard" / "hub_data.json").read_text(encoding="utf-8"))["players"]
P = pd.DataFrame([{"id": p["id"], "player": p["player"], "kind": p["kind"], "age": p["age"], "traj": p["trajectory"]} for p in hub])
P["pid"] = P["id"].str[1:].astype(int)
P["norm"] = P["player"].apply(normn)
dyn = pd.read_csv(D / "hashtag_dynasty_2026-09-24.csv")
dyn["norm"] = dyn["player"].apply(normn)
P = P.merge(dyn[["norm", "rank"]].rename(columns={"rank": "dyn_rank"}), on="norm", how="left")
adp = pd.read_csv(D / "espn_adp.csv")
adp = adp[(adp["season_id"] == 2027) & adp["adp"].notna() & adp["PLAYER_ID"].notna()].copy()
adp["pid"] = adp["PLAYER_ID"].astype(int)
P = P.merge(adp.sort_values("adp").drop_duplicates("pid")[["pid", "adp"]], on="pid", how="left")
print(f"hub players {len(P)} | matched to dynasty top-200: {P['dyn_rank'].notna().sum()} of {len(dyn)} | with ESPN ADP: {P['adp'].notna().sum()}")
miss = dyn[~dyn["norm"].isin(P["norm"])]["player"].tolist()
print("dynasty names not in hub (mostly 2027 draft/unscored):", miss[:25])


def vor(traj, k):
    o, tot = opp(k), 0.0
    for v in traj:
        if v < o:
            break
        tot += v - o
    return tot


for k in (0, 3, 5, 19):
    P[f"vor{k}"] = P["traj"].apply(lambda t: vor(t, k))
P["y1"] = P["traj"].apply(lambda t: t[0])

# ---- 1. is VOR rank-stable across K? (Tommy's observation)
pos = P[P["vor3"] > 0]
print("\n== VOR rank correlation across keeper counts (players with VOR>0 at K=3, n=%d)" % len(pos))
for a, b in [(0, 3), (3, 5), (3, 19), (0, 19)]:
    print(f"  Spearman VOR@K={a} vs VOR@K={b}: {spearmanr(pos[f'vor{a}'], pos[f'vor{b}'])[0]:.3f}")

# ---- 2. how do VOR(K) rankings relate to the two anchors?
print("\n== Spearman vs anchors (higher = closer)")
dm, am = P[P["dyn_rank"].notna()], P[P["adp"].notna()]
print(f"{'metric':22s} {'vs dynasty (n=%d)' % len(dm):>20s} {'vs ESPN ADP (n=%d)' % len(am):>22s}")
for col in ["y1", "vor0", "vor3", "vor5", "vor19"]:
    print(f"{col:22s} {-spearmanr(dm[col], dm['dyn_rank'])[0]:20.3f} {-spearmanr(am[col], am['adp'])[0]:22.3f}")

# ---- 3. horizon-weighted, uncertainty-aware value family
SD = np.array([6.4, 8.2, 9.4, 10.3, 11.1, 11.8, 12.4])


def hv(traj, delta, r):
    t = np.asarray(traj, dtype=float)[:7]
    s = SD[: len(t)]
    z = (t - r) / s
    ev = (t - r) * norm.cdf(z) + s * norm.pdf(z)  # E[(v - r)+]
    return float((delta ** np.arange(len(t)) * ev).sum())


print("\n== horizon-weighted value V = sum_t delta^t E[(v_t - r)+], r = replacement pts/g")
print("  (delta=0 is pure redraft; delta=1 is undiscounted full dynasty)")
rows = []
for r in (20, 26, 32):
    for delta in (0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
        vals = P["traj"].apply(lambda t: hv(t, delta, r))
        d_c = -spearmanr(vals[dm.index], dm["dyn_rank"])[0]
        a_c = -spearmanr(vals[am.index], am["adp"])[0]
        rows.append((r, delta, d_c, a_c))
        P[f"hv_{r}_{delta}"] = vals
print(f"  {'r':>3s} {'delta':>6s} {'vs dynasty':>11s} {'vs ADP':>8s}")
for r, dl, d_c, a_c in rows:
    print(f"  {r:3d} {dl:6.1f} {d_c:11.3f} {a_c:8.3f}")

best_dyn = max(rows, key=lambda x: x[2])
best_adp = max(rows, key=lambda x: x[3])
print(f"\nbest match to DYNASTY: r={best_dyn[0]}, delta={best_dyn[1]} (rho {best_dyn[2]:.3f});  best match to REDRAFT ADP: r={best_adp[0]}, delta={best_adp[1]} (rho {best_adp[3]:.3f})")

# ---- 4. the Boozer test: rank under each view (among hub players)
for who in ["Cameron Boozer", "Cooper Flagg", "Victor Wembanyama", "Dylan Harper", "AJ Dybantsa", "Jalen Johnson", "Kevin Durant", "Stephen Curry", "LeBron James"]:
    row = P[P["player"] == who]
    if row.empty:
        continue
    i = row.index[0]
    rk = lambda col: int((P[col] > P.loc[i, col]).sum() + 1)
    print(f"  {who:20s} year-1 rank {rk('y1'):3d} | VOR@K=3 rank {rk('vor3'):3d} | VOR@K=19 rank {rk('vor19'):3d} | "
          f"V(delta=0) rank {rk(f'hv_{best_adp[0]}_0'):3d} | V(delta={best_dyn[1]}) rank {rk(f'hv_{best_dyn[0]}_{best_dyn[1]}'):3d} | "
          f"actual: ADP {row['adp'].iloc[0] if pd.notna(row['adp'].iloc[0]) else float('nan'):.0f}, dynasty {row['dyn_rank'].iloc[0] if pd.notna(row['dyn_rank'].iloc[0]) else float('nan'):.0f}")
