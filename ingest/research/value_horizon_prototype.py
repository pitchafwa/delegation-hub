"""Prototype of a keeper-count-aware ASSET VALUE (separate from keeper-decision VOR).

  V(K) = sum_{t=0..6}  delta^t * S_t * E[(v_t - c_t(K))+]        (t=0 = next season)
    v_t ~ N(traj_t, sd_t * upside(age))     forecast uncertainty grows with horizon; young players
                                             get a wider (option-value / ceiling) distribution
    S_t   career survival: probability he's still a rotation player t seasons out (from data, by age)
    c_0   next-season replacement level (you always play year 1)
    c_t(K) t>=1: the KEEP cutoff = pts/g of the (K x 12 teams)-th best player. K=0 -> nobody can be
           kept -> future years worth nothing (pure redraft); K=19 -> cutoff falls to ~replacement.

Compares fits against the dynasty anchor (Hashtag) and redraft anchor (ESPN ADP), and shows how the
ranking of young vs old players moves with K.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.argv = ["x"]
exec(open(ROOT / "value_horizon_analysis.py", encoding="utf-8").read().split("# ---- 1. is VOR rank-stable")[0])  # reuse P, dm, am, vor(), opp

SD = np.array([6.4, 8.2, 9.4, 10.3, 11.1, 11.8, 12.4])
NTEAMS = 12
dm, am = P[P['dyn_rank'].notna()], P[P['adp'].notna()]

# ---- career survival by age, from real data: P(still a rotation player next season | relevant now)
panel = pd.read_csv(D / "breakout_panel.csv")
pp = panel[(panel["GP"] >= 20) & (panel["mpg"] >= 10) & (panel["fpg"] >= 25) & (panel["yr"] <= 2024)].copy()
pp["present"] = ((pp["mpg_next"] >= 10) & (pp["GP_next"] >= 20)).astype(int)
pp["ageb"] = pp["AGE"].clip(20, 40).round()
surv_by_age = pp.groupby("ageb")["present"].mean()
# smooth with a monotone-ish rolling fit for older ages
surv_by_age = surv_by_age.rolling(3, center=True, min_periods=1).mean()
print("P(still a rotation player next season | fantasy-relevant now), by age:")
print({int(a): round(float(v), 2) for a, v in surv_by_age.items() if a in (22, 25, 28, 30, 32, 34, 36, 38, 40)})


def surv_path(age0, n=7):
    s, out = 1.0, []
    for j in range(n):
        out.append(s)  # season t=0 is next season -> the player is currently active
        a = float(np.clip(round(age0 + j), 20, 40))
        s *= float(surv_by_age.get(a, surv_by_age.iloc[-1]))
    return np.array(out)


def keep_cutoff(k, y1_sorted):
    if k <= 0:
        return np.inf
    idx = min(int(k * NTEAMS) - 1, len(y1_sorted) - 1)
    return float(y1_sorted[idx])


P["age_f"] = P["age"].fillna(25.0)
y1_sorted = np.sort(P["y1"].to_numpy())[::-1]
REPL = 22.0  # what you'd replace a starter with for next season (freely available level)


def asset_value(traj, age, k, delta, ceil_a, use_surv=True, tail_fix=True):
    t = np.asarray(traj, dtype=float)[:7]
    n = len(t)
    up = 1.0 + ceil_a * max(0.0, 24.0 - age) / 4.0
    s = SD[:n] * up
    S = surv_path(age, n) if use_surv else np.ones(n)
    c = np.array([REPL] + [max(keep_cutoff(k, y1_sorted), REPL)] * (n - 1))
    z = (t - c) / s
    ev = np.where(np.isfinite(c), (t - c) * norm.cdf(z) + s * norm.pdf(z), 0.0)
    return float((delta ** np.arange(n) * S * ev).sum())


def score(k, delta, a, surv=True):
    v = pd.Series([asset_value(tr, ag, k, delta, a, surv) for tr, ag in zip(P["traj"], P["age_f"])], index=P.index)
    return v


print("\n== which ingredients close the gap to the dynasty crowd? (K=19 ~ full dynasty, delta = yearly discount)")
print(f"  {'delta':>5s} {'upside a':>8s} {'survival':>8s} | {'vs dynasty':>10s} {'vs ADP':>7s}")
res = []
for surv in (False, True):
    for a in (0.0, 0.5, 1.0, 2.0):
        for delta in (0.85, 0.95, 1.0):
            v = score(19, delta, a, surv)
            d_c, a_c = -spearmanr(v[dm.index], dm["dyn_rank"])[0], -spearmanr(v[am.index], am["adp"])[0]
            res.append((delta, a, surv, d_c, a_c))
for delta, a, surv, d_c, a_c in sorted(res, key=lambda x: -x[3])[:8]:
    print(f"  {delta:5.2f} {a:8.1f} {str(surv):>8s} | {d_c:10.3f} {a_c:7.3f}")
print("  ... (baseline: VOR@K=19 vs dynasty 0.786, year-1 vs ADP 0.853)")
bd = max(res, key=lambda x: x[3])
BEST = dict(delta=bd[0], a=bd[1], surv=bd[2])
print("best dynasty-matching setting:", BEST, f"rho={bd[3]:.3f}")

print("\n== does K now move the ranking the right way? rank among all hub players (1 = most valuable)")
Ks = [0, 1, 3, 5, 8, 19]
tab = {}
for k in Ks:
    tab[k] = score(k, BEST["delta"], BEST["a"], BEST["surv"])
    d_c, a_c = -spearmanr(tab[k][dm.index], dm["dyn_rank"])[0], -spearmanr(tab[k][am.index], am["adp"])[0]
    print(f"  K={k:2d}: rho vs dynasty {d_c:.3f} | vs ESPN ADP {a_c:.3f}   (keep cutoff c = {keep_cutoff(k, y1_sorted):.1f} pts/g)")
who = ["Cameron Boozer", "Cooper Flagg", "Dylan Harper", "AJ Dybantsa", "Jalen Johnson", "Victor Wembanyama", "Kevin Durant", "Stephen Curry", "LeBron James"]
print(f"\n  {'':20s}" + "".join(f"K={k:<4d}" for k in Ks) + "  | real: ADP  dynasty")
for w in who:
    row = P[P["player"] == w]
    if row.empty:
        continue
    i = row.index[0]
    ranks = [int((tab[k] > tab[k][i]).sum() + 1) for k in Ks]
    ad = row["adp"].iloc[0]
    dy = row["dyn_rank"].iloc[0]
    print(f"  {w:20s}" + "".join(f"{r:<6d}" for r in ranks) + f"  | {ad if pd.notna(ad) else float('nan'):6.0f} {dy if pd.notna(dy) else float('nan'):7.0f}")

print("\n== top 20 by asset value at K=0 / K=5 / K=19")
tops = {k: tab[k].sort_values(ascending=False).head(20).index for k in (0, 5, 19)}
for i in range(20):
    print(f"  {i+1:2d}  " + " | ".join(f"{P.loc[tops[k][i], 'player']:24s}" for k in (0, 5, 19)))
yng = P[(P["age_f"] <= 21.5) & P["dyn_rank"].notna()]
print(f"\nyoung (<=21.5) players in dynasty top-200: n={len(yng)}")
for k in (0, 5, 19):
    print(f"  K={k:2d}: rho vs dynasty within young subset {-spearmanr(tab[k][yng.index], yng['dyn_rank'])[0]:.3f}")
old = P[(P["age_f"] >= 31) & P["dyn_rank"].notna()]
print(f"old (>=31) players in dynasty top-200: n={len(old)}")
for k in (0, 5, 19):
    print(f"  K={k:2d}: rho vs dynasty within old subset {-spearmanr(tab[k][old.index], old['dyn_rank'])[0]:.3f}")
