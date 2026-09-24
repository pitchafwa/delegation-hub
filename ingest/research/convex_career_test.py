"""Convex CAREER value: price the variance that persists across a career.

Market value is convex in career value (elite assets are worth disproportionately more; the top of a roster is harder to replace).
A young player's career outcome is highly variable AND persistent (a hit stays a hit for years), so his expected market value is
above the market value of his average outcome. The earlier attempt applied convexity per season with independent seasons, which
averages the variance away. Here we simulate whole careers:
   path_h = E_h + dev_h,  dev ~ N(0, D R D)   D = the spread of real outcomes for that age group / draft tier (by season),
                                              R = year-to-year correlation of deviations (AR(1), estimated from real data)
   in the league at season h iff U < P(present at h)  (one draw per career: once out, out)
   career value V = sum_h delta^(h-1) * in_league_h * (path_h - c_h(K))+            (same keep-cutoff logic as the asset value)
   value_gamma = ( E[V^gamma] )^(1/gamma)          gamma = 1 -> the ordinary expected asset value; gamma > 1 -> convex market value
Fit gamma on ONE outlet (Hashtag crowd), test on the other (RotoWire) and vice versa.
"""
import contextlib
import re
import sys
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


class _Q:
    def write(self, s): pass
    def flush(self): pass
    def reconfigure(self, **k): pass


src = open("horizon_test.py", encoding="utf-8").read()
head = src.split('names = list(live["PLAYER_NAME"])')[0]
with contextlib.redirect_stdout(_Q()):
    exec(head)  # fc, live, E_v/E_p -> Ev10/Ep10, Sv10/Sp10, cut(), Rv/Rp analog residual vectors, pres_res, ...

DELTA_SIM, HH, NSIM = 0.95, 10, 1500
rng = np.random.default_rng(3)
names = list(live["PLAYER_NAME"]) + [q["player"] for q in pros]
ages = np.concatenate([live_age_next, pr_age])

# ---- spread by season for each player's group, and the AR(1) persistence of deviations
Rall = pd.DataFrame(np.vstack([Rv, Rp]))
cm = Rall.corr(min_periods=20).to_numpy()
rho = float(np.nanmean([cm[i, i + 1] for i in range(5)]))
print(f"year-to-year correlation of deviations from projection (real data): adjacent-year mean = {rho:.2f}")
rho = float(np.clip(rho, 0.3, 0.95))
Rm = rho ** np.abs(np.subtract.outer(np.arange(HH), np.arange(HH)))
sd_vet = {g: np.array([np.std(fc.res[h][g]) if len(fc.res[h][g]) > 30 else np.std(np.concatenate(list(fc.res[h].values()))) for h in range(1, 7)]) for g in range(4)}
sd_pro = {t: np.array([np.std(pres_res[h][t]) if len(pres_res[h][t]) > 20 else np.std(np.concatenate(list(pres_res[h].values()))) for h in range(1, 7)]) for t in range(4)}
ext = lambda a: np.concatenate([a, np.full(HH - 6, a[-1])])
L_vet = {g: np.linalg.cholesky(np.outer(ext(sd_vet[g]), ext(sd_vet[g])) * Rm + 1e-6 * np.eye(HH)) for g in range(4)}
L_pro = {t: np.linalg.cholesky(np.outer(ext(sd_pro[t]), ext(sd_pro[t])) * Rm + 1e-6 * np.eye(HH)) for t in range(4)}

# presence probability by season (cumulative-nonincreasing) for each player
n_v, n_p = len(live), len(pros)
P_v = np.ones((n_v, HH)); P_p = np.ones((n_p, HH))
Xv = live
Xp_ = pd.DataFrame({"logpick": np.log(np.clip(pr_pick, 1, 61)), "dage": pr_age, "talent": pr_talent, "is1": (pr_pick == 1).astype(float)})
for h in range(1, 7):
    P_v[:, h - 1] = fc.s[h].predict_proba(live[F])[:, 1]
    P_p[:, h - 1] = ps_[h].predict_proba(Xp_[PF])[:, 1]
for h in range(7, HH + 1):
    P_v[:, h - 1] = P_v[:, 5] * (Sv10[:, h - 1])
    P_p[:, h - 1] = P_p[:, 5] * (Sp10[:, h - 1])
P_v = np.minimum.accumulate(P_v, axis=1); P_p = np.minimum.accumulate(P_p, axis=1)

grp_v = AGEB(live_age_next)
grp_p = np.digitize(pr_pick, [4, 11, 31])
Z = rng.standard_normal((NSIM, HH))
U = rng.random(NSIM)


def career_values(K):
    """simulated career values (NSIM x players)"""
    c_future = max(cut(K), REPL)
    disc = DELTA_SIM ** np.arange(HH)
    out = np.zeros((NSIM, n_v + n_p))
    for i in range(n_v + n_p):
        if i < n_v:
            E, P, L = Ev10[i], P_v[i], L_vet[grp_v[i]]
        else:
            E, P, L = Ep10[i - n_v], P_p[i - n_v], L_pro[grp_p[i - n_v]]
        path = np.maximum(E[None, :] + Z @ L.T, 0.0)
        inleague = U[:, None] < P[None, :]
        cutoffs = np.array([REPL] + [c_future] * (HH - 1))
        surplus = np.maximum(path - cutoffs[None, :], 0.0) * inleague
        if not np.isfinite(cut(K)):
            surplus[:, 1:] = 0.0
        out[:, i] = (surplus * disc[None, :]).sum(axis=1)
    return out


def nn_(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


dyn = pd.read_csv(D / "hashtag_dynasty_2026-09-24.csv"); dyn["norm"] = dyn["player"].apply(nn_)
ROTO = {"Victor Wembanyama": 1, "Nikola Jokic": 2, "Shai Gilgeous-Alexander": 3, "Luka Doncic": 4, "Cade Cunningham": 5, "Cooper Flagg": 6, "Jayson Tatum": 7,
        "Jalen Johnson": 8, "Anthony Edwards": 9, "Tyrese Maxey": 10, "Scottie Barnes": 11, "Tyrese Haliburton": 12, "Chet Holmgren": 13, "Josh Giddey": 14,
        "Cameron Boozer": 15, "Amen Thompson": 16, "Evan Mobley": 17, "Jalen Williams": 18, "Giannis Antetokounmpo": 19, "Donovan Mitchell": 20,
        "Karl-Anthony Towns": 21, "Deni Avdija": 22, "Devin Booker": 23, "Trey Murphy": 24, "Brandon Miller": 25, "Jalen Brunson": 26, "Alperen Sengun": 27,
        "Bam Adebayo": 28, "LaMelo Ball": 29, "Trae Young": 30, "Caleb Wilson": 37, "Darryn Peterson": 38, "Dylan Harper": 40, "Kon Knueppel": 41,
        "AJ Dybantsa": 42, "VJ Edgecombe": 46, "Ace Bailey": 97}
rw = pd.DataFrame({"norm": [nn_(k) for k in ROTO], "rw": list(ROTO.values())})
GAMMAS = (1.0, 1.25, 1.5, 2.0, 3.0)
results, values = {}, {}
for K in (5, 19):
    V = career_values(K)
    for g in GAMMAS:
        ce = (np.mean(V ** g, axis=0)) ** (1.0 / g)
        values[(K, g)] = ce
        b = pd.DataFrame({"player": names, "age": ages, "v": ce}); b["norm"] = b["player"].apply(nn_)
        b = b.drop_duplicates("norm").merge(dyn[["norm", "rank"]].rename(columns={"rank": "hash"}), on="norm", how="left").merge(rw, on="norm", how="left")
        h_, r_ = b.dropna(subset=["hash"]), b.dropna(subset=["rw"])
        results[(K, g)] = (-spearmanr(h_.v, h_.hash)[0], -spearmanr(r_.v, r_.rw)[0])
print("\nagreement with the two independent expert sources as the career-level convexity gamma rises (gamma=1 = plain expected value)")
for K in (5, 19):
    print(f"\nK={K}:  gamma | Hashtag crowd (n=173) | RotoWire (n=37)")
    for g in GAMMAS:
        a, b = results[(K, g)]
        print(f"        {g:4.2f} |   {a:.3f}               |  {b:.3f}")
print("\nrank at K=19 (1=most valuable) by gamma  [Hashtag | RotoWire]")
print(f"{'':16s} " + " ".join(f"g={g:<4}" for g in GAMMAS))
sers = {g: pd.Series(values[(19, g)], index=names).groupby(level=0).first() for g in GAMMAS}
for who in ["Cooper Flagg", "Cameron Boozer", "AJ Dybantsa", "Darryn Peterson", "Caleb Wilson", "Dylan Harper", "Ace Bailey", "Kon Knueppel", "VJ Edgecombe", "Nikola Jokić", "LeBron James", "Kevin Durant"]:
    if who not in sers[1.0].index:
        continue
    row = f"{who:16s} " + " ".join(f"{int((sers[g] > sers[g][who]).sum() + 1):6d}" for g in GAMMAS)
    hh, rr = dyn[dyn.norm == nn_(who)]["rank"], rw[rw.norm == nn_(who)]["rw"]
    print(row + f"   | {int(hh.iloc[0]) if len(hh) else '':>4} | {int(rr.iloc[0]) if len(rr) else '':>4}")
np.save(D / "_career_values_K19.npy", np.vstack([values[(19, g)] for g in GAMMAS]))
