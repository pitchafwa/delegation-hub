"""Fix the aging SHAPE used to project players forward.

Problem (measured): the per-stat aging curve is one constant up-slope to a single peak age, then one
constant down-slope. Real careers grow faster early and decay faster late, so trajectories were too flat
in both directions (young players +3-4 pts/g too low, 31+ players 2-5 pts/g too high; top-3 picks +8 by
year 3).

Method: keep the Kalman FILTER exactly as is (it tracks a player's current level), and replace only the
forward-projection slope with a smooth age-dependent annual slope  slope_stat(age)  (piecewise-linear in
age, knots 19..38), fit by regression to REAL outcomes:
    for many forecast origins (season-ends 2012-2021) and horizons 1-4 seasons:
        actual_rate(target season) - filter_state_at_origin - league_era_drift
                 = sum_j c_j * (years spent near age knot j on the projection path)
    ridge + second-difference smoothing, weighted by target-season minutes.
League-wide drift (e.g. the 3-point boom) is removed from targets so it isn't mistaken for aging.
Evaluation set = every player who was a rotation player at the origin (>=800 min) -- selected on what
was known THEN, no look-ahead to who later became a star.

Validation: fit on outcomes through 2019 only, test on later origins/targets, compare with the old
(peak, slope_up, slope_down) shape on fantasy pts/g bias by age group and RMSE.

Writes data/aging_shape.json (final fit on all data) and data/aging_shape_validation.json.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from kalman_engine import run_filter_all_players  # noqa: E402

D = ROOT / "data"
KSTATS = ["PTS", "REB", "AST", "BLK", "TOV", "FG3M", "FTM", "FTA", "MIN"]
KNOTS = np.array([19, 21, 23, 25, 27, 29, 31, 33, 35, 38], dtype=float)
LAMBDA = 2.0          # smoothness (second differences) relative to weighted squared error
ORIGINS = list(range(2012, 2025))  # origin season START years (last observed season)
MAXH = 4
LAST_SEASON = 2025

df = pd.read_csv(D / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
df["sy"] = df["SEASON"].str[:4].astype(int)
season_start = df.groupby("sy")["GAME_DATE"].min()
params = {s: json.load(open(D / f"kalman_fit_{s}.json"))["params"] for s in KSTATS}

# ---- per player-season actuals (rate per minute; MIN = minutes per game)
g = df.groupby(["PLAYER_ID", "sy"])
act = g.agg(gp=("MIN", "size"), minutes=("MIN", "sum")).reset_index()
for s in KSTATS + ["STL", "TD3"]:
    if s == "MIN":
        act["MIN"] = (g["MIN"].mean()).values
    elif s == "TD3":
        act["TD3"] = (g["TD3"].sum() / g.size()).values
    else:
        act[s] = (g[s].sum() / g["MIN"].sum()).values
act = act[(act["gp"] >= 20) & (act["minutes"] >= 300)].set_index(["PLAYER_ID", "sy"])
# league-average rate by season (minutes-weighted) -> era drift to strip out
league = {}
for s in KSTATS:
    if s == "MIN":
        league[s] = act.groupby("sy")["MIN"].mean()
    else:
        tot = df.groupby("sy")[s].sum() / df.groupby("sy")["MIN"].sum()
        league[s] = tot


def hat(age):
    """piecewise-linear hat-function basis weights over KNOTS for scalar or array ages"""
    a = np.clip(np.atleast_1d(age).astype(float), KNOTS[0], KNOTS[-1])
    B = np.zeros((len(a), len(KNOTS)))
    for j in range(len(KNOTS)):
        left = KNOTS[j - 1] if j > 0 else KNOTS[0] - 1
        right = KNOTS[j + 1] if j < len(KNOTS) - 1 else KNOTS[-1] + 1
        up = np.clip((a - left) / (KNOTS[j] - left), 0, 1)
        dn = np.clip((right - a) / (right - KNOTS[j]), 0, 1)
        B[:, j] = np.minimum(up, dn)
    return B


def old_slope(stat, age):
    Q, R, peak, su, sd = params[stat]
    return np.where(np.asarray(age) - peak <= 0, su, sd)


# ---- filter at each origin, per stat
rows = []
t0 = time.time()
for stat in KSTATS:
    Q, R, peak, su, sd = params[stat]
    is_min = stat == "MIN"
    for oy in ORIGINS:
        fit = df[df["sy"] <= oy]
        pid = fit["PLAYER_ID"].to_numpy(); days = fit["DAYS_SINCE_LAST"].to_numpy(dtype=float); age = fit["AGE_AT_GAME"].to_numpy(dtype=float)
        mins = fit["MIN"].to_numpy(dtype=float)
        obs = mins if is_min else fit[stat].to_numpy(dtype=float)
        gain = np.ones_like(mins) if is_min else mins
        x0 = float(mins.sum()) / len(mins) if is_min else float(obs.sum()) / max(float(gain.sum()), 1.0)
        post = run_filter_all_players(pid, days, age, gain, obs, Q, R, peak, su, sd, x0)
        f2 = fit.assign(_post=post)
        last = f2.groupby("PLAYER_ID").tail(1)
        last = last[last["sy"] == oy].set_index("PLAYER_ID")
        m_season = fit[fit["sy"] == oy].groupby("PLAYER_ID")["MIN"].sum()
        last = last[m_season.reindex(last.index).fillna(0) >= 800]
        nstart = season_start.get(oy + 1, pd.NaT)
        for h in range(1, MAXH + 1):
            ty = oy + h
            if ty > LAST_SEASON:
                continue
            ta = act.xs(ty, level="sy") if ty in act.index.get_level_values("sy") else None
            if ta is None:
                continue
            common = last.index.intersection(ta.index)
            if len(common) < 30:
                continue
            gap = ((nstart - last.loc[common, "GAME_DATE"]).dt.days.clip(lower=0) / 365.0).to_numpy() if pd.notna(nstart) else np.full(len(common), 0.5)
            era = float(league[stat].get(ty, np.nan) - league[stat].get(oy, np.nan))
            rows.append(pd.DataFrame({
                "stat": stat, "origin": oy, "h": h, "PLAYER_ID": common, "b": last.loc[common, "_post"].to_numpy(),
                "y": ta.loc[common, stat].to_numpy(), "w": ta.loc[common, "minutes"].to_numpy(), "age0": last.loc[common, "AGE_AT_GAME"].to_numpy(),
                "gap": gap, "era": era}))
    print(f"  filter runs done for {stat} ({time.time()-t0:.0f}s)", flush=True)
R = pd.concat(rows, ignore_index=True)
print(f"regression rows: {len(R)} across {R['origin'].nunique()} origins")


def exposure(age0, gap, h):
    """sum over the projection path of hat-basis weights: partial year at age0, then (h-1) whole-year steps at age0+k"""
    E = hat(age0) * gap[:, None]
    for k in range(0, h - 1):
        E = E + hat(age0 + k)
    return E


def exposure_all(df_):
    E = np.zeros((len(df_), len(KNOTS)))
    for h in df_["h"].unique():
        m = (df_["h"] == h).to_numpy()
        E[m] = exposure(df_.loc[m, "age0"].to_numpy(), df_.loc[m, "gap"].to_numpy(), int(h))
    return E


def old_path(df_):
    out = np.zeros(len(df_))
    for stat in df_["stat"].unique():
        for h in df_["h"].unique():
            m = ((df_["stat"] == stat) & (df_["h"] == h)).to_numpy()
            if not m.any():
                continue
            a0, gp = df_.loc[m, "age0"].to_numpy(), df_.loc[m, "gap"].to_numpy()
            tot = gp * old_slope(stat, a0) / 1.0
            for k in range(0, int(h) - 1):
                tot = tot + old_slope(stat, a0 + k)
            out[m] = tot
    return out


D2 = np.diff(np.eye(len(KNOTS)), n=2, axis=0)  # second-difference penalty matrix


def fit_shape(sub, stat):
    s = sub[sub["stat"] == stat]
    E = exposure_all(s)
    d = (s["y"] - s["b"] - s["era"]).to_numpy()
    w = s["w"].to_numpy()
    w = w / w.mean()
    A = E.T @ (E * w[:, None]) + LAMBDA * len(s) * 1e-3 * (D2.T @ D2) + 1e-9 * np.eye(len(KNOTS))
    c = np.linalg.solve(A, E.T @ (w * d))
    return c


def predict_new(sub, coefs):
    out = np.zeros(len(sub))
    for stat in sub["stat"].unique():
        m = (sub["stat"] == stat).to_numpy()
        out[m] = exposure_all(sub[m]) @ coefs[stat]
    return out


# ---- validation: fit only on targets <= 2019, test on later targets
TRAIN_MAXT = 2019
fit_rows = R[R["origin"] + R["h"] <= TRAIN_MAXT]
test_rows = R[(R["origin"] + R["h"] > TRAIN_MAXT) & (R["origin"] >= 2016)].copy()
coefs_val = {s: fit_shape(fit_rows, s) for s in KSTATS}
OLDP, NEWP = old_path(test_rows), predict_new(test_rows, coefs_val)
test_rows["pred_old"] = (test_rows["b"] + OLDP).clip(lower=0)
test_rows["pred_new"] = (test_rows["b"] + NEWP).clip(lower=0)
test_rows["pred_b50"] = (test_rows["b"] + 0.5 * OLDP + 0.5 * NEWP).clip(lower=0)  # global 50/50 (for reference)
def mixed_path(df_, coefs, tmax):
    """keep the OLD slope below ~28, ramp to the fitted slope by 31+ (per projection step, by that step's age)"""
    out = np.zeros(len(df_))
    for stat in df_["stat"].unique():
        for h in df_["h"].unique():
            m = ((df_["stat"] == stat) & (df_["h"] == h)).to_numpy()
            if not m.any():
                continue
            a0, gp = df_.loc[m, "age0"].to_numpy(), df_.loc[m, "gap"].to_numpy()

            def u(a):
                new = hat(a) @ coefs[stat]
                old = old_slope(stat, a)
                return old + tmax * np.clip((a - 28.0) / 3.0, 0, 1) * (new - old)

            tot = gp * u(a0)
            for k in range(0, int(h) - 1):
                tot = tot + u(a0 + k)
            out[m] = tot
    return out


test_rows["pred_split"] = (test_rows["b"] + mixed_path(test_rows, coefs_val, 1.0)).clip(lower=0)   # old below 28, fitted at 31+
test_rows["pred_b75"] = (test_rows["b"] + mixed_path(test_rows, coefs_val, 0.6)).clip(lower=0)     # same, but only 60% of the way
print(f"\nvalidation: fit on {len(fit_rows)} rows (targets <= {TRAIN_MAXT}); test on {len(test_rows)} rows (later targets)")
print("per-stat weighted RMSE on held-out targets: old shape -> new shape")
val = {"per_stat": {}}
for s in KSTATS:
    t = test_rows[test_rows["stat"] == s]
    w = t["w"] / t["w"].sum()
    ro, rn = np.sqrt((w * (t["y"] - t["pred_old"]) ** 2).sum()), np.sqrt((w * (t["y"] - t["pred_new"]) ** 2).sum())
    bo, bn = (w * (t["y"] - t["pred_old"] - t["era"])).sum(), (w * (t["y"] - t["pred_new"] - t["era"])).sum()
    val["per_stat"][s] = {"rmse_old": float(ro), "rmse_new": float(rn)}
    print(f"  {s:4s} {ro:.4f} -> {rn:.4f}  ({(1 - rn / ro) * 100:+.1f}%)")

# ---- fantasy pts/g composition on held-out (needs every stat's prediction for the same player-origin-horizon)
wide = {}
VARS = ["pred_old", "pred_new", "pred_b50", "pred_b75", "pred_split"]
for col in ["y"] + VARS:
    wide[col] = test_rows.pivot_table(index=["origin", "h", "PLAYER_ID"], columns="stat", values=col)
ok = wide["y"].dropna().index
info = test_rows.drop_duplicates(["origin", "h", "PLAYER_ID"]).set_index(["origin", "h", "PLAYER_ID"]).loc[ok, ["age0", "w"]]
STL_TD3 = act[["STL", "TD3"]]


def fantasy(rates, stl, td3):
    m = rates["MIN"]
    return (rates["PTS"] * m + 1.5 * rates["REB"] * m + 2 * rates["AST"] * m + 3 * stl * m + 3 * rates["BLK"] * m + rates["FG3M"] * m
            + 2 * rates["FTM"] * m - rates["FTA"] * m - rates["TOV"] * m + 3 * td3)


idx = pd.MultiIndex.from_arrays([[i[2] for i in ok], [i[0] + i[1] for i in ok]])
stl = STL_TD3["STL"].reindex(idx).to_numpy(); td3 = STL_TD3["TD3"].reindex(idx).to_numpy()
fy = fantasy(wide["y"].loc[ok], stl, td3)
fvars = {v: fantasy(wide[v].loc[ok], stl, td3).to_numpy() for v in VARS}
res = pd.DataFrame({"y": fy.to_numpy(), **fvars, "age0": info["age0"].to_numpy() + 1.0, "w": info["w"].to_numpy()}).dropna()
# grading set = players the OLD model already expected to be fantasy-relevant at the origin (known then; no peeking at outcomes)
res = res[res["pred_old"] >= 20]
res["grp"] = pd.cut(res["age0"], [17, 21.5, 23.5, 26.5, 30.5, 45], labels=["<=21", "22-23", "24-26", "27-30", "31+"])
names = {"pred_old": "old", "pred_new": "new", "pred_b50": "global50", "pred_b75": "fix31+ (60%)", "pred_split": "fix31+ (100%)"}
print("\nfantasy pts/g on held-out targets: bias (actual - forecast; + = forecast too LOW) | RMSE")
print(f"{'age':>6s} {'n':>5s} " + " ".join(f"{names[v]:>16s}" for v in VARS))
val["fantasy_by_age"] = {}
for grp, t in list(res.groupby("grp")) + [("ALL", res)]:
    row = {}
    line = f"{str(grp):>6s} {len(t):5d} "
    for v in VARS:
        b, r = float((t["y"] - t[v]).mean()), float(np.sqrt(((t["y"] - t[v]) ** 2).mean()))
        row[names[v]] = {"bias": b, "rmse": r}
        line += f" {b:+6.2f} | {r:6.2f}  "
    print(line)
    val["fantasy_by_age"][str(grp)] = {"n": int(len(t)), **row}
val["fantasy_overall"] = val["fantasy_by_age"]["ALL"]

# ---- final fit on everything -> production file
coefs = {s: fit_shape(R, s) for s in KSTATS}
out = {"mix_strength": 0.6, "ramp_ages": [28.0, 31.0], "knots": KNOTS.tolist(), "slopes": {s: [float(x) for x in coefs[s]] for s in KSTATS},
       "note": "annual per-stat slope (rate units/yr) at each age knot; np.interp between knots, flat outside 19-38. Era drift removed.",
       "fit_rows": int(len(R)), "origins": [ORIGINS[0], ORIGINS[-1]]}
(D / "aging_shape.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
(D / "aging_shape_validation.json").write_text(json.dumps(val, indent=1), encoding="utf-8")
print("\nfitted annual slopes (per stat) at ages", KNOTS.astype(int).tolist())
for s in KSTATS:
    print(f"  {s:4s}", np.round(coefs[s], 4).tolist(), f"| old: up {params[s][3]:+.4f} to age {params[s][2]:.0f}, then {params[s][4]:+.4f}")
