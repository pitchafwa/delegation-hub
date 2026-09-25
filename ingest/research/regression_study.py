"""Buy-low / sell-high: which under- and over-performances REVERT and which are REAL?

For every player-season t (30+ games, 12+ mpg) we predict the change in league-scored fantasy points per game next season, y = fpg(t+1) - fpg(t), from things
that identify luck versus role:
  d_hist   fpg(t) minus his average over the previous two seasons (a spike or a slump versus his own recent level)
  luck_3   shooting luck on threes and twos: made shots minus attempts x his own prior career %, in fantasy points/game (3PM is worth 4: 3 PTS + 1; 2PM worth 2)
  luck_ft  free-throw luck the same way (a made FT is worth 2: 1 PT + 1 FTM)
  mpg_chg  change in minutes per game (a role change persists, a shooting streak does not)
  gp_frac  games played / 82 (an injury-shortened season tends to bounce back)
  age, age^2, fpg level
Fit (ridge) on seasons up to 2020 and test on 2021-2024 targets, out of sample. Then for the latest season it splits each player's predicted change into these drivers and
labels him:  SELL-HIGH (expected to fall, mostly luck/spike),  BUY-LOW (expected to rise: bad shooting luck, or an injury-shortened season),  REAL (a big deviation from his history
that the model does NOT expect to revert: a role/minutes change), or nothing.
Writes dashboard/regression_signals.json (keyed by hub id 'c<PLAYER_ID>') and prints the study.
Run from ingest/:  uv run python research/regression_study.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
HUB = R.parent.parent / "dashboard"
b = pd.read_csv(R / "data" / "player_season_base.csv")
b["end"] = b.SEASON.str[:4].astype(int) + 1
b = b[b.GP > 0].copy()
b["fpg"] = (b.PTS + 1.5 * b.REB + 2 * b.AST + 3 * b.STL + 3 * b.BLK + b.FG3M + 2 * b.FTM - b.FTA - b.TOV + 3 * b.TD3) / b.GP
b["mpg"] = b.MIN / b.GP
b["fg2m"], b["fg2a"] = b.FGM - b.FG3M, b.FGA - b.FG3A
b = b.sort_values(["PLAYER_ID", "end"]).reset_index(drop=True)
LG = {"p3": 0.355, "p2": 0.52, "ft": 0.78}
K_SHRINK = {"p3": 250.0, "p2": 500.0, "ft": 150.0}      # attempts of prior evidence that equal one league-average prior


def prior_pct(g, made, att, key):
    """career % BEFORE each season, attempt-weighted and shrunk toward the league average"""
    cm = g[made].cumsum() - g[made]
    ca = g[att].cumsum() - g[att]
    return (cm + LG[key] * K_SHRINK[key]) / (ca + K_SHRINK[key])


rows = []
for pid, g in b.groupby("PLAYER_ID"):
    g = g.reset_index(drop=True)
    g["e3"], g["e2"], g["eft"] = prior_pct(g, "FG3M", "FG3A", "p3"), prior_pct(g, "fg2m", "fg2a", "p2"), prior_pct(g, "FTM", "FTA", "ft")
    g["luck_3"] = (4 * (g.FG3M - g.FG3A * g.e3) + 2 * (g.fg2m - g.fg2a * g.e2)) / g.GP
    g["luck_ft"] = 2 * (g.FTM - g.FTA * g.eft) / g.GP
    g["hist"] = (g.fpg.shift(1) + g.fpg.shift(2)) / 2
    g["hist"] = g["hist"].fillna(g.fpg.shift(1))
    g["has_hist"] = g["hist"].notna().astype(float)
    g["d_hist"] = (g.fpg - g["hist"]).fillna(0.0)
    g["mpg_chg"] = (g.mpg - g.mpg.shift(1)).fillna(0.0)
    g["nxt"] = g.fpg.shift(-1)
    g["nxt_gp"] = g.GP.shift(-1)
    g["seasons_before"] = np.arange(len(g))
    rows.append(g)
P = pd.concat(rows, ignore_index=True)
P["gp_frac"] = P.GP / 82.0
P["age"] = P.AGE
P["age2"] = (P.AGE - 27) ** 2
P["luck_3"] = P.luck_3.where(P.seasons_before >= 1, 0.0)      # no prior career to compare with: no luck estimate
P["luck_ft"] = P.luck_ft.where(P.seasons_before >= 1, 0.0)
# ---- age-adjust first: the average change by age is NOT a buy/sell signal (everyone 33 declines); only deviations from it are
usable = (P.GP >= 30) & (P.mpg >= 12)
P["age_b"] = P.AGE.clip(19, 37).astype(int)
tr0 = P[usable & P.nxt.notna() & (P.nxt_gp >= 20) & (P.end <= 2020)]
aging = (tr0.nxt - tr0.fpg).groupby(tr0.age_b).mean()
aging = aging.rolling(3, center=True, min_periods=1).mean()
P["aging"] = P.age_b.map(aging)
P["prev_fpg"] = P.groupby("PLAYER_ID").fpg.shift(1)
P["prev_aging"] = (P.age_b - 1).map(aging)
P["surprise"] = (P.fpg - (P.prev_fpg + P.prev_aging)).fillna(0.0)          # this season minus what last season + normal aging predicted
P["y"] = (P.nxt - P.fpg) - P.aging                                             # next-season change beyond normal aging
FEATS = ["surprise", "luck_3", "luck_ft", "mpg_chg", "gp_frac", "fpg"]
train = P[usable & P.nxt.notna() & (P.nxt_gp >= 20) & (P.end <= 2020) & (P.seasons_before >= 1)]
test = P[usable & P.nxt.notna() & (P.nxt_gp >= 20) & (P.end >= 2021) & (P.seasons_before >= 1)]
mu, sd = train[FEATS].mean(), train[FEATS].std()
model = Ridge(alpha=10.0).fit(((train[FEATS] - mu) / sd).to_numpy(), train.y.to_numpy())
pred = model.predict(((test[FEATS] - mu) / sd).to_numpy())
y = test.y.to_numpy()
print(f"train {len(train):,} player-seasons (<=2020), test {len(test):,} (2021-2024 -> next seasons); target = next-season change beyond normal aging")
print(f"MAE: normal aging only {np.abs(y).mean():.2f} | + reversion model {np.abs(y - pred).mean():.2f} pts/g   (R^2 {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f})")
print("coefficients (pts/g of extra next-season change per 1 pt/g of the feature):")
coef = model.coef_ / sd.to_numpy()
for f, c in zip(FEATS, coef):
    print(f"   {f:9s} {c:+.3f}")
te = test.assign(pred=pred, act=y)
for lab, m in (("model says fall 1.5+ beyond aging", te.pred <= -1.5), ("model says rise 1.5+ beyond aging", te.pred >= 1.5)):
    print(f"   {lab}: n={m.sum():4d}  actual extra change {te.act[m].mean():+.2f}")
print(f"   surprise >= +5 (beat expectation): next-season extra change {te[te.surprise >= 5].act.mean():+.2f} (n={(te.surprise >= 5).sum()});  <= -5: {te[te.surprise <= -5].act.mean():+.2f} (n={(te.surprise <= -5).sum()})")
print(f"   shooting luck >= +2 pts/g: {te[te.luck_3 >= 2].act.mean():+.2f} (n={(te.luck_3 >= 2).sum()});  <= -2: {te[te.luck_3 <= -2].act.mean():+.2f} (n={(te.luck_3 <= -2).sum()})")
print(f"   minutes up 4+ mpg: {te[te.mpg_chg >= 4].act.mean():+.2f} (n={(te.mpg_chg >= 4).sum()});  down 4+: {te[te.mpg_chg <= -4].act.mean():+.2f} (n={(te.mpg_chg <= -4).sum()})")
print(f"   played <50 games: {te[te.GP < 50].act.mean():+.2f} (n={(te.GP < 50).sum()});  70+: {te[te.GP >= 70].act.mean():+.2f}")

# ---------------- current signals (latest season)
last_end = P.end.max()
cur = P[(P.end == last_end) & (P.GP >= 15) & (P.mpg >= 12)].copy()
X = ((cur[FEATS] - mu) / sd).to_numpy()
contrib = pd.DataFrame(X * model.coef_, columns=FEATS, index=cur.index)
cur["pred_delta"] = cur.aging + model.intercept_ + contrib.sum(axis=1)
cur["extra"] = model.intercept_ + contrib.sum(axis=1)     # beyond normal aging
cur["c_luck"] = contrib.luck_3 + contrib.luck_ft
cur["c_hist"] = contrib.surprise
cur["c_role"] = contrib.mpg_chg
cur["c_inj"] = contrib.gp_frac
out = {}
for r in cur.itertuples():
    d = r.extra
    drv = {"shooting": r.c_luck, "spike_or_slump": r.c_hist, "minutes_change": r.c_role, "games_missed": r.c_inj}
    top = sorted(drv.items(), key=lambda kv: -abs(kv[1]))[:2]
    label = None
    if d <= -1.5:
        label = "sell_high"
    elif d >= 1.5:
        label = "buy_low"
    elif abs(r.surprise) >= 4 and abs(d) < 1.0 and r.fpg >= 22:
        label = "real_up" if r.surprise > 0 else "real_down"
    out[f"c{int(r.PLAYER_ID)}"] = {"name": r.PLAYER_NAME, "fpg": round(float(r.fpg), 1), "hist": None if pd.isna(r.hist) else round(float(r.hist), 1), "pred_delta": round(float(d), 1), "surprise": round(float(r.surprise), 1), "label": label,
                                   "luck_pts": round(float(r.luck_3 + r.luck_ft), 1), "gp": int(r.GP),
                                   "drivers": [{"k": k, "pts": round(float(v), 1)} for k, v in top if abs(v) >= 0.4]}
(HUB / "regression_signals.json").write_text(json.dumps({"season": int(last_end), "study": {"mae_persistence": round(float(np.abs(y).mean()), 2), "mae_model": round(float(np.abs(y - pred).mean()), 2),
                                                                                              "n_train": int(len(train)), "n_test": int(len(test))}, "players": out}, separators=(",", ":")), encoding="utf-8")
lab = pd.Series([v["label"] for v in out.values()]).value_counts()
print(f"\nlatest season {last_end}: {len(out)} players scored;", lab.to_dict())
s = pd.DataFrame(out).T
s["pred_delta"] = s.pred_delta.astype(float)
print("\nTOP SELL-HIGH (expected to fall):")
print(s[s.label == "sell_high"].sort_values("pred_delta").head(12)[["name", "fpg", "hist", "pred_delta", "luck_pts", "gp"]].to_string())
print("\nTOP BUY-LOW (expected to rise):")
print(s[s.label == "buy_low"].sort_values("pred_delta", ascending=False).head(12)[["name", "fpg", "hist", "pred_delta", "luck_pts", "gp"]].to_string())
print("\nREAL (big change vs history that should hold):")
print(s[s.label.isin(["real_up", "real_down"])].sort_values("fpg", ascending=False).head(10)[["name", "label", "fpg", "hist", "pred_delta"]].to_string())
