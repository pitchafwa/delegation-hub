"""Study 6: how uncertain is a head-to-head week once some days have been played?  (calibrates the live matchup tracker's win probability)

Data: the league's real 2025-26 daily lineups (league_days_2026.json: every team's started players and points per day) and the real matchup pairings (ESPN).
For every regular-season matchup (7-day weeks; 6-day week 1 and the 14-day week 17 are excluded) and every day boundary d = 0..6 (d days played):
    margin_now      = points so far, team A minus team B
    E_rem           = expected remaining margin from a CRUDE baseline (each team's average points per day in its previous 3 weeks x days left) -- the live tracker will use the
                      cap-aware plan instead, which is more accurate, so the spread measured here is conservative
    resid           = final margin - margin_now - E_rem
We fit sigma(rem days) and check calibration: the predicted win probability Phi((margin_now + E_rem) / sigma) against how often the leader really won.
Writes win_prob_model.json.  Run from ingest/:  uv run python research/win_prob_study.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
D = R / "data"
days = json.load(open(D / "league_days_2026.json"))
lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
raw = lg.espn_request.league_get(params={"view": ["mMatchupScore"]})
pairs = [(m["matchupPeriodId"], m["home"]["teamId"], m["away"]["teamId"]) for m in raw["schedule"] if "away" in m and m.get("playoffTierType") in (None, "NONE")]
# points per team per day (started players only; slot ids 0-11 are starting slots)
daily = {}
mp_days = {}
for sp, recs in days.items():
    for r in recs:
        pts = sum((e[4] or 0) for e in r["e"] if e[2] in range(12) and e[3])
        daily[(r["mp"], r["team"], int(sp))] = pts
        mp_days.setdefault(r["mp"], set()).add(int(sp))
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
rows = []
for mp, a, b in pairs:
    ds = sorted(mp_days.get(mp, []))
    if len(ds) != 7 or mp < 4:
        continue
    # baseline daily rate: each team's average points per day over the previous 3 weeks
    def rate(team):
        prev = [daily.get((m, team, sp), np.nan) for m in range(mp - 3, mp) for sp in sorted(mp_days.get(m, [])) if m not in (1, 17)]
        prev = [x for x in prev if not np.isnan(x)]
        return float(np.mean(prev)) if prev else np.nan
    ra, rb = rate(a), rate(b)
    final = sum(daily.get((mp, a, sp), 0) for sp in ds) - sum(daily.get((mp, b, sp), 0) for sp in ds)
    for d in range(0, 7):
        now = sum(daily.get((mp, a, sp), 0) for sp in ds[:d]) - sum(daily.get((mp, b, sp), 0) for sp in ds[:d])
        rem = 7 - d
        e_rem = (ra - rb) * rem
        rows.append(dict(mp=mp, a=a, b=b, d=d, rem=rem, now=now, e_rem=e_rem, final=final, resid=final - now - e_rem))
df = pd.DataFrame(rows).dropna()
print(f"{df.groupby(['mp', 'a']).ngroups} matchups, {len(df)} day-boundary observations")
sig = {}
print("\nresidual SD of the final margin around (margin so far + baseline remaining margin), by days left:")
for rem in range(1, 8):
    x = df[df.rem == rem]
    sig[rem] = float(x.resid.std())
    print(f"   {rem} days left: SD {sig[rem]:.0f}   (SD per sqrt(day) {sig[rem] / math.sqrt(rem):.0f}; n={len(x)})")
df["mu"] = df.now + df.e_rem
df["sigma"] = df.rem.map(sig)
df["p"] = [Phi(m / s) for m, s in zip(df.mu, df.sigma)]
df["win"] = (df.final > 0).astype(float)
print("\ncalibration of the win probability (all day boundaries):")
df["bin"] = pd.cut(df.p, [0, .1, .2, .35, .5, .65, .8, .9, 1.0])
print(df.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("win", "mean"), n=("win", "size")).round(3).to_string())
print("Brier:", round(float(((df.p - df.win) ** 2).mean()), 4), "vs 0.25 for always-50%")
# the single-game/day granularity: per-day SD of a team's points
dd = pd.Series({k: v for k, v in daily.items() if k[0] not in (1, 17)})
print("\nper-day points of one team: mean %.0f, SD %.0f" % (dd.mean(), dd.std()))
json.dump({"sigma": sig, "note": "SD of final margin around (margin so far + crude baseline remaining margin) by days left in a 7-day week; the live tracker rescales for other lengths with sqrt(days)"}, open(R / "win_prob_model.json", "w"))

# ---- shrink: the baseline remaining margin over-states real differences between teams (regression to the mean)
df["y_rem"] = df.final - df.now
b = float((df.e_rem * df.y_rem).sum() / (df.e_rem ** 2).sum())
print(f"\nslope of the actual remaining margin on the crude expected remaining margin: {b:.2f}")
for beta in (b, 0.72):
    res = df.y_rem - beta * df.e_rem
    sg = {rem: float(res[df.rem == rem].std()) for rem in range(1, 8)}
    p = [Phi((n + beta * e) / sg[r_]) for n, e, r_ in zip(df.now, df.e_rem, df.rem)]
    tmp = df.assign(p=p, bin=pd.cut(pd.Series(p, index=df.index), [0, .1, .2, .35, .5, .65, .8, .9, 1.0]))
    print(f"\nwith shrink {beta:.2f} (residual SD by days left: {[round(v) for v in sg.values()]}): calibration")
    print(tmp.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("win", "mean"), n=("win", "size")).round(3).to_string())
    print("Brier", round(float(((tmp.p - tmp.win) ** 2).mean()), 4))
    if beta == 0.72:
        json.dump({"beta": 0.72, "sigma": sg, "sigma_unshrunk": sig, "note": "win probability = Phi((margin so far + 0.72 x expected remaining margin) / sigma(days left)); sigma in points of margin for a 7-day week, rescaled by sqrt(days) otherwise"}, open(R / "win_prob_model.json", "w"))

# ---- symmetrise (each matchup counted from both sides) and recalibrate
print("\nmean residual (bias) for the home-listed team:", round(float((df.final - df.now - 0.72 * df.e_rem).mean()), 1))
m = df.copy()
m["now"], m["e_rem"], m["final"] = -m.now, -m.e_rem, -m.final
S2 = pd.concat([df, m], ignore_index=True)
S2["y_rem"] = S2.final - S2.now
S2["win"] = (S2.final > 0).astype(float)
for beta in (0.72,):
    res = S2.y_rem - beta * S2.e_rem
    sg = {rem: float(res[S2.rem == rem].std()) for rem in range(1, 8)}
    for scale in (1.0, 1.1, 1.2):
        p = [Phi((n + beta * e) / (sg[r_] * scale)) for n, e, r_ in zip(S2.now, S2.e_rem, S2.rem)]
        t2 = S2.assign(p=p, bin=pd.cut(pd.Series(p, index=S2.index), [0, .1, .2, .35, .5, .65, .8, .9, 1.0]))
        print(f"\nsymmetrised, shrink {beta}, sigma x {scale}: Brier {((t2.p - t2.win) ** 2).mean():.4f}")
        print(t2.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("win", "mean"), n=("win", "size")).round(3).T.to_string())
    json.dump({"beta": beta, "sigma": sg, "note": "win probability = Phi((margin so far + 0.72 x expected remaining margin) / sigma(days left)); sigma = SD of the final margin in a 7-day week, rescaled by sqrt(days left / 7) otherwise (measured on 2025-26, both sides of every matchup)"}, open(R / "win_prob_model.json", "w"))
