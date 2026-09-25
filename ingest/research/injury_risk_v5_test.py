"""Redo of the tier model with (a) an INJURY-only outcome (game days listed Out with an injury/illness reason, from the reports) instead of games-played shortfall
(which also counts coaching decisions, rest and load management) and (b) longer-memory history features (career-ish average, count of bad seasons)."""
import sys, os
sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parent.as_posix())
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import injury_common as C
sys.stdout.reconfigure(encoding="utf-8")
sb = pd.read_csv(C.D / "player_season_base.csv"); sb["yr"] = sb.SEASON.str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["len"] = np.where(sb.yr.isin([2019, 2020]), 72, 82); sb["missed"] = (1 - sb.GP / sb.len).clip(lower=0); sb["mpg"] = sb.MIN / sb.GP
S = sb.set_index(["PLAYER_ID", "yr"])
rows = pd.read_pickle(C.D / "injury_rows_05pm.pkl")
lg = pd.read_csv(C.D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates(); lg["player_key"] = lg.PLAYER_NAME.map(C.key_of_log)
k2 = lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
rows["pid"] = rows.player_key.map(k2)
rows["yr"] = rows.season.str[:4].astype(int)
io = rows[(rows.status == "Out") & rows.kind.isin(["injury", "illness"]) & rows.pid.notna()].groupby(["pid", "yr"]).size()      # game days listed Out with an injury/illness reason (5:30 PM report)
IO = io.to_dict()
recs = []
for t in range(2022, 2026):
    p2 = sb[(sb.yr == t - 2) & (sb.mpg >= 15) & (sb.GP >= 20)].PLAYER_ID
    prev = sb[(sb.yr == t - 1) & (sb.GP >= 5) & ((sb.mpg >= 15) | sb.PLAYER_ID.isin(p2))]
    for r in prev.itertuples():
        pid = r.PLAYER_ID
        if (pid, t) not in S.index or S.loc[(pid, t)].GP < 1:
            continue
        ms = [S.loc[(pid, t - k)].missed if (pid, t - k) in S.index else np.nan for k in range(1, 6)]
        m1, m2, m3 = ms[0], ms[1], ms[2]
        have = [m for m in ms if not np.isnan(m)]
        f = dict(t=t, pid=pid, m1=m1, m2=m2 if not np.isnan(m2) else m1, m3=m3 if not np.isnan(m3) else m1, age=r.AGE + 1, mpg=r.mpg,
                 mean4=np.mean(have[:4]), mean5=np.mean(have), nbad=sum(m >= 0.25 for m in have[:4]), maxm=max(have[:4]), nseas=len(have),
                 io1=IO.get((pid, t - 1), 0) if t - 1 >= 2021 else np.nan, io2=IO.get((pid, t - 2), 0) if t - 2 >= 2021 else np.nan,
                 y_gp=int(S.loc[(pid, t)].missed >= 0.30), y_inj=IO.get((pid, t), 0), gp=S.loc[(pid, t)].GP)
        recs.append(f)
D = pd.DataFrame(recs)
D["io1"] = D.io1.fillna(D.m1 * 82 * 0.6); D["io2"] = D.io2.fillna(D.io1)          # before 2021 there are no reports: proxy from games missed
print(len(D), "player-seasons")
print(f"share missing 25+ games (any reason): {D.y_gp.mean():.3f};  game days listed Out for injury/illness: median {D.y_inj.median():.0f}, mean {D.y_inj.mean():.1f}")
for th in (10, 15, 20, 25):
    print(f"   share with {th}+ injury game-days out: {(D.y_inj >= th).mean():.3f}")
print("correlation of the two outcomes:", round(np.corrcoef(D.y_gp, (D.y_inj >= 15))[0, 1], 3))
# yo-yo group: healthy last season but bad earlier
yo = D[(D.m1 < 0.2) & ((D.m2 + D.m3) / 2 > 0.35)]
print(f"\nyo-yo players (missed <20% last year but avg >35% the two years before): n={len(yo)}; next season 25+ games missed (any reason) {yo.y_gp.mean():.3f}, 15+ injury days out {(yo.y_inj >= 15).mean():.3f}"
      f"  | vs healthy-both {(D[(D.m1 < 0.2) & ((D.m2 + D.m3) / 2 < 0.2)].y_gp.mean()):.3f}, {((D[(D.m1 < 0.2) & ((D.m2 + D.m3) / 2 < 0.2)].y_inj >= 15).mean()):.3f}")


def cv(cols, y, name):
    pr = np.zeros(len(D))
    for t in sorted(D.t.unique()):
        tr, te = D.t != t, D.t == t
        mu, sd = D.loc[tr, cols].mean(), D.loc[tr, cols].std().replace(0, 1)
        m = LogisticRegression(C=0.5, max_iter=3000).fit((D.loc[tr, cols] - mu) / sd, y[tr])
        pr[te.to_numpy()] = m.predict_proba((D.loc[te, cols] - mu) / sd)[:, 1]
    print(f"   {name:52s} AUC {roc_auc_score(y, pr):.3f}  Brier {brier_score_loss(y, pr):.4f}")
    return pr
B0 = ["m1", "m2", "m3", "age", "mpg"]
B1 = B0 + ["mean4", "nbad", "maxm"]
B2 = B1 + ["io1", "io2"]
for oname, y in (("ANY-REASON 25+ games missed", D.y_gp), ("INJURY 15+ game days out", (D.y_inj >= 15).astype(int)), ("INJURY 10+ game days out", (D.y_inj >= 10).astype(int))):
    print("\noutcome:", oname, f"(base rate {y.mean():.3f})")
    cv(B0, y, "last 3 seasons + age + minutes (current)")
    cv(B1, y, "+ 4-season average, count of bad seasons, worst")
    cv(B2, y, "+ injury game-days out in the last two seasons")
D.to_pickle(C.D / "injury_v5_frame.pkl")

# ---- expected injury game-days regression + percentile tiers, leave-one-season-out
from sklearn.linear_model import Ridge
print("\n=== expected injury game-days out (regression) ===")
cols = B2
pred = np.zeros(len(D))
for t in sorted(D.t.unique()):
    tr, te = D.t != t, D.t == t
    mu, sd = D.loc[tr, cols].mean(), D.loc[tr, cols].std().replace(0, 1)
    m = Ridge(alpha=20.0).fit((D.loc[tr, cols] - mu) / sd, D.y_inj[tr])
    pred[te.to_numpy()] = np.clip(m.predict((D.loc[te, cols] - mu) / sd), 0, None)
D["pred"] = pred
print("MAE model", round(np.abs(D.y_inj - D.pred).mean(), 2), "vs constant", round(np.abs(D.y_inj - D.y_inj.mean()).mean(), 2), "| corr", round(np.corrcoef(D.pred, D.y_inj)[0, 1], 3))
D["tier"] = pd.qcut(D.pred.rank(pct=True), [0, .15, .35, .65, .85, 1.0], labels=False)
print(D.groupby("tier").agg(n=("y_inj", "size"), pred=("pred", "mean"), actual_days=("y_inj", "mean"), p15=("y_inj", lambda x: (x >= 15).mean()), missed_any_25=("y_gp", "mean")).round(2).to_string())
