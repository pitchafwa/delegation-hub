"""Step 2 (2026-09-27, at Tommy's request): does the rookie "open production" measure (team_context.py) close the accuracy gap found for
VETERAN team-changers (team_change_test.py)? And does the injury usage-flow model (usage_flow.py) generalize to normal offseason roster churn
(trades/free agency/retirement), not just in-season injuries, for the TEAMMATES who stay behind?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F
import team_context as TC
import usage_flow as UF

sys.stdout.reconfigure(encoding="utf-8")
g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
g = g[g.season != "2025-26"].copy()
g["fp"] = F.fp(g)
per = g.groupby(["pid", "season"]).agg(
    min_sum=("min", "sum"), fp_sum=("fp", "sum"), gp=("min", "size"), mpg=("min", "mean"),
    reb=("reb", "mean"), ast=("ast", "mean"), blk=("blk", "mean"), stl=("stl", "mean"), fg3m=("fg3m", "mean"),
    team_last=("team", "last"), team_first=("team", "first")).reset_index()
per["fpg"] = per.fp_sum / per.gp
per["rate"] = per.fp_sum / per.min_sum
seasons = sorted(per.season.unique())
nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
per["season_next"] = per.season.map(nxt)
team_next = per.set_index(["pid", "season_next"]).team_first if False else None
# each player's team at the START of next season (his team_first row in that season)
start_team = per.set_index(["pid", "season"]).team_first.to_dict()
per["team_id_next"] = [start_team.get((pid, nxt.get(season)), np.nan) for pid, season in zip(per.pid, per.season)]
per = per.rename(columns={"team_last": "team_id", "gp": "GP"})
_all_teams = sorted(set(per.team_id.dropna()) | set(per.team_first.dropna()) | set(per.team_id_next.dropna()))
_t2i = {t: i for i, t in enumerate(_all_teams)}
per["team_id"] = per.team_id.map(_t2i)
per["team_id_next"] = per.team_id_next.map(_t2i)
per["team_first_id"] = per.team_first.map(_t2i)
GOOD = set(zip(per[(per.GP >= 40) & (per.mpg >= 20)].pid, per[(per.GP >= 40) & (per.mpg >= 20)].season))

# ==================== PART 1: does "open production at the destination" explain the team-changer gap? ====================
print("=" * 70)
print("PART 1: open production at the destination, for veterans who changed teams")
print("=" * 70)
rows = []
for yr_prev in seasons[:-1]:
    ctx = TC.team_context(TC.prep_panel(per[per.season == yr_prev].assign(yr=yr_prev)), yr_prev, "team_id_next")
    yr_next = nxt[yr_prev]
    for pid, season in [(p, s) for p, s in GOOD if s == yr_prev]:
        if (pid, yr_next) not in GOOD:
            continue
        r0 = per[(per.pid == pid) & (per.season == yr_prev)].iloc[0]
        r1 = per[(per.pid == pid) & (per.season == yr_next)].iloc[0]
        changed = int(r0.team_id != r1.team_first_id)
        dest = _t2i.get(r1.team_first)
        open_fp = ctx["open_fp"].get(dest, np.nan)
        rows.append(dict(pid=pid, season=season, rate0=r0.rate, rate1=r1.rate, changed=changed, open_fp=open_fp))
P = pd.DataFrame(rows).dropna(subset=["open_fp"])
print(f"{len(P)} pairs ({P.changed.sum()} team changes) with a computable destination context")

mu, sd = P[["rate0"]].mean(), P[["rate0"]].std()
base = Ridge(alpha=1.0).fit((P[["rate0"]] - mu) / sd, P.rate1)
P["res_base"] = P.rate1 - base.predict((P[["rate0"]] - mu) / sd)

ch = P[P.changed == 1]
print(f"\nchanged-teams residual (rate0 only): RMSE {np.sqrt((ch.res_base**2).mean()):.4f}  mean {ch.res_base.mean():+.4f}  n={len(ch)}")
print(f"correlation of that residual with open_fp at the destination: {ch.res_base.corr(ch.open_fp):+.3f}")

mu2, sd2 = P[["rate0", "open_fp"]].mean(), P[["rate0", "open_fp"]].std()
m2 = Ridge(alpha=1.0).fit((P[["rate0", "open_fp"]] - mu2) / sd2, P.rate1)
P["res2"] = P.rate1 - m2.predict((P[["rate0", "open_fp"]] - mu2) / sd2)
ch2 = P[P.changed == 1]
print(f"with open_fp added:                    RMSE {np.sqrt((ch2.res2**2).mean()):.4f}  mean {ch2.res2.mean():+.4f}")
print(f"coefficient on open_fp: {m2.coef_[1]:+.5f} (per-team-fp-unit; positive = joining a team with more open production predicts a real rate gain)")
stayed = P[P.changed == 0]
print(f"(for context, stayed-put players' residual RMSE with the same 2-feature model: {np.sqrt(((stayed.rate1 - m2.predict((stayed[['rate0','open_fp']]-mu2)/sd2))**2).mean()):.4f})")

# ==================== PART 2: does the INJURY usage-flow model generalize to permanent departures? ====================
print("\n" + "=" * 70)
print("PART 2: does the injury usage-flow model explain teammates' gains after a normal (non-injury) departure?")
print("=" * 70)
rows2 = []
for yr_prev in seasons[:-1]:
    yr_next = nxt[yr_prev]
    cur = per[per.season == yr_prev]
    for team, grp in cur.groupby("team_id"):  # numeric team id now
        roster = grp[grp.mpg >= 6].copy()
        if len(roster) < 5:
            continue
        players = []
        for _, r in roster.iterrows():
            reb36, ast36, blk36, stl36, tp36 = (r.reb / r.mpg * 36, r.ast / r.mpg * 36, r.blk / r.mpg * 36, r.stl / r.mpg * 36, r.fg3m / r.mpg * 36) if r.mpg > 0 else (0, 0, 0, 0, 0)
            pos = UF.pos_probs(reb36, ast36, blk36, stl36, tp36)
            departed = r.team_id_next != team and r.mpg >= 12
            players.append(dict(id=r.pid, fp=max(r.fpg, 0.5), mpg=r.mpg, pos=pos, p_out=(1.0 if departed else 0.0), w=1.0, _stayed=(r.team_id_next == team)))
        if not any(p["p_out"] > 0 for p in players):
            continue
        up = UF.uplifts(players)
        for p in players:
            if p["p_out"] > 0 or not p["_stayed"]:
                continue
            nxt_row = per[(per.pid == p["id"]) & (per.season == yr_next)]
            if len(nxt_row) == 0:
                continue
            rows2.append(dict(pid=p["id"], season=yr_prev, team=team, fp_prior=p["fp"], fp_next=nxt_row.iloc[0].fpg, pred=up.get(p["id"], 0.0)))
D = pd.DataFrame(rows2)
D["actual"] = D.fp_next - D.fp_prior
print(f"{len(D)} returning-teammate observations across seasons with a rotation-caliber (12+ mpg) departure on their team")
print(f"correlation(predicted uplift, actual next-season change): {D['pred'].corr(D['actual']):+.3f}")
for lo, hi in [(-0.01, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 20)]:
    sub = D[(D.pred >= lo) & (D.pred < hi)]
    if len(sub) < 10:
        continue
    print(f"  predicted uplift {lo:.1f}-{hi:.1f}: n={len(sub):4d}  mean actual change {sub.actual.mean():+.3f}  (mean predicted {sub.pred.mean():.3f})")
zero = D[D.pred < 0.05]
nonzero = D[D.pred >= 0.05]
print(f"\nno predicted uplift (n={len(zero)}): mean actual change {zero.actual.mean():+.3f}")
print(f"some predicted uplift (n={len(nonzero)}): mean actual change {nonzero.actual.mean():+.3f}")

mu3, sd3 = D[["pred"]].mean(), D[["pred"]].std()
m3 = Ridge(alpha=1.0).fit((D[["pred"]] - mu3) / sd3, D.actual)
print(f"regression slope of actual on predicted: {(m3.coef_[0] / sd3.iloc[0]):.3f} fp/game per predicted fp/game (1.0 = the injury model's size transfers exactly; the ORIGINAL in-season injury effect calibrated to about 0.70 of the raw estimate -- see usage_flow.py's kappa/lam)")
