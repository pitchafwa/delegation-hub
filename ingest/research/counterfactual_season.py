"""Counterfactual: what if one team had followed the planner's lineups and suggested adds every week of 2025-26, with every other team scoring exactly what it really scored?
Per week (independent weeks, no compounding of roster changes):
  lineups   the cap-aware planner sets each day's lineup from information known that morning, on the team's real roster that day (backtest_weekly_planner.py part 1)
  adds      the planner's suggested add/drop moves for the week, valued with real box scores (part 2, same rules and the 0.6 free-agent shrink)
  weeks the backtest skips (1-day-cap oddities: week 17, playoffs) keep the team's real score
Opponents' scores are their REAL scores that week; results are re-derived from the real pairings. Standings and playoff seeding are recomputed from those results.
Run from ingest/:  uv run python research/counterfactual_season.py [team_id]      (default 12 = DRNK)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
D = R / "data"
TEAM = int(sys.argv[1]) if len(sys.argv) > 1 else 12
lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mSettings", "mTeam"]})
teams = {t["id"]: (t.get("abbrev") or t.get("name")) for t in raw["teams"]}
sett = raw["settings"]
print("regular season matchups:", sett["scheduleSettings"].get("matchupPeriodCount"), "| playoff teams:", sett["scheduleSettings"].get("playoffTeamCount"), "| playoff matchup length:", sett["scheduleSettings"].get("playoffMatchupPeriodLength"))
sched = []
for m in raw["schedule"]:
    if "away" not in m or "home" not in m:
        continue
    sched.append(dict(mp=m["matchupPeriodId"], home=m["home"]["teamId"], away=m["away"]["teamId"], hp=m["home"].get("totalPoints"), ap=m["away"].get("totalPoints"), tier=m.get("playoffTierType")))
S = pd.DataFrame(sched)
reg = S[S.tier.isin(["NONE"]) | S.tier.isna()] if S.tier.notna().any() else S
print(S.groupby("tier", dropna=False).mp.agg(["min", "max", "size"]).to_string())
lin = pd.read_csv(D / "backtest_lineups_2026.csv")
adds = pd.read_csv(D / f"backtest_adds_2026_team{TEAM}.csv")
me = lin[lin.team == TEAM].set_index("mp")
ad = adds.set_index("mp")
real = {}
for r in S.itertuples():
    real[(r.mp, r.home)] = r.hp
    real[(r.mp, r.away)] = r.ap
rows = []
for mp in sorted(set(S[S.home.eq(TEAM) | S.away.eq(TEAM)].mp)):
    m = S[(S.mp == mp) & (S.home.eq(TEAM) | S.away.eq(TEAM))].iloc[0]
    opp = m.away if m.home == TEAM else m.home
    actual = real[(mp, TEAM)]
    opp_pts = real[(mp, opp)]
    cf = actual
    lin_pts = add_gain = 0.0
    if mp in me.index:
        # part 1 measured the planner on the team's real daily rosters; the gain over the real lineups is planner - actual
        lin_gain = float(me.loc[mp, "planner"] - me.loc[mp, "actual"])
        cf = actual + lin_gain
        lin_pts = lin_gain
    if mp in ad.index and ad.loc[mp, "moves"] > 0:
        add_gain = float(ad.loc[mp, "realized"])
        cf += add_gain
    rows.append(dict(mp=mp, opp=teams[opp], actual=actual, opp_actual=opp_pts, lineup_gain=lin_pts, add_gain=add_gain, counterfactual=cf, tier=m.tier))
W = pd.DataFrame(rows)
W["real_result"] = np.where(W.actual > W.opp_actual, "W", np.where(W.actual < W.opp_actual, "L", "T"))
W["cf_result"] = np.where(W.counterfactual > W.opp_actual, "W", np.where(W.counterfactual < W.opp_actual, "L", "T"))
W.to_csv(D / f"counterfactual_team{TEAM}.csv", index=False)
print(f"\nTEAM {teams[TEAM]}")
reg = W[W.tier.isin(["NONE"]) | W.tier.isna()]
print(reg.round(1).to_string(index=False))
def rec(x, col):
    w = (x[col] == "W").sum(); l = (x[col] == "L").sum(); t = (x[col] == "T").sum()
    return f"{w}-{l}" + (f"-{t}" if t else "")
print(f"\nregular season record: real {rec(reg, 'real_result')} -> counterfactual {rec(reg, 'cf_result')}")
print(f"mean points per week: real {reg.actual.mean():.0f} -> counterfactual {reg.counterfactual.mean():.0f}  (lineups {reg.lineup_gain.mean():+.0f}, adds {reg.add_gain.mean():+.0f})")
# league standings recomputed: everyone else keeps real results EXCEPT the games against this team
stand = {t: [0, 0, 0.0] for t in teams}          # wins, losses, points for
stand_cf = {t: [0, 0, 0.0] for t in teams}
for r in S.itertuples():
    if not (r.tier in ("NONE",) or pd.isna(r.tier)):
        continue
    for name, table in (("real", stand), ("cf", stand_cf)):
        hp, ap = r.hp, r.ap
        if name == "cf":
            cfp = W.set_index("mp").counterfactual
            if r.home == TEAM and r.mp in cfp.index: hp = cfp[r.mp]
            if r.away == TEAM and r.mp in cfp.index: ap = cfp[r.mp]
        if hp is None or ap is None: continue
        table[r.home][2] += hp; table[r.away][2] += ap
        if hp > ap: table[r.home][0] += 1; table[r.away][1] += 1
        elif hp < ap: table[r.home][1] += 1; table[r.away][0] += 1
def show(table, title):
    df = pd.DataFrame({teams[t]: v for t, v in table.items()}, index=["W", "L", "PF"]).T.sort_values(["W", "PF"], ascending=False)
    df["rank"] = range(1, len(df) + 1)
    print(f"\n{title}"); print(df.round(0).astype({"W": int, "L": int}).to_string())
    return df
a = show(stand, "REAL regular-season standings (wins, then points for)")
b = show(stand_cf, "COUNTERFACTUAL standings (this team follows the plan; everyone else scores exactly what they scored)")
