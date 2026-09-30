"""Preliminary research: how has "team motivation" (playoff race position at season's end) affected star/rotation players' minutes, games played
and fantasy points down the stretch, and did that change after the 2019 lottery reform (flattened top-3 odds)? Baseline for judging whether teams
behave differently under the 2026-27 rules (real punishment for finishing bottom-3 -- less incentive to be there; the flattening already cut most
of the "race to the bottom for the #1 pick" incentive in 2019, but the whole bottom tier still had reason to rest players and lose games on purpose).

Method: for every team-season 2010-11..2025-26, rank teams 1 (worst) to 30 (best) by final win%. For each of a team's rotation players (>=20 mpg on
the season, >=40 games), compare his last-20-games average (minutes, fantasy points, DNP rate) to his first-(N-20)-games average the same season.
Group by final-rank bucket (1-3 / 4-8 / 9-16 / 17-30) and by era (2010-11..2018-19 pre-reform vs 2019-20..2025-26 post-reform; 2011-12 lockout and
2019-20/2020-21 covid-disrupted seasons are flagged and checked separately, not folded into the headline numbers).
Caveat raised by Tommy: a team that has traded away its own future first-round pick has no draft-lottery incentive to lose, whatever its standing --
tag those team-seasons (pick_traded_teams.py once a source is found) and exclude/compare them once that data exists. Not done in this pass.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F

D = Path(__file__).resolve().parent / "data" / "tank"
LAST_N = 20
MIN_MPG, MIN_GP = 20.0, 40

# ---------------- team standings: final rank (1=worst) per team-season
tg = pd.read_csv(D / "team_games.csv")
tg["GAME_DATE"] = pd.to_datetime(tg.GAME_DATE)
tg["win"] = (tg.WL == "W").astype(int)
final = tg.groupby(["season", "TEAM_ABBREVIATION"]).agg(w=("win", "sum"), gp=("win", "size")).reset_index()
final["wp"] = final.w / final.gp
final["rank"] = final.groupby("season")["wp"].rank(method="first").astype(int)      # 1 = worst record that season
from team_abbr import canon
final["team"] = final.TEAM_ABBREVIATION.map(canon)
RANKMAP = final.set_index(["season", "team"])["rank"].to_dict()
GP_SEASON = final.set_index(["season", "team"])["gp"].to_dict()

# ---------------- player games (already built for the form study: minutes, fantasy points, per game)
g = pd.read_pickle(F.D.parent / "form" / "games.pkl")
g = g[g.season != "2025-26"].copy()          # in progress; not a full season yet
g["fp"] = F.fp(g)
rows = []
for (season, pid, team), pl in g.groupby(["season", "pid", "team"], sort=False):
    pl = pl.sort_values("date")
    n = len(pl)
    if n < MIN_GP or pl["min"].mean() < MIN_MPG:
        continue
    rk = RANKMAP.get((season, team))
    gp_season = GP_SEASON.get((season, team))
    if rk is None or gp_season is None or n < LAST_N + 10:
        continue
    first, last = pl.iloc[: n - LAST_N], pl.iloc[n - LAST_N :]
    rows.append(dict(season=season, team=team, name=pl.name.iloc[0], rank=rk, gp_team=gp_season,
                      m_first=first["min"].mean(), m_last=last["min"].mean(), fp_first=first.fp.mean(), fp_last=last.fp.mean(),
                      dnp_first=0.0, dnp_last=0.0))     # dnp handled separately below (needs the team schedule, not just games played)
P = pd.DataFrame(rows)
P["era"] = np.select([P.season.isin(["2011-12"]), P.season.isin(["2019-20", "2020-21"]), P.season <= "2018-19"], ["lockout", "covid", "pre2019"], default="post2019")
P["bucket"] = pd.cut(P["rank"], [0, 3, 8, 16, 30], labels=["1-3 (bottom)", "4-8", "9-16", "17-30 (top)"])
P["dm"] = P.m_last - P.m_first
P["dfp"] = P.fp_last - P.fp_first
P["dm_pct"] = P.dm / P.m_first
P["dfp_pct"] = P.dfp / P.fp_first
print(f"{len(P)} player-seasons (rotation players, >=20mpg & >=40gp), {P.season.nunique()} seasons")

# ---------------- headline: late-season minutes/points change by bucket, pre- vs post-2019 (clean seasons only)
clean = P[P.era.isin(["pre2019", "post2019"])]
print("\n=== Minutes change, last 20 games vs the rest of the season (rotation players) ===")
print(clean.groupby(["era", "bucket"], observed=True).agg(n=("dm", "size"), d_min=("dm", "mean"), d_min_pct=("dm_pct", "mean"), d_fp=("dfp", "mean"), d_fp_pct=("dfp_pct", "mean")).round(3))

# ---------------- availability: DNP rate in the last 20 team games vs the rest, using the team schedule (not just games he appeared in)
print("\n=== Availability (share of TEAM's remaining games he plays), last 20 vs earlier ===")
avail_rows = []
gp_key = g.set_index(["season", "team", "pid"])
tg2 = tg.rename(columns={"TEAM_ABBREVIATION": "team_raw"}).copy()
tg2["team"] = tg2.team_raw.map(canon)
for (season, pid, team), pl in g.groupby(["season", "pid", "team"], sort=False):
    n = len(pl)
    if n < MIN_GP or pl["min"].mean() < MIN_MPG:
        continue
    rk = RANKMAP.get((season, team))
    if rk is None:
        continue
    tdays = tg2[(tg2.season == season) & (tg2.team == team)].sort_values("GAME_DATE")
    if len(tdays) < LAST_N + 10:
        continue
    played_dates = set(pl.date.dt.normalize())
    tdays = tdays.assign(played=tdays.GAME_DATE.isin(played_dates).astype(int))
    first_t, last_t = tdays.iloc[: len(tdays) - LAST_N], tdays.iloc[len(tdays) - LAST_N :]
    avail_rows.append(dict(season=season, team=team, rank=rk, avail_first=first_t.played.mean(), avail_last=last_t.played.mean()))
A = pd.DataFrame(avail_rows)
A["era"] = np.select([A.season.isin(["2011-12"]), A.season.isin(["2019-20", "2020-21"]), A.season <= "2018-19"], ["lockout", "covid", "pre2019"], default="post2019")
A["bucket"] = pd.cut(A["rank"], [0, 3, 8, 16, 30], labels=["1-3 (bottom)", "4-8", "9-16", "17-30 (top)"])
A["d_avail"] = A.avail_last - A.avail_first
cleanA = A[A.era.isin(["pre2019", "post2019"])]
print(cleanA.groupby(["era", "bucket"], observed=True).agg(n=("d_avail", "size"), avail_first=("avail_first", "mean"), avail_last=("avail_last", "mean"), d_avail=("d_avail", "mean")).round(3))

# ---------------- robustness: covid/lockout seasons on their own
print("\n=== covid/lockout seasons (not in the headline numbers) ===")
odd = P[P.era.isin(["covid", "lockout"])]
print(odd.groupby(["era", "bucket"], observed=True).agg(n=("dm", "size"), d_min=("dm", "mean"), d_fp=("dfp", "mean")).round(3))

# ---------------- year-by-year trend for the bottom-3 bucket specifically (the group whose #1-pick odds were flattened in 2019)
print("\n=== Bottom-3 teams, minutes change by season (trend around the 2019 reform) ===")
b3 = P[P["rank"] <= 3]
print(b3.groupby("season", observed=True).agg(n=("dm", "size"), d_min=("dm", "mean"), d_min_pct=("dm_pct", "mean")).round(3))

P.to_csv(D / "tank_player_panel.csv", index=False)
A.to_csv(D / "tank_availability_panel.csv", index=False)
print(f"\nwrote {D / 'tank_player_panel.csv'} and tank_availability_panel.csv")
