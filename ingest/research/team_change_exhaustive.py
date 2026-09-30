"""Exhaustive deep-dive (2026-09-27, at Tommy's request): is there ANY meaningful way to predict how a team change affects a player's production,
trying variables not otherwise used in this project? Builds every candidate signal below, tests each alone, then together (linear AND gradient-
boosted, to catch non-linear/interaction effects a simple regression would miss), on real 2010-2024 rotation-to-rotation team-change transitions.

Candidates tried:
  team quality change   origin win% -> destination win% (a step up or down in competitiveness)
  pace change            destination team pace proxy - origin (box-score-derived: FGA + 0.44*FTA + TOV per team-game)
  3PA-rate change        destination 3PA/FGA - origin (style/shot-diet fit)
  positional crowd        how much OTHER production already sits at his position on the new team (specific, not team-wide)
  incumbent star at his position   the single biggest same-position player already there (is there a clear starter in his way)
  role signal (old)       was he a top-2-minutes player on his old team (a lead option, vs a role piece)
  trade vs likely FA/waiver   proxy: did another rotation player move the OPPOSITE direction between the same two teams that offseason (a real trade)
  coach change (destination)   did the team he's joining also change head coach
  coach change (origin)   did HIS old team change coach (he may have been let go by a new regime)
  health heading in       was he significantly hurt (health-confirmed) in his last season before the move (2022+ sub-sample only)
  salary/contract signal   got a raise vs pay cut on the move (2000-2019 sub-sample only, nba_salaries.csv coverage)
  open production (already tested) kept as a baseline comparison feature
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
import form_common as F
import injury_common as C
import team_context as TC
import usage_flow as UF

sys.stdout.reconfigure(encoding="utf-8")
D = C.D

# ---------------- base per-player-season panel (games.pkl; already built for every earlier study in this thread) ----------------
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
start_team = per.set_index(["pid", "season"]).team_first.to_dict()
per["team_id_next"] = [start_team.get((pid, nxt.get(season)), np.nan) for pid, season in zip(per.pid, per.season)]
per = per.rename(columns={"team_last": "team_id", "gp": "GP"})
_all_teams = sorted(set(per.team_id.dropna()) | set(per.team_first.dropna()) | set(per.team_id_next.dropna()))
_t2i = {t: i for i, t in enumerate(_all_teams)}
per["team_num"] = per.team_id.map(_t2i)
per["team_first_num"] = per.team_first.map(_t2i)
per["team_next_num"] = per.team_id_next.map(_t2i)
GOOD_MASK = (per.GP >= 40) & (per.mpg >= 20)
GOOD = set(zip(per[GOOD_MASK].pid, per[GOOD_MASK].season))

# ---------------- team quality (win%) per season, from the tanking-study team game log ----------------
tg = pd.read_csv(D / "tank" / "team_games.csv")
from team_abbr import canon
tg["team"] = tg.TEAM_ABBREVIATION.map(canon)
tg["win"] = (tg.WL == "W").astype(int)
WP = tg.groupby(["season", "team"]).win.mean().to_dict()

# ---------------- team pace / 3PA-rate proxy, built from the SAME game logs (team totals per game, no extra network calls) ----------------
tg_box = g.groupby(["team", "season", "gid"]).agg(fga=("fga", "sum"), fg3a=("fg3a", "sum"), fta=("fta", "sum"), tov=("tov", "sum")).reset_index()
tg_box["pace_proxy"] = tg_box.fga + 0.44 * tg_box.fta + tg_box.tov
style = tg_box.groupby(["team", "season"]).agg(pace=("pace_proxy", "mean"), tpa=("fg3a", "mean"), fga=("fga", "mean")).reset_index()
style["tpa_rate"] = style.tpa / style.fga
STYLE = style.set_index(["team", "season"])[["pace", "tpa_rate"]].to_dict("index")

# ---------------- coaches (team_coaches.csv: NBA numeric team_id, yr = season-start year) ----------------
from nba_api.stats.static import teams as _teams_static
TEAM_ABBR = {t["id"]: t["abbreviation"] for t in _teams_static.get_teams()}
coaches = pd.read_csv(D / "team_coaches.csv")
coaches = coaches[coaches.coach_type == "Head Coach"].copy()
coaches["team_abbr"] = coaches.team_id.map(TEAM_ABBR)
coaches["yr"] = coaches.yr.astype(int)
HEAD = coaches.groupby(["team_abbr", "yr"]).coach.first().to_dict()


def coach_changed(team, season):
    yr = int(season[:4])
    a, b = HEAD.get((team, yr - 1)), HEAD.get((team, yr))
    if a is None or b is None:
        return np.nan
    return int(a != b)


# ---------------- injury health-confirmed (2022+; reused from the earlier retests) ----------------
r5 = pd.read_pickle(D / "injury_rows_05pm.pkl")
h = r5[r5.status.isin(["Out", "Doubtful"]) & r5.kind.isin(["injury", "illness", "mgmt"])].drop_duplicates(["player_key", "game_date"]).copy()
lg = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "PLAYER_NAME"]).drop_duplicates()
lg["player_key"] = lg.PLAYER_NAME.map(C.key_of_log)
k2id = lg.drop_duplicates("player_key").set_index("player_key").PLAYER_ID.to_dict()
h["pid"] = h.player_key.map(k2id)
h = h.dropna(subset=["pid"]).copy()
h["pid"] = h.pid.astype(int)
h["yr"] = h.season.str[:4].astype(int)
HM = h.groupby(["pid", "yr"]).size().to_dict()

# ---------------- salaries (2000-2019 coverage only) ----------------
sal = pd.read_csv(D / "nba_salaries.csv")
sal.columns = [c.strip() for c in sal.columns]
import re
import unicodedata


def nkey(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


sal["nkey"] = sal.name.map(nkey)
sal["SEASON_YEAR"] = sal.season - 1
SAL = sal.groupby(["nkey", "SEASON_YEAR"]).salary.max().to_dict()
names_by_pid = lg.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID").PLAYER_NAME.to_dict()
NAME_KEY = {pid: nkey(nm) for pid, nm in names_by_pid.items()}

print(f"data ready: {len(WP)} team-seasons w/ record, {len(STYLE)} w/ style, {len(HEAD)} w/ coach, {len(SAL)} salary rows, health data from 2022+")

# ==================== assemble the transition panel ====================
rows = []
for pid, season in GOOD:
    yr_next = nxt.get(season)
    if yr_next is None or (pid, yr_next) not in GOOD:
        continue
    r0 = per[(per.pid == pid) & (per.season == season)].iloc[0]
    r1 = per[(per.pid == pid) & (per.season == yr_next)].iloc[0]
    origin, dest = r0.team_id, r1.team_first
    changed = int(origin != dest)
    if not changed:
        continue                                                   # this study is about team-changers specifically
    # team quality
    wp0, wp1 = WP.get((season, origin)), WP.get((yr_next, dest))
    # style
    s0, s1 = STYLE.get((origin, season)), STYLE.get((dest, yr_next))
    # role on old team: rank by fpg among his own team's rotation players that season
    old_team_mates = per[(per.season == season) & (per.team_id == origin) & (per.mpg >= 15)]
    old_rank = int((old_team_mates.fpg > r0.fpg).sum()) + 1
    # positional crowd + incumbent star at destination (via usage_flow's position classifier)
    dest_mates = per[(per.season == season) & (per.team_id_next == dest) & (per.team_id != dest) & (per.mpg >= 15)]  # OTHER arrivals, exclude him -- returning below
    dest_returning = per[(per.season == season) & (per.team_id == dest) & (per.team_id_next == dest) & (per.mpg >= 15)]
    dest_roster = pd.concat([dest_mates, dest_returning])
    reb36, ast36, blk36, stl36, tp36 = (r0.reb / r0.mpg * 36, r0.ast / r0.mpg * 36, r0.blk / r0.mpg * 36, r0.stl / r0.mpg * 36, r0.fg3m / r0.mpg * 36) if r0.mpg > 0 else (0,) * 5
    his_pos = UF.pos_probs(reb36, ast36, blk36, stl36, tp36)
    crowd, best_rival = 0.0, 0.0
    for _, m in dest_roster.iterrows():
        if m.mpg <= 0:
            continue
        mreb36, mast36, mblk36, mstl36, mtp36 = m.reb / m.mpg * 36, m.ast / m.mpg * 36, m.blk / m.mpg * 36, m.stl / m.mpg * 36, m.fg3m / m.mpg * 36
        mpos = UF.pos_probs(mreb36, mast36, mblk36, mstl36, mtp36)
        sim = sum(a * b for a, b in zip(his_pos, mpos))
        crowd += sim * m.fpg
        best_rival = max(best_rival, sim * m.fpg)
    # trade vs likely FA/waiver: another rotation player moved dest->origin the same offseason?
    reciprocal = int(((per.season == season) & (per.team_id == dest) & (per.team_id_next == origin) & (per.mpg >= 12)).any())
    # coach continuity
    cc_dest, cc_origin = coach_changed(dest, yr_next), coach_changed(origin, season)
    # health
    yr = int(season[:4])
    hm = HM.get((pid, yr), 0) if yr >= 2022 else np.nan
    # salary raise (2000-2019 only)
    sy = int(season[:4])
    s_prior, s_next = SAL.get((NAME_KEY.get(pid), sy - 1)), SAL.get((NAME_KEY.get(pid), sy))
    raise_pct = (s_next / s_prior - 1) if (s_prior and s_next and s_prior > 0) else np.nan
    rows.append(dict(
        pid=pid, season=season, rate0=r0.rate, rate1=r1.rate, mpg0=r0.mpg, age_proxy=season,
        wp_change=(wp1 - wp0) if (wp0 is not None and wp1 is not None) else np.nan,
        pace_change=(s1["pace"] - s0["pace"]) if (s0 and s1) else np.nan,
        tpa_change=(s1["tpa_rate"] - s0["tpa_rate"]) if (s0 and s1) else np.nan,
        old_rank=old_rank, crowd=crowd, best_rival=best_rival,
        reciprocal=reciprocal, cc_dest=cc_dest, cc_origin=cc_origin, hm=hm, raise_pct=raise_pct,
    ))
P = pd.DataFrame(rows)
print(f"\n{len(P)} veteran team-change transitions assembled")
for c in ["wp_change", "pace_change", "tpa_change", "old_rank", "crowd", "best_rival", "reciprocal", "cc_dest", "cc_origin", "hm", "raise_pct"]:
    print(f"  {c:12s} coverage {P[c].notna().mean():.0%}")

P["dchange"] = P.rate1 - P.rate0


def corr(col):
    sub = P.dropna(subset=[col, "dchange"])
    if len(sub) < 20:
        return None, len(sub)
    return sub[col].corr(sub.dchange), len(sub)


print("\n=== each candidate alone: correlation with the rate CHANGE (rate1 - rate0) ===")
for c in ["wp_change", "pace_change", "tpa_change", "old_rank", "crowd", "best_rival", "reciprocal", "cc_dest", "cc_origin", "hm", "raise_pct"]:
    r, n = corr(c)
    print(f"  {c:12s} r={r:+.3f}" if r is not None else f"  {c:12s} (n={n}, too few)", f" n={n}")

print("\n=== each candidate, controlling for his own prior rate (residual correlation -- isolates the feature from 'he was already declining/improving') ===")
mu0, sd0 = P[["rate0"]].mean(), P[["rate0"]].std()
base = Ridge(alpha=1.0).fit((P[["rate0"]] - mu0) / sd0, P.rate1)
P["res0"] = P.rate1 - base.predict((P[["rate0"]] - mu0) / sd0)
for c in ["wp_change", "pace_change", "tpa_change", "old_rank", "crowd", "best_rival", "reciprocal", "cc_dest", "cc_origin", "hm", "raise_pct"]:
    sub = P.dropna(subset=[c, "res0"])
    if len(sub) < 20:
        continue
    print(f"  {c:12s} r={sub[c].corr(sub.res0):+.3f}  n={len(sub)}")

print("\n=== combined models (leave-one-season-out), full-coverage features only (n stays at 460) ===")
FULL_COLS = ["rate0", "mpg0", "wp_change", "pace_change", "tpa_change", "old_rank", "crowd", "best_rival", "reciprocal"]
PF = P.dropna(subset=FULL_COLS + ["cc_dest", "cc_origin"]).copy()
PF["cc_dest"] = PF.cc_dest.fillna(0)
PF["cc_origin"] = PF.cc_origin.fillna(0)
COLS = FULL_COLS + ["cc_dest", "cc_origin"]
print(f"n={len(PF)} with every full-coverage feature present")


def loso_eval(df, cols, model_fn):
    errs = []
    for s in df.season.unique():
        tr, te = df[df.season != s], df[df.season == s]
        if len(te) < 8:
            continue
        m = model_fn()
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        m.fit((tr[cols] - mu) / sd, tr.rate1)
        pred = m.predict((te[cols] - mu) / sd)
        errs.append(te.rate1.values - pred)
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2)))


print(f"  rate0 only:                          RMSE {loso_eval(PF, ['rate0'], lambda: Ridge(alpha=1.0)):.4f}")
print(f"  rate0 + mpg0:                         RMSE {loso_eval(PF, ['rate0', 'mpg0'], lambda: Ridge(alpha=1.0)):.4f}")
print(f"  + every candidate above (ridge):      RMSE {loso_eval(PF, COLS, lambda: Ridge(alpha=3.0)):.4f}")
print(f"  + every candidate (gradient boosted): RMSE {loso_eval(PF, COLS, lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=25, l2_regularization=1.0)):.4f}")

# feature importance from a full-data GBM fit (not held out -- just to see which candidates the model actually leans on)
gbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=25, l2_regularization=1.0).fit(PF[COLS], PF.rate1)
from sklearn.inspection import permutation_importance
pi = permutation_importance(gbm, PF[COLS], PF.rate1, n_repeats=20, random_state=0)
print("\npermutation importance (higher = GBM leans on it more):")
for c, imp in sorted(zip(COLS, pi.importances_mean), key=lambda x: -x[1]):
    print(f"  {c:12s} {imp:+.5f}")

print("\n=== sub-analyses on the reduced-coverage features ===")
subH = P.dropna(subset=["hm", "res0"])
print(f"health (2022+, n={len(subH)}): hurt (1+ health-confirmed game missed) vs not, mean residual after rate0:")
for grp, sub in subH.assign(hurt=(subH.hm > 0).astype(int)).groupby("hurt"):
    print(f"  hurt={grp}  n={len(sub)}  mean res {sub.res0.mean():+.4f}")
subS = P.dropna(subset=["raise_pct", "res0"])
print(f"\nsalary change (2000-2019, n={len(subS)}): correlation of raise_pct with residual: {subS.raise_pct.corr(subS.res0):+.3f}" if len(subS) > 20 else "too few salary-matched rows")
if len(subS) > 20:
    for grp, sub in subS.assign(tier=pd.cut(subS.raise_pct, [-1, -0.1, 0.1, 1, 100], labels=["pay cut", "flat", "modest raise", "big raise"])).groupby("tier", observed=True):
        print(f"  {grp:14s} n={len(sub):3d}  mean residual {sub.res0.mean():+.4f}")

print("\n=== does 'traded' (reciprocal move found) vs likely FA/waiver differ? ===")
for grp, sub in P.dropna(subset=["res0"]).groupby("reciprocal"):
    print(f"  reciprocal(traded-signal)={grp}  n={len(sub)}  mean residual {sub.res0.mean():+.4f}")

print("\n=== interaction check: age at the transition -- do young players handle a team change better/worse? ===")
ages = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "AGE"]).drop_duplicates(["PLAYER_ID", "SEASON"])
AGE_AT = ages.set_index(["PLAYER_ID", "SEASON"]).AGE.to_dict()
P["age0"] = [AGE_AT.get((pid, season)) for pid, season in zip(P.pid, P.season)]
print(f"age coverage: {P.age0.notna().mean():.0%}; distribution -- <=24: {(P.age0<=24).sum()}, 25-29: {((P.age0>24)&(P.age0<=29)).sum()}, 30+: {(P.age0>29).sum()}")
for lo, hi, lbl in [(0, 24, "<=24"), (24, 29, "25-29"), (29, 99, "30+")]:
    sub = P[(P.age0 > lo) & (P.age0 <= hi)]
    subr = sub.dropna(subset=["res0"])
    print(f"  age {lbl:6s} n={len(sub):3d}  mean rate CHANGE {sub.dchange.mean():+.4f}  |  mean residual after rate0 {subr.res0.mean():+.4f}")
print("does best_rival / wp_change matter more for young team-changers specifically?")
for lbl, sub in (("<=26", P[P.age0 <= 26]), (">26", P[P.age0 > 26])):
    subm = sub.dropna(subset=["res0", "best_rival"])
    wpm = sub.dropna(subset=["res0", "wp_change"])
    wp_corr = wpm.wp_change.corr(wpm.res0) if len(wpm) > 10 else float("nan")
    print(f"  age {lbl:5s} n={len(subm):3d}  corr(best_rival, residual)={subm.best_rival.corr(subm.res0):+.3f}  corr(wp_change, residual)={wp_corr:+.3f}")
