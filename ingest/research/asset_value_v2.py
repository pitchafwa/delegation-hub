"""Keeper-count-aware ASSET VALUE, v2: empirical forecast + survival + real outcome spread.

For a player x and horizon h (seasons out):
   A_h(c) = P(still a rotation player | x) * E[ (pts/g_h - c)+ | present ]
 P(present)   logistic on age / production / minutes / pedigree (from real career outcomes, incl. retirements)
 pts/g | present   ridge forecast + the EMPIRICAL residual spread for that age group (keeps the skew, so
                   young high-ceiling players carry real upside instead of a symmetric bell)
   V(K) = sum_{h=1..6} delta^(h-1) * A_h(c_h(K))
   c_1 = replacement level (you always play next season)
   c_h(K), h>=2 = keep cutoff = pts/g of the (K x 12 teams)-th best player; K=0 -> no future value.
   Keepers cost nothing in this league (12 teams), only a roster slot.
Incoming prospects: same structure keyed on draft slot, draft age and pre-NBA talent percentile (small but consistent held-out gain), outcomes = past draftees.

UNIFIED PROJECTION (one path per player, feeding BOTH VOR and Asset value; see engine_blend_test.py and
prospect_engine_test.py for the held-out tests behind the weights):
  veterans:  E_h = w*ridge_h + (1-w)*Kalman_h,  w = 0.75 for next season, 0.5 after
  prospects: E_h = (1-L)*empirical_h + L*(Output B trajectory + ceiling calibration),  L = 0.4 for picks <=10, tapering to 0 at 16+
Optional transparent scouting lever: ingest/research/prospect_overrides.json {"Name": {"equivalent_pick": n}}.

Out-of-sample test vs "current output persists", then compared with the two anchors.
Usage: python asset_value_v2.py -> data/asset_value.csv
"""
import json
import os
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
NTEAMS, H, REPL, LAST_YR, DELTA = 12, 10, 22.0, 2025, 0.95  # 10-season horizon (dynasty standard; was 6), 5%/yr discount
# how the rookie team-situation boost is used: "decay" = years 1-3 with the measured fade (default); "carry" = year 1 only, and that
# boosted year-1 level becomes the baseline every later season builds on (level shift for all years). Variant files get a _carry suffix.
CTX_MODE = os.environ.get("CTX_MODE", "decay")
SFX = "_carry" if CTX_MODE == "carry" else ""

# ------------------------------------------------------------------ vet panel with outcomes at +1..+6
panel = pd.read_csv(D / "breakout_panel_ctx2.csv")
panel["logpick"] = np.log(panel["draft_pick"].clip(1, 61))
panel["young"] = (panel["AGE"] <= 24) * (24 - panel["AGE"])
fp = panel.set_index(["PLAYER_ID", "yr"])
for h in range(1, H + 1):
    idx = pd.MultiIndex.from_arrays([panel["PLAYER_ID"], panel["yr"] + h])
    f, g, m = (fp[c].reindex(idx).to_numpy() for c in ("fpg", "GP", "mpg"))
    present = ((g >= 10) & (m >= 8)).astype(float)
    obs = panel["yr"].to_numpy() + h <= LAST_YR
    panel[f"pres{h}"] = np.where(obs, present, np.nan)
    panel[f"v{h}"] = np.where(obs & (present == 1), f, np.nan)  # pts/g given present

F = ["fpg", "AGE", "young", "mpg", "late_mpg", "late_fpg", "late_fp_per36", "USG_PCT", "PIE", "logpick", "exp", "d_fpg", "d_mpg"]
# TEAM SITUATION (held-out tests: vets +2% next-season accuracy, +4% for players who changed teams; fades after year 1):
# did he move, and how much production/minutes are vacated at his team net of arrivals
S = ["moved", "dest_net_usg", "dest_vac_min"]
FH = lambda h: F + S if h == 1 else F
POOL = panel[(panel["GP"] >= 20) & (panel["mpg"] >= 10)].copy()
AGEB = lambda a: np.digitize(a, [22.5, 25.5, 29.5])  # 4 age groups


def rg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))


def lg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


class Forecaster:
    """fit on vet rows; predict A_h(c) for any rows. `asof` masks outcomes not yet observable."""

    def __init__(self, train, asof=None, hmax=H):
        self.m, self.s, self.res = {}, {}, {}
        for h in range(1, hmax + 1):
            t = train.copy()
            if asof is not None:
                ok = t["yr"] + h <= asof
                t = t[ok]
            pres = t[t[f"pres{h}"].notna()]
            if len(pres) < 200:
                continue
            self.s[h] = lg().fit(pres[F], pres[f"pres{h}"].astype(int))
            pv = pres[pres[f"pres{h}"] == 1]
            self.m[h] = rg().fit(pv[FH(h)], pv[f"v{h}"])
            r = pv[f"v{h}"].to_numpy() - self.m[h].predict(pv[FH(h)])
            b = AGEB(pv["AGE"].to_numpy())
            self.res[h] = {g: r[b == g] for g in range(4)}

    def value(self, X, age, h, c, mu=None):
        """A_h(c) per row; c scalar; mu optionally overrides the ridge mean (used for the unified path)"""
        if h not in self.m:
            return np.zeros(len(X))
        mu = self.m[h].predict(X[FH(h)]) if mu is None else mu
        ps = self.s[h].predict_proba(X[F])[:, 1]
        b = AGEB(age)
        out = np.zeros(len(X))
        for g in range(4):
            sel = b == g
            if not sel.any():
                continue
            rr = self.res[h][g] if len(self.res[h][g]) > 30 else np.concatenate(list(self.res[h].values()))
            rr = rr[:: max(1, len(rr) // 300)]
            out[sel] = ps[sel] * np.maximum(mu[sel, None] + rr[None, :] - c, 0).mean(axis=1)
        return out


# ------------------------------------------------------------------ out-of-sample test
print("== analog-free empirical forecast vs 'current output persists'  (test base years 2015-2019, trained only on outcomes observable then)")
rows = []
for ty in (2015, 2016, 2017, 2018, 2019):
    test = POOL[POOL["yr"] == ty]
    fc = Forecaster(POOL[POOL["yr"] < ty], asof=ty, hmax=4)
    for h in (1, 2, 3, 4):
        real = np.where(np.isnan(test[f"pres{h}"]), np.nan, np.where(test[f"pres{h}"] == 1, test[f"v{h}"], 0.0))
        ok = ~np.isnan(real)
        for c in (22.0, 35.0, 45.0):
            pa = fc.value(test, test["AGE"].to_numpy(), h, c)
            pn = np.maximum(test["fpg"].to_numpy() - c, 0)
            rr = np.maximum(real - c, 0)
            rows.append(dict(h=h, c=c, rho_model=spearmanr(pa[ok], rr[ok])[0], rho_naive=spearmanr(pn[ok], rr[ok])[0],
                             rmse_model=float(np.sqrt(np.mean((pa[ok] - rr[ok]) ** 2))), rmse_naive=float(np.sqrt(np.mean((pn[ok] - rr[ok]) ** 2)))))
print(pd.DataFrame(rows).groupby(["h", "c"]).mean().round(3).to_string())

# ------------------------------------------------------------------ live vets
fc = Forecaster(POOL[POOL["yr"] < LAST_YR])
live = POOL[POOL["yr"] == LAST_YR].reset_index(drop=True)
live_age_next = live["AGE"].to_numpy() + 1

from build_breakout_context import situation_features  # noqa: E402
from team_context import prep_panel, team_context  # noqa: E402

_espn = pd.read_csv(D / "espn_adp.csv")
_espn = _espn[(_espn["season_id"] == LAST_YR + 2) & _espn["PLAYER_ID"].notna()].copy()
_espn["PLAYER_ID"] = _espn["PLAYER_ID"].astype(int)
_espn["abbr"] = _espn["pro_team"].replace({"PHL": "PHI", "PHO": "PHX", "NOR": "NOP", "NO": "NOP"})
_a2id = panel.dropna(subset=["team_id"]).drop_duplicates("team").set_index("team")["team_id"].to_dict()
_espn["team_id_now"] = _espn["abbr"].map(_a2id)
_espn = _espn.sort_values("adp").drop_duplicates("PLAYER_ID")
_lp = panel[panel["yr"] == LAST_YR].drop(columns=["team_id_next", "team_next", "moved", "dest_net_usg", "dest_vac_min", "coach_change"], errors="ignore")
_lp = _lp.merge(_espn[["PLAYER_ID", "team_id_now"]].rename(columns={"team_id_now": "team_id_next"}), on="PLAYER_ID", how="left")
_lp["totmin"] = _lp["totmin"].fillna(0)
_lp = situation_features(_lp, None)
live = live.drop(columns=S, errors="ignore").merge(_lp[["PLAYER_ID"] + S], on="PLAYER_ID", how="left")
_pp = prep_panel(panel)
_lpp = _pp[_pp["yr"] == LAST_YR].merge(_espn[["PLAYER_ID", "team_id_now"]], on="PLAYER_ID", how="left")
LIVE_CTX = team_context(_lpp, LAST_YR, "team_id_now")  # 2026-27 team situation for every team

# ------------------------------------------------------------------ prospects: draft-slot model
u = pd.read_csv(D / "rookie_model_dataset_unified.csv")[["PLAYER_ID", "player", "real_draft_year", "real_draft_number", "draft_age", "talent_pctile"]].drop_duplicates("PLAYER_ID")
sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
             - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
sb["mpg"] = sb["MIN"] / sb["GP"].replace(0, np.nan)
sf = sb.set_index(["PLAYER_ID", "yr"])
hd = u[(u["real_draft_year"] >= 2010)].copy()
hd["logpick"] = np.log(hd["real_draft_number"].fillna(61).clip(1, 61))
hd["dage"] = hd["draft_age"].fillna(hd["draft_age"].median())
hd["talent"] = hd["talent_pctile"].fillna(hd["talent_pctile"].median())
hd["is1"] = (hd["real_draft_number"] == 1).astype(float)  # #1 overall pick = its own tier (held-out validated)
for h in range(1, H + 1):  # h=1 is the rookie season
    idx = pd.MultiIndex.from_arrays([hd["PLAYER_ID"], hd["real_draft_year"].astype(int) + h - 1])
    f, g, m = (sf[c].reindex(idx).to_numpy() for c in ("fpg", "GP", "mpg"))
    present = ((g >= 10) & (m >= 8)).astype(float)
    obs = hd["real_draft_year"].to_numpy() + h - 1 <= LAST_YR
    hd[f"pres{h}"] = np.where(obs, present, np.nan)
    hd[f"v{h}"] = np.where(obs & (present == 1), f, np.nan)
_gl = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "GAME_DATE", "TEAM"])
_gl["sy"] = _gl["SEASON"].str[:4].astype(int)
_first = _gl.sort_values("GAME_DATE").groupby(["PLAYER_ID", "sy"]).first().reset_index()
_abbr_id = panel.dropna(subset=["team_id"]).drop_duplicates(["team", "yr"]).set_index(["team", "yr"])["team_id"].to_dict()
_open = {}
_open_all = []
for _cls in sorted(hd["real_draft_year"].unique()):
    _cls = int(_cls)
    if _cls - 1 < panel["yr"].min() or _cls > LAST_YR:
        continue
    _ctx = team_context(_pp, _cls - 1)
    for _r in _first[_first["sy"] == _cls].itertuples():
        _tid = _abbr_id.get((_r.TEAM, _cls))
        if _tid is not None and int(_tid) in _ctx.index:
            _open[(_r.PLAYER_ID, _cls)] = float(_ctx.loc[int(_tid), "open_fp"])
    _open_all.extend(_ctx["open_fp"].tolist())
_omu, _osd = float(np.mean(_open_all)), float(np.std(_open_all))
OPEN_CLIP = lambda x: np.clip(x, _omu - 2 * _osd, _omu + 2 * _osd)  # cap at +/-2 sd: no extrapolating past the historical range
hd["open"] = OPEN_CLIP(np.array([_open.get((pid, int(c)), np.nan) for pid, c in zip(hd["PLAYER_ID"], hd["real_draft_year"])], dtype=float))
hd["open_top"] = hd["open"] * (hd["real_draft_number"].fillna(61) <= 15)  # lottery picks get the minutes: their effect is larger
PF = ["logpick", "dage", "talent", "is1"]
PFH = lambda h: PF + ["open", "open_top"] if (h == 1 if CTX_MODE == "carry" else h <= 3) else PF  # rookie-team opportunity: +3.8/+2.5/+1.9 pts/g per sd in yrs 1/2/3, fading after (held-out gain in yrs 1-2)
pm, ps_, pres_res = {}, {}, {}
for h in range(1, H + 1):
    t = hd[hd[f"pres{h}"].notna()]
    ps_[h] = lg().fit(t[PF], t[f"pres{h}"].astype(int))
    pv = t[t[f"pres{h}"] == 1]
    pm[h] = rg().fit(pv[PFH(h)], pv[f"v{h}"])
    r = pv[f"v{h}"].to_numpy() - pm[h].predict(pv[PFH(h)])
    tier = np.digitize(np.exp(pv["logpick"].to_numpy()), [4, 11, 31])
    pres_res[h] = {g_: r[tier == g_] for g_ in range(4)}


def prospect_value(picks, ages, h, c, mu=None):
    X = pd.DataFrame({"logpick": np.log(np.clip(picks, 1, 61)), "dage": ages, "talent": pr_talent, "is1": (np.asarray(picks) == 1).astype(float), "open": pr_open, "open_top": pr_open * (np.asarray(picks) <= 15)})
    p = ps_[h].predict_proba(X[PF])[:, 1]
    mu = pm[h].predict(X[PFH(h)]) if mu is None else mu
    tier = np.digitize(picks, [4, 11, 31])
    out = np.zeros(len(X))
    for g_ in range(4):
        sel = tier == g_
        if sel.any():
            rr = pres_res[h][g_] if len(pres_res[h][g_]) > 20 else np.concatenate(list(pres_res[h].values()))
            out[sel] = p[sel] * np.maximum(mu[sel, None] + rr[None, :] - c, 0).mean(axis=1)
    return out


# bring in the hub's prospects
hub = json.loads((ROOT.parent.parent / "dashboard" / "hub_data.json").read_text(encoding="utf-8"))["players"]
pros = [q for q in hub if q["kind"] == "prospect"]
_ovr_path = ROOT / "prospect_overrides.json"
OVR = json.loads(_ovr_path.read_text(encoding="utf-8")) if _ovr_path.exists() else {}
pr_pick_actual = np.array([q["pick"] if q.get("pick") else 61 for q in pros], dtype=float)
pr_pick = np.array([float(OVR.get(q["player"], {}).get("equivalent_pick", a)) for q, a in zip(pros, pr_pick_actual)], dtype=float)
for q, a, e in zip(pros, pr_pick_actual, pr_pick):
    if a != e:
        print(f"  OVERRIDE: {q['player']} valued as pick #{e:.0f} instead of #{a:.0f} ({OVR[q['player']].get('note', '')})")
pr_age = np.array([q["age"] if q.get("age") else 20.5 for q in pros], dtype=float)
_teams = pd.read_csv(ROOT / "prospect_teams_2026.csv").set_index("player")["team"].to_dict()
pr_team = [_teams.get(q["player"]) for q in pros]
pr_open_raw = np.array([float(LIVE_CTX.loc[int(_a2id[t]), "open_fp"]) if (t in _a2id and int(_a2id[t]) in LIVE_CTX.index) else np.nan for t in pr_team], dtype=float)
pr_open = OPEN_CLIP(pr_open_raw)
pr_talent = np.array([q["talent_pctile"] if q.get("talent_pctile") is not None else hd["talent"].median() for q in pros], dtype=float)


# ------------------------------------------------------------------ UNIFIED projection paths
import ast

kt = pd.read_csv(D / "current_player_trajectories.csv")
ktm = dict(zip(kt["PLAYER_ID"], kt["trajectory"].apply(ast.literal_eval)))
kal_v = np.full((len(live), H), np.nan)
for i, pid in enumerate(live["PLAYER_ID"]):
    t = ktm.get(pid)
    if t:
        kal_v[i, : min(H, len(t))] = t[:H]
W_VET = lambda h: 0.75 if h == 1 else 0.5
E_v = np.full((len(live), H), np.nan)
for h in range(1, H + 1):
    ridge_mu = fc.m[h].predict(live[FH(h)])
    a = kal_v[:, h - 1]
    E_v[:, h - 1] = np.where(np.isnan(a), ridge_mu, W_VET(h) * ridge_mu + (1 - W_VET(h)) * a)

# SCOUTING-INFORMATION UPLIFT for young elite-pedigree players (top-5 pick, age <= 21). Calibrated, NOT a validated production effect:
# after the 10-season horizon these players still sat below BOTH independent expert sources (mean rank 44 vs Hashtag 32, RotoWire 40 at
# full dynasty); +3 pts/g every season puts the group between the two outlets at 5 keepers and at full dynasty (39 / 35; see
# elite_talent_shift_test.py). The market's own later re-ranking also moved young players up ~25 spots relative to established ones.
def smooth_tail(E, start=5):
    """each season beyond the 5th has its own fitted model, which adds a few points of season-to-season noise to the far path;
    a light 1-2-1 smoothing over seasons 6+ removes it without touching the first five seasons"""
    S = E.copy()
    for j in range(start, E.shape[1] - 1):
        S[:, j] = 0.25 * E[:, j - 1] + 0.5 * E[:, j] + 0.25 * E[:, j + 1]
    return S


E_v = smooth_tail(E_v)
ELITE_UPLIFT = float(os.environ.get("ELITE_UPLIFT", "3.0"))
elite_v = (np.round(np.exp(live["logpick"].to_numpy())) <= 5) & (live_age_next <= 21.5)
# NOTE: the scouting prior is applied ONLY when pricing (asset value / ceiling asset value), never to the projected pts/g paths,
# peaks or VOR shown on the site -- those stay the unadjusted model output.

pt = pd.read_csv(D / "prospect_trajectories.csv")
ptm = dict(zip(pt["PLAYER_ID"], pt["trajectory"].apply(ast.literal_eval)))
_cal = json.loads((D / "prospect_calibration.json").read_text(encoding="utf-8"))


def cal_traj(traj, pick):
    if pick > 15:
        return list(traj)
    taper = float(np.clip((16 - pick) / 6.0, 0.0, 1.0))
    out = []
    for k, v in enumerate(traj):
        a, b, c = _cal["coef"][str(min(k, _cal["max_k"]))]
        out.append(max(v + taper * (a + b * float(np.log(pick)) + c * float(pick == 1)), 0.0))
    return out


A_p = np.full((len(pros), H), np.nan)
for i, (q, pk) in enumerate(zip(pros, pr_pick)):
    t = ptm.get(int(q["id"][1:]))
    if t:
        A_p[i, : min(H, len(t))] = cal_traj(t[:H], pk)
Xp = pd.DataFrame({"logpick": np.log(np.clip(pr_pick, 1, 61)), "dage": pr_age, "talent": pr_talent, "is1": (pr_pick == 1).astype(float), "open": pr_open, "open_top": pr_open * (pr_pick <= 15)})
lam = 0.4 * np.clip((16 - pr_pick) / 6.0, 0.0, 1.0)
E_p = np.full((len(pros), H), np.nan)
for h in range(1, H + 1):
    Bh = pm[h].predict(Xp[PFH(h)])
    Bh0 = pm[h].predict(Xp.assign(open=np.nan, open_top=np.nan)[PFH(h)])  # same model with a league-average team (median fill)
    a = A_p[:, h - 1] + (Bh - Bh0)                        # the team-situation effect applies to both engines
    E_p[:, h - 1] = np.where(np.isnan(A_p[:, h - 1]), Bh, (1 - lam) * Bh + lam * a)
E_p = smooth_tail(E_p)
elite_p = (pr_pick <= 5) & (pr_age <= 21.5)
# price-side copies of the paths: the same projection plus the scouting prior for young top-5 picks
EA_v = E_v + ELITE_UPLIFT * elite_v[:, None]
EA_p = E_p + ELITE_UPLIFT * elite_p[:, None]
ADJ1 = pm[1].predict(Xp[PFH(1)]) - pm[1].predict(Xp.assign(open=np.nan, open_top=np.nan)[PFH(1)])
if CTX_MODE == "carry":
    for _h in range(2, H + 1):
        E_p[:, _h - 1] += ADJ1  # the boosted year-1 level is the baseline: every later season is shifted by the same amount
_pc = pd.DataFrame({"PLAYER_ID": [int(q["id"][1:]) for q in pros], "team": pr_team, "open_fp": pr_open_raw, "open_z": (pr_open_raw - _omu) / _osd})
for _h in (1, 2, 3):
    _pc[f"adj{_h}"] = ADJ1 if CTX_MODE == "carry" else pm[_h].predict(Xp[PFH(_h)]) - pm[_h].predict(Xp.assign(open=np.nan, open_top=np.nan)[PFH(_h)])
_pc.to_csv(D / f"prospect_context{SFX}.csv", index=False)


# expected next-season value per player (for keep cutoffs): E[pts/g_1] incl. absent as 0
def mean_y1_vet():
    p = fc.s[1].predict_proba(live[F])[:, 1]
    return EA_v[:, 0] * p


def mean_y1_pro():
    X = pd.DataFrame({"logpick": np.log(np.clip(pr_pick, 1, 61)), "dage": pr_age, "talent": pr_talent, "is1": (pr_pick == 1).astype(float)})
    return EA_p[:, 0] * ps_[1].predict_proba(X[PF])[:, 1]


y1 = np.concatenate([mean_y1_vet(), mean_y1_pro()])
y1s = np.sort(y1)[::-1]
cut = lambda k: np.inf if k <= 0 else float(y1s[min(int(k * NTEAMS) - 1, len(y1s) - 1)])

# ------------------------------------------------------------------ 90th-PERCENTILE CAREER ("ceiling outcome")
# The typical career whose TOTAL value (discounted pts/g above replacement, independent of keeper count) is at the 90th
# percentile of the range of real outcomes for players like him. Real analogs supply the spread by age group / draft tier and
# the year-to-year persistence of deviations (if he hits, he hits for years).
CEIL_PCT = 90


def analog_vectors_vet():
    pool = POOL[POOL["yr"] <= LAST_YR - 6].copy()
    R, PRES = np.full((len(pool), H), np.nan), np.zeros((len(pool), H))
    for h in range(1, H + 1):
        ok = pool[f"pres{h}"].notna().to_numpy()
        mu = fc.m[h].predict(pool[FH(h)])
        PRES[:, h - 1] = np.where(ok, pool[f"pres{h}"].fillna(0).to_numpy(), 0)
        R[:, h - 1] = np.where(ok & (pool[f"pres{h}"].to_numpy() == 1), pool[f"v{h}"].to_numpy() - mu, np.nan)
    return AGEB(pool["AGE"].to_numpy() + 1), R, PRES


def analog_vectors_pro():
    pool = hd[hd["real_draft_year"] <= LAST_YR - 5].copy()
    R, PRES = np.full((len(pool), H), np.nan), np.zeros((len(pool), H))
    for h in range(1, H + 1):
        ok = pool[f"pres{h}"].notna().to_numpy()
        mu = pm[h].predict(pool[PFH(h)])
        PRES[:, h - 1] = np.where(ok, pool[f"pres{h}"].fillna(0).to_numpy(), 0)
        R[:, h - 1] = np.where(ok & (pool[f"pres{h}"].to_numpy() == 1), pool[f"v{h}"].to_numpy() - mu, np.nan)
    return np.digitize(np.exp(pool["logpick"].to_numpy()), [4, 11, 31]), R, PRES


def ceiling_paths(E, groups, pool_groups, R, PRES):
    """Expected career shape GIVEN total career value lands at the CEIL_PCT-th percentile.
    Deviations from projection are jointly normal across years with (a) each group's own spread and (b) the real,
    pooled year-to-year correlation of deviations (persistent: a hit tends to stay a hit). For score = sum_h w_h * dev_h
    (w = discount weights), E[dev | score at its z-quantile] = z * Cov w / sqrt(w' Cov w). Smooth, no injury-year artifacts;
    it is the 90th-percentile career for a player who stays in the league (retirement risk is priced separately)."""
    from scipy.stats import norm as _norm
    z = float(_norm.ppf(CEIL_PCT / 100.0))
    Rdf = pd.DataFrame(R)
    corr = Rdf.corr(min_periods=20).fillna(0.0).to_numpy().copy()
    np.fill_diagonal(corr, 1.0)
    ev, evec = np.linalg.eigh(corr)
    corr = (evec * np.clip(ev, 1e-3, None)) @ evec.T  # nearest PSD
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    pooled_sd = Rdf.std().fillna(Rdf.std().mean()).to_numpy()
    w = DELTA ** np.arange(H)
    dev = {}
    for g in np.unique(groups):
        rows = pool_groups == g
        sd = Rdf[rows].std().to_numpy() if rows.sum() >= 8 else pooled_sd
        sd = np.where(np.isnan(sd), pooled_sd, sd)
        cov = np.outer(sd, sd) * corr
        dev[g] = z * (cov @ w) / np.sqrt(w @ cov @ w)
    n = len(E)
    out = np.zeros((n, H))
    for i in range(n):
        out[i, :] = np.maximum(E[i, :H] + dev[groups[i]], 0.0)
    return out


gv, Rv, Pv = analog_vectors_vet()
C_v = ceiling_paths(E_v, AGEB(live_age_next), gv, Rv, Pv)
gp_, Rp, Pp = analog_vectors_pro()
C_p = ceiling_paths(E_p, np.digitize(pr_pick, [4, 11, 31]), gp_, Rp, Pp)


def total_value(k, delta=DELTA):
    c_future = max(cut(k), REPL)
    v_vet = np.zeros(len(live)); v_pro = np.zeros(len(pros))
    for h in range(1, H + 1):
        c = REPL if h == 1 else c_future
        if not np.isfinite(cut(k)) and h > 1:
            continue
        wgt = delta ** (h - 1)
        v_vet += wgt * fc.value(live, live_age_next, h, c, mu=EA_v[:, h - 1])
        v_pro += wgt * prospect_value(pr_pick, pr_age, h, c, mu=EA_p[:, h - 1])
    return np.concatenate([v_vet, v_pro])


res = pd.DataFrame({"player": list(live["PLAYER_NAME"]) + [q["player"] for q in pros],
                    "PLAYER_ID": list(live["PLAYER_ID"]) + [int(q["id"][1:]) for q in pros],
                    "age": np.concatenate([live_age_next, pr_age]),
                    "kind": ["current"] * len(live) + ["prospect"] * len(pros)})
Ks = (0, 1, 3, 5, 8, 19)  # shown in the printed diagnostics
for k in range(20):       # all keeper counts the dashboard slider can pick
    res[f"av{k}"] = total_value(k)
res["exp_y1"] = y1
res.to_csv(D / f"asset_value{SFX}.csv", index=False)


def ceiling_asset(C, k):
    """asset value of the 90th-percentile career treated as the realized path (no extra spread): sum of discounted
    surplus over the keep cutoff in each year he clears it -- same rules as total_value"""
    out = np.zeros(len(C))
    c_future = max(cut(k), REPL)
    for h in range(1, H + 1):
        if not np.isfinite(cut(k)) and h > 1:
            continue
        c = REPL if h == 1 else c_future
        out += DELTA ** (h - 1) * np.maximum(C[:, h - 1] - c, 0.0)
    return out


C_all = np.vstack([C_v, C_p])
C_price = C_all + ELITE_UPLIFT * np.concatenate([elite_v, elite_p])[:, None]  # ceiling asset value is priced with the same prior; the path itself is not shifted
cp = pd.DataFrame(C_all, columns=[f"C{h}" for h in range(1, H + 1)])
cp["PLAYER_ID"] = list(live["PLAYER_ID"]) + [int(q["id"][1:]) for q in pros]
for k in range(20):
    cp[f"CA{k}"] = ceiling_asset(C_price, k)
cp.to_csv(D / f"ceiling_paths{SFX}.csv", index=False)
up = pd.DataFrame(np.vstack([E_v, E_p]), columns=[f"E{h}" for h in range(1, H + 1)])
up["PLAYER_ID"] = list(live["PLAYER_ID"]) + [int(q["id"][1:]) for q in pros]
up["kind"] = ["current"] * len(live) + ["prospect"] * len(pros)
for h in range(1, H + 1):
    up[f"K{h}"] = np.concatenate([kal_v[:, h - 1], A_p[:, h - 1]])
up.to_csv(D / f"unified_paths{SFX}.csv", index=False)


def nn_(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


res["norm"] = res["player"].apply(nn_)
dyn = pd.read_csv(D / "hashtag_dynasty_latest.csv"); dyn["norm"] = dyn["player"].apply(nn_)
adp = pd.read_csv(D / "espn_adp.csv"); adp = adp[(adp["season_id"] == 2027) & adp["adp"].notna() & adp["PLAYER_ID"].notna()]
adp["PLAYER_ID"] = adp["PLAYER_ID"].astype(int)
res = res.drop_duplicates("norm").merge(dyn[["norm", "rank"]].rename(columns={"rank": "dyn"}), on="norm", how="left")
res = res.merge(adp.sort_values("adp").drop_duplicates("PLAYER_ID")[["PLAYER_ID", "adp"]], on="PLAYER_ID", how="left")
dm, am = res[res["dyn"].notna()], res[res["adp"].notna()]
print(f"\n== agreement with anchors (Spearman); dynasty n={len(dm)}, ESPN ADP n={len(am)}   [old approach: VOR@K=19 0.786 / y1 0.853]")
for k in Ks:
    print(f"  K={k:2d} (keep cutoff {cut(k):5.1f} pts/g): vs dynasty {-spearmanr(dm[f'av{k}'], dm['dyn'])[0]:.3f} | vs ESPN ADP {-spearmanr(am[f'av{k}'], am['adp'])[0]:.3f}")
young, old = dm[dm["age"] <= 22], dm[dm["age"] >= 31]
for k in (0, 5, 19):
    print(f"  K={k:2d}: within young(<=22, n={len(young)}) vs dynasty {-spearmanr(young[f'av{k}'], young['dyn'])[0]:.3f} | within old(>=31, n={len(old)}) {-spearmanr(old[f'av{k}'], old['dyn'])[0]:.3f}")
print(f"\n  {'':20s}" + "".join(f"K={k:<4d}" for k in Ks) + "  | real: ADP  dynasty")
for w in ["Cameron Boozer", "Cooper Flagg", "Dylan Harper", "AJ Dybantsa", "Jalen Johnson", "Victor Wembanyama", "Nikola Jokić", "Kevin Durant", "Stephen Curry", "LeBron James"]:
    r = res[res["player"] == w]
    if r.empty:
        continue
    ranks = [int((res[f"av{k}"] > r[f"av{k}"].iloc[0]).sum() + 1) for k in Ks]
    print(f"  {w:20s}" + "".join(f"{x:<6d}" for x in ranks) + f"  | {r['adp'].iloc[0] if pd.notna(r['adp'].iloc[0]) else float('nan'):6.0f} {r['dyn'].iloc[0] if pd.notna(r['dyn'].iloc[0]) else float('nan'):7.0f}")
print("\n== top 20 by asset value at K=0 / K=5 / K=19")
tops = {k: res.sort_values(f"av{k}", ascending=False).head(20)["player"].tolist() for k in (0, 5, 19)}
for i in range(20):
    print(f"  {i+1:2d}  " + " | ".join(f"{tops[k][i]:24s}" for k in (0, 5, 19)))
