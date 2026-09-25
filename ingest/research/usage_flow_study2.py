"""Study 4b: refine WHO absorbs a missing player's production: minutes-based shares, role-specific capture, and positional similarity.
Position: starters carry G/F/C in the box score; every player-season gets P(G), P(F), P(C) from a multinomial model of style (assists, rebounds, blocks per minute) fit on starters.
Run after usage_flow_study.py.  Writes data/usage_flow_model.json when finished."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
t = pd.read_pickle(D / "usage_flow_panel2.pkl")
ab = pd.read_pickle(D / "usage_flow_absences.pkl")
raw = pd.concat([pd.read_csv(D / "game_logs" / f"regular_season_box_scores_2010_2024_part_{i}.csv", usecols=["season_year", "personId", "position", "minutes", "reboundsTotal", "assists", "blocks", "steals", "threePointersMade"]) for i in (1, 2, 3)])
def mn(x):
    if pd.isna(x): return 0.0
    if isinstance(x, str) and ":" in x:
        a, b = x.split(":"); return float(a) + float(b) / 60
    return float(x)
raw["m"] = raw.minutes.map(mn)
ps = raw[raw.m > 0].groupby(["personId", "season_year"]).agg(m=("m", "sum"), reb=("reboundsTotal", "sum"), ast=("assists", "sum"), blk=("blocks", "sum"), stl=("steals", "sum"), tp=("threePointersMade", "sum"), n=("m", "size")).reset_index()
for c in ("reb", "ast", "blk", "stl", "tp"):
    ps[c + "36"] = ps[c] / ps.m * 36
starter_pos = raw[raw.position.notna()].groupby(["personId", "season_year"]).position.agg(lambda s: s.mode().iat[0]).rename("pos").reset_index()
ps = ps.merge(starter_pos, on=["personId", "season_year"], how="left")
F = ["reb36", "ast36", "blk36", "stl36", "tp36"]
lab = ps[ps.pos.notna() & (ps.m > 300)]
clf = LogisticRegression(max_iter=2000).fit(lab[F], lab.pos)
print("position model accuracy on starters:", round(clf.score(lab[F], lab.pos), 3), clf.classes_)
P = pd.DataFrame(clf.predict_proba(ps[F].fillna(0)), columns=[f"p{c}" for c in clf.classes_], index=ps.index)
ps = pd.concat([ps, P], axis=1)
pp = ps[["personId", "season_year", "pC", "pF", "pG"]]
t = t.merge(pp, on=["personId", "season_year"], how="left")
ab = ab.merge(pp, on=["personId", "season_year"], how="left")
# pair each absent player with each teammate who played in that team-game
pairs = t[["gameId", "teamTricode", "personId", "season", "bs", "bs_min", "pC", "pF", "pG", "gd"]].merge(
    ab[["gameId", "teamTricode", "personId", "fp15", "mp15", "pC", "pF", "pG"]].rename(columns={"personId": "xid", "fp15": "x_fp", "mp15": "x_mp", "pC": "xC", "pF": "xF", "pG": "xG"}), on=["gameId", "teamTricode"])
pairs["sim"] = pairs.pC * pairs.xC + pairs.pF * pairs.xF + pairs.pG * pairs.xG
pairs = pairs[pairs.personId != pairs.xid]
agg = pairs.assign(vs=pairs.x_fp * pairs.sim, vm_s=pairs.x_mp * pairs.sim).groupby(["gameId", "teamTricode", "personId"]).agg(Vsim=("vs", "sum"), Vsim_min=("vm_s", "sum")).reset_index()
t = t.merge(agg, on=["gameId", "teamTricode", "personId"], how="left").fillna({"Vsim": 0.0, "Vsim_min": 0.0})
keys = [t.personId, t.season]
dm = lambda x: x - x.groupby(keys).transform("mean")
y = dm(t.fp)
def fit(cols, name):
    X = np.column_stack([dm(c).to_numpy() for c in cols]); yy = y.to_numpy()
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r2 = 1 - ((yy - X @ beta) ** 2).sum() / (yy ** 2).sum()
    print(f"   {name:58s} coefs {np.round(beta, 3)}  R2 {r2:.4f}")
    return beta
sh = lambda w: w / w.groupby([t.gameId, t.teamTricode]).transform("sum")
sh_p = sh(t.bs.clip(lower=0.5) ** 0.5)
sh_m = sh(t.bs_min.clip(lower=1))
sh_m2 = sh(t.bs_min.clip(lower=1) ** 0.5)
print("\nA. share definition (single kappa)")
fit([t.V_fp * sh_p], "production^0.5 shares")
fit([t.V_fp * sh_m], "minutes shares")
fit([t.V_fp * sh_m2], "minutes^0.5 shares")
fit([t.V_fp * sh(t.bs_min.clip(lower=1) ** 0.5 * t.bs.clip(lower=0.5) ** 0.25)], "minutes^0.5 x production^0.25")
print("\nB. positional similarity (production^0.5 shares): total vacated + similarity-weighted vacated")
fit([t.V_fp * sh_p, t.Vsim * sh_p], "total + similar-position")
print("\nC. capture by teammate role: separate kappa per season-minutes tier")
tiers = pd.cut(t.bs_min, [0, 15, 22, 30, 60], labels=False)
fit([t.V_fp * sh_p * (tiers == k) for k in range(4)], "kappa by tier (<15, 15-22, 22-30, 30+ mpg)")
fit([t.V_fp * sh_p * (tiers == k) for k in range(4)] + [t.Vsim * sh_p * (tiers == k) for k in range(4)], "kappa by tier + similarity by tier")
print("\nD. size of the absentee: separate kappa for absent minutes 12-20 / 20-28 / 28+ (vacated production split by the absentee's size)")
ab["size"] = pd.cut(ab.mp15, [0, 20, 28, 60], labels=False)
V = ab.pivot_table(index=["gameId", "teamTricode"], columns="size", values="fp15", aggfunc="sum", fill_value=0.0)
V.columns = [f"V{int(c)}" for c in V.columns]
t = t.merge(V.reset_index(), on=["gameId", "teamTricode"], how="left").fillna({"V0": 0.0, "V1": 0.0, "V2": 0.0})
b = fit([t.V0 * sh_p, t.V1 * sh_p, t.V2 * sh_p], "kappa by absentee size (12-20, 20-28, 28+ mpg)")
b2 = fit([t.V0 * sh_p * (tiers == k) for k in range(4)] + [t.V1 * sh_p * (tiers == k) for k in range(4)] + [t.V2 * sh_p * (tiers == k) for k in range(4)], "size x role (12 kappas)")
print(np.round(b2.reshape(3, 4), 2), "(rows: absentee size small/mid/large; columns: teammate tier <15,15-22,22-30,30+ mpg)")

# ---------------- final parameters for the planner
Xf = [t.V_fp * sh_p * (tiers == k) for k in range(4)] + [t.Vsim * sh_p * (tiers == k) for k in range(4)]
bf = fit(Xf, "FINAL: kappa by tier + similarity by tier")
# season-length training only up to 2019-20 for an honest out-of-sample check on 2020-21..2023-24
tr = (t.season <= "2019-20").to_numpy()
Xa = np.column_stack([dm(c).to_numpy() for c in Xf]); ya = y.to_numpy()
b_tr, *_ = np.linalg.lstsq(Xa[tr], ya[tr], rcond=None)
te = ~tr
r2_te = 1 - ((ya[te] - Xa[te] @ b_tr) ** 2).sum() / (ya[te] ** 2).sum()
print("   out-of-sample R2 (fit <=2019-20, test 2020-21..2023-24):", round(r2_te, 4), " coefs", np.round(b_tr, 3))

# ---------------- how long does an absence last? hazard of being Out again at the next listing, by how long he has been out (official reports, 5 seasons)
e = pd.read_pickle(D / "injury_rows_05pm.pkl")
e = e[e.kind.isin(["injury", "illness"]) | e.status.eq("Out")].sort_values(["player_key", "gd"]).copy()
e["out"] = (e.status == "Out").astype(int)
e["gap"] = e.groupby("player_key").gd.diff().dt.days
e["newrun"] = ((e.out != e.groupby("player_key").out.shift(1)) | (e.gap > 7) | e.gap.isna()).astype(int)
e["run"] = e.groupby("player_key").newrun.cumsum()
e["streak"] = e.groupby(["player_key", "run"]).cumcount() + 1
e["next_out"] = e.groupby("player_key").out.shift(-1)
e["next_gap"] = e.groupby("player_key").gd.shift(-1) - e.gd
o = e[(e.out == 1) & e.next_out.notna() & (e.next_gap.dt.days <= 7)]
bins = [0, 1, 2, 3, 5, 8, 14, 10000]
labels = ["1", "2", "3", "4-5", "6-8", "9-14", "15+"]
o = o.assign(bk=pd.cut(o.streak, bins, labels=labels))
haz = o.groupby("bk", observed=True).next_out.agg(["mean", "size"])
print("\nP(still Out at the next listing) by how many listings he has already been Out:\n", haz.round(3).to_string())
# a player who is not listed at the next report has either returned (not listed) or the report skipped him: treat 'not listed within 7 days' as returned
json.dump({"kappa": [round(float(x), 4) for x in bf[:4]], "lam": [round(float(x), 4) for x in bf[4:]], "share_gamma": 0.5, "tier_mpg": [15, 22, 30],
           "pos_classes": list(map(str, clf.classes_)), "pos_coef": clf.coef_.round(5).tolist(), "pos_intercept": clf.intercept_.round(5).tolist(), "pos_features": F,
           "hazard": {str(k): round(float(v), 3) for k, v in haz["mean"].items()}, "hazard_n": {str(k): int(v) for k, v in haz["size"].items()}, "hazard_bins": bins, "hazard_labels": labels},
          open(Path(__file__).resolve().parent / "usage_flow_model.json", "w"), indent=1)
print("wrote usage_flow_model.json")
