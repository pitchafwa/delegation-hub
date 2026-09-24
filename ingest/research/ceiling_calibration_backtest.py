"""Are our young-player trajectories too conservative on ceilings?

For every past draftee (classes 2010-2022, so 4 seasons of outcomes exist) compare the trajectory
the hub would have shown (prospect_trajectories.csv: Output B rookie prior aged with the Kalman
aging curves) to what they actually produced in seasons 0..4 (real fantasy pts/g, >=20 GP).
Split by draft-pick tier. Output B is fit in-sample here, but the question is about GROWTH after the
rookie year (aging-curve slopes), which doesn't depend on that fit.
Survivorship caveat: a player with no >=20 GP season at year k is missing (busts wash out of later
years), so later-year "actual" is biased UP -- we report how many are present.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"

pt = pd.read_csv(D / "prospect_trajectories.csv")
pt["trajectory"] = pt["trajectory"].apply(ast.literal_eval)
u = pd.read_csv(D / "rookie_model_dataset_unified.csv")[["PLAYER_ID", "real_draft_number", "data_source"]].drop_duplicates("PLAYER_ID")
pt = pt.merge(u[["PLAYER_ID", "real_draft_number"]], on="PLAYER_ID", how="left")
pt["pick"] = pt["real_draft_number"].fillna(61)

sb = pd.read_csv(D / "player_season_base.csv")
sb["yr"] = sb["SEASON"].str[:4].astype(int)
sb = sb.sort_values("GP", ascending=False).drop_duplicates(["PLAYER_ID", "yr"])
sb["fpg"] = (sb["PTS"] + 1.5 * sb["REB"] + 2 * sb["AST"] + 3 * sb["STL"] + 3 * sb["BLK"] + sb["FG3M"] + 2 * sb["FTM"]
             - sb["FTA"] - sb["TOV"] + 3 * sb["TD3"]) / sb["GP"].replace(0, np.nan)
act = sb[sb["GP"] >= 20].set_index(["PLAYER_ID", "yr"])["fpg"]

rows = []
for r in pt.itertuples():
    if not (2010 <= r.real_draft_year <= 2021):
        continue
    for k in range(5):
        a = act.get((r.PLAYER_ID, int(r.real_draft_year) + k), np.nan)
        rows.append(dict(pid=r.PLAYER_ID, player=r.player, cls=int(r.real_draft_year), pick=r.pick, k=k,
                         pred=r.trajectory[k], actual=a))
df = pd.DataFrame(rows)
df["tier"] = pd.cut(df["pick"], [0, 3, 10, 20, 40, 100], labels=["picks 1-3", "4-10", "11-20", "21-40", "41+/UDFA"])
df["err"] = df["actual"] - df["pred"]

print("mean predicted vs mean ACTUAL fantasy pts/g by season-since-draft (only players with a >=20 GP season that year)")
print(f"{'tier':10s} {'k':>2s} {'n_players':>9s} {'present':>8s} {'pred':>6s} {'actual':>7s} {'bias':>6s} {'p90 actual':>10s} {'share actual > pred+8':>22s}")
for tier in ["picks 1-3", "4-10", "11-20", "21-40"]:
    for k in range(5):
        s = df[(df["tier"] == tier) & (df["k"] == k)]
        n_all = len(s)
        s = s[s["actual"].notna()]
        if len(s) < 5:
            continue
        print(f"{tier:10s} {k:2d} {n_all:9d} {len(s)/n_all:8.0%} {s['pred'].mean():6.1f} {s['actual'].mean():7.1f} {s['err'].mean():+6.1f} "
              f"{s['actual'].quantile(.9):10.1f} {(s['err'] > 8).mean():22.0%}")
    print()

# star ceiling: among top-10 picks, what share of the class ever reached a star level, vs what our trajectories imply
top = pt[(pt["pick"] <= 10) & pt["real_draft_year"].between(2010, 2021)]
peak_actual = act.groupby(level=0).max()
top = top.assign(peak_actual=top["PLAYER_ID"].map(peak_actual), peak_pred=top["trajectory"].apply(lambda t: max(t[:5])))
print(f"top-10 picks 2010-2021 (n={len(top)}): peak pts/g within first 5 seasons")
print(f"  predicted (our trajectory): mean {top['peak_pred'].mean():.1f}, share >=50: {(top['peak_pred']>=50).mean():.0%}, >=60: {(top['peak_pred']>=60).mean():.0%}")
pa = top["peak_actual"]
print(f"  actual: mean {pa.mean():.1f}, share >=50: {(pa>=50).mean():.0%}, >=60: {(pa>=60).mean():.0%}  (never played 20 GP: {pa.isna().mean():.0%})")
print("\nlargest positive misses (actual year-3 far above prediction), top-10 picks:")
m = df[(df["tier"].isin(["picks 1-3", "4-10"])) & (df["k"] == 3) & df["actual"].notna()].sort_values("err", ascending=False).head(8)
print(m[["player", "cls", "pick", "pred", "actual", "err"]].round(1).to_string(index=False))
print("\nlargest negative misses:")
m = df[(df["tier"].isin(["picks 1-3", "4-10"])) & (df["k"] == 3) & df["actual"].notna()].sort_values("err").head(8)
print(m[["player", "cls", "pick", "pred", "actual", "err"]].round(1).to_string(index=False))
