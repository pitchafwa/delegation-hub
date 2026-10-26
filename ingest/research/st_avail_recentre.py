import sys; sys.path.insert(0, 'research')
import numpy as np, pandas as pd, st_harness as H
from scipy.special import logit, expit
T, D = H.T, H.D
pA = pd.read_parquet(D / "st_phase4_avail.parquet")["pA"]
tg = T.sort_values(["info_id", "k"])
y = tg["played"].astype(float)
print("played share by season:", tg.groupby("season")["played"].mean().round(3).to_dict())
print("model mean by season:  ", pA.groupby(tg["season"]).mean().round(3).to_dict())
# re-centre: shift in log-odds so the last season before the judged one matches (rolling: each test season uses the season before it)
seasons = sorted(tg["season"].unique())
out = pd.Series(np.nan, index=tg.index)
for s in H.TEST:
    prev = seasons[seasons.index(s) - 1]
    if prev == "2019-20":
        prev = seasons[seasons.index(s) - 2]
    m = tg["season"] == prev
    lo, hi = -1.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if expit(logit(pA[m].clip(1e-4, 1 - 1e-4)) + mid).mean() > y[m].mean(): hi = mid
        else: lo = mid
    ms = tg["season"] == s
    out[ms] = expit(logit(pA[ms].clip(1e-4, 1 - 1e-4)) + mid)
    print(f"  {s}: shift {mid:+.3f} from {prev}")
print(H.avail_table({"flat": H.flat_avail(), "model": pA, "model re-centred on last season": out}).round(3).to_string())
out.to_frame("p").to_parquet(D / "st_phase4_avail_recentred.parquet")
lvA = H.predict_by_k  # level of A is rebuilt in st_phase4; here use season DELCO and v1 only for a lean check of the availability part
v1 = H.baselines()["v1 (live since 10-26)"]
print(H.week_table({"v1 + flat": v1, "v1 + model": v1, "v1 + model re-centred": v1}, {"v1 + flat": H.flat_avail(), "v1 + model": pA, "v1 + model re-centred": out}).round(2).to_string())
