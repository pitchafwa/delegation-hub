"""Does the Asset value model itself carry the biases we just fixed in the VOR trajectories?

(a) veterans: held-out bias/RMSE of its expected pts/g (given still playing) by age group, at 1-3 seasons out
    (origins 2015-2019, trained only on outcomes observable then; graded on players it (or the naive rule)
    already rated fantasy-relevant at the origin -- known then, no peeking).
(b) prospects: leave-one-draft-class-out bias by season-since-draft for picks 1-3 / 4-10 / 11-15.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
sys.argv = ["x"]
src = open(__file__.replace("asset_value_bias_check.py", "asset_value_v2.py"), encoding="utf-8").read()
exec(src.split("# ------------------------------------------------------------------ out-of-sample test")[0])  # panel, Forecaster, F, POOL...
exec(src.split("# ------------------------------------------------------------------ prospects: draft-slot model")[1].split("# bring in the hub's prospects")[0])  # hd, rg, PF...

print("(a) VETERANS: held-out mean of (actual - model forecast) of pts/g among players still playing; + = model too LOW")
rows = []
for ty in (2015, 2016, 2017, 2018, 2019):
    test = POOL[(POOL["yr"] == ty) & (POOL["fpg"] >= 25)]
    fc = Forecaster(POOL[POOL["yr"] < ty], asof=ty, hmax=3)
    for h in (1, 2, 3):
        pv = test[test[f"pres{h}"] == 1]
        if len(pv) < 20:
            continue
        mu = fc.m[h].predict(pv[F])
        rows.append(pd.DataFrame({"h": h, "age": pv["AGE"].to_numpy() + 1, "err": pv[f"v{h}"].to_numpy() - mu}))
r = pd.concat(rows)
r["grp"] = pd.cut(r["age"], [17, 21.5, 23.5, 26.5, 30.5, 45], labels=["<=21", "22-23", "24-26", "27-30", "31+"])
tab = r.groupby(["grp", "h"])["err"].agg(["mean", "count"]).unstack("h")
print(tab.round(2).to_string())

print("\n(b) PROSPECTS: leave-one-draft-class-out bias of the model's expected pts/g given playing (+ = model too LOW)")
hd2 = hd.copy()
out = []
for cls in sorted(hd2["real_draft_year"].unique()):
    if cls > 2021:
        continue
    tr, te = hd2[hd2["real_draft_year"] != cls], hd2[hd2["real_draft_year"] == cls]
    for h in range(1, 6):
        t = tr[tr[f"pres{h}"] == 1]
        m = rg().fit(t[PF], t[f"v{h}"])
        tt = te[te[f"pres{h}"] == 1]
        if tt.empty:
            continue
        out.append(pd.DataFrame({"h": h, "pick": np.exp(tt["logpick"].to_numpy()), "err": tt[f"v{h}"].to_numpy() - m.predict(tt[PF])}))
o = pd.concat(out)
o["tier"] = pd.cut(o["pick"], [0, 3.5, 10.5, 15.5], labels=["picks 1-3", "4-10", "11-15"])
print(o.groupby(["tier", "h"])["err"].agg(["mean", "count"]).unstack("h").round(2).to_string())
