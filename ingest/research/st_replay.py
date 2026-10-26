"""Short-term program, Phase 6.1 (RESEARCH_short_term_projection_plan.md): replay 2025-26 team-weeks (stream_rules_replay.py, rules "now") with
different next-few-days numbers driving adds, drops and lineups, scored on what really happened (week points + measured future cost of drops).
  LEVEL_SRC=ewma     the replay's own recency-weighted average (default of the replay tool)
  LEVEL_SRC=v1       v1 (live since 10-26): season DELCO + 1.25 + 0.11 x points gap + 0.36 x minutes gap
  LEVEL_SRC=A        Phase 4 model A, coefficients by days ahead (st_phase5.json)
  LEVEL_SRC=A_avail  A + the chance-to-play model (re-centred) in place of the flat 85/81/61%
Run from ingest/:  LEVEL_SRC=A .venv/Scripts/python research/st_replay.py     (writes data/st_replay_<src>.csv; ONLY_VARIANTS defaults to none,now)
"""
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
SRC = os.environ.get("LEVEL_SRC", "ewma")
os.environ.setdefault("ONLY_VARIANTS", "none,now")
import stream_rules_replay as SR                          # noqa: E402
B = SR.B
D = SR.D

if SRC != "ewma":
    import st_harness as H
    P = H.P[H.P["season"] == "2025-26"]
    keyn = P["name"].map(B.norm)
    print(f"{SRC}: {keyn.nunique()} players in the panel, {len(set(keyn) & set(B.series))} matched to the replay's names", flush=True)
    if SRC == "v1":
        lv_by_info = {i: [v] * 7 for i, v in H.baselines()["v1 (live since 10-26)"].reindex(P.index).items()}
    else:
        A = H.a_feats()
        co = json.load(open(D / "st_phase5.json"))["coef_days"]
        cols = []
        for b, c in co.items():
            v = P["delco"] + c["const"] + sum(c[n] * A[n].reindex(P.index).fillna(0.0) for n in A)
            cols.append(v)
        mat = np.column_stack([c.to_numpy() for c in cols])
        lv_by_info = dict(zip(P.index, mat))
    DB = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 10), (11, 14)]
    hist = {}
    for n, d0, i in sorted(zip(keyn, P["date"].dt.date, P.index), key=lambda x: (x[0], x[1])):
        hist.setdefault(n, []).append((d0, i))
    _orig_level = B.level

    def info_before(n, d):
        h = hist.get(n)
        if not h:
            return None
        best = None
        for d0, i in h:
            if d0 < d:
                best = (d0, i)
            else:
                break
        return best

    def level_new(n, d):
        ib = info_before(n, d)
        if ib is None or (d - ib[0]).days > 14:
            return _orig_level(n, d)
        ahead = min(14, (d - ib[0]).days + 2)                 # the rest of the week, centred a couple of days out
        b = next(j for j, (lo, hi) in enumerate(DB) if lo <= ahead <= hi)
        v = lv_by_info[ib[1]][b if len(lv_by_info[ib[1]]) > 1 else 0]
        return float(v) if np.isfinite(v) else _orig_level(n, d)

    B.level = level_new
    B._lvl_cache.clear()
    if SRC == "A_avail":
        pR = pd.read_parquet(D / "st_phase4_avail_recentred.parquet")["p"]
        T = H.T
        tt = T.loc[pR.dropna().index.intersection(T.index)]
        tt = tt[tt["info_id"].isin(P.index)]
        chance = {(i, int(a)): float(p) for i, a, p in zip(tt["info_id"], tt["days_ahead"], pR.reindex(tt.index))}
        _orig_pf = B.p_future

        def day_opts(players, d, today, oracle):
            cands = []
            for p in players:
                if d == today:
                    if p["played"]:
                        cands.append((p["level"], p["elig"], p, 1.0))
                elif p["sched"].get(d, False):
                    ib = info_before(p["n"], today)
                    pf = chance.get((ib[1], (d - ib[0]).days)) if ib else None
                    if pf is None:
                        pf = _orig_pf(p["level"])
                    cands.append((pf * p["level"], p["elig"], p, pf))
            cands.sort(key=lambda x: -x[0])
            acc = B.greedy_accept(cands)
            vals, cnt = [0.0], [0.0]
            for ci in acc:
                vals.append(vals[-1] + cands[ci][0])
                cnt.append(cnt[-1] + cands[ci][3])
            return cands, acc, vals, cnt

        B.day_opts = day_opts

if __name__ == "__main__":
    out_path = D / f"st_replay_{SRC}.csv"
    _to_csv = pd.DataFrame.to_csv

    def to_csv(self, path, *a, **k):                         # keep each source's results in its own file
        if str(path).endswith("stream_rules_replay.csv"):
            path = out_path
        return _to_csv(self, path, *a, **k)

    pd.DataFrame.to_csv = to_csv
    SR.main()
