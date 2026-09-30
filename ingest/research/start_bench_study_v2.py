"""Follow-up to start_bench_study.py (Tommy, 2026-09-30): is the +8.5 fp/g start-vs-bench effect already partly explained by the usage-flow
system's "teammate is out" boost? If a fringe player's start nights mostly happen BECAUSE a teammate is hurt, and usage_flow_panel.pkl already
gives him a boost for that same absence, adding a separate flat "+8.5 for starting" on top would double-count the same underlying event.

Reuses usage_flow_panel.pkl (already built for the usage-flow study; has V_min = total trailing-minutes of ABSENT rotation teammates that game,
0 if none) to split every fringe-starter's games into "healthy roster" (V_min==0, nobody out) vs "a teammate is out" (V_min>0), then redoes the
SAME within-player start-vs-bench comparison separately in each bucket. The healthy-roster number is the one that's safe to add as a NEW signal
without overlapping usage-flow; the teammate-out number tells us how much of the naive effect usage-flow may already be covering.

Usage: uv run python research/start_bench_study_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
MIN_GAMES_EACH = 10

d = pd.read_pickle(D / "usage_flow_panel.pkl")
print(f"loaded {len(d):,} teammate-game rows")
d["start"] = d["start"].astype(bool)
d["healthy"] = d.V_min == 0
print(f"{(~d.healthy).mean()*100:.1f}% of games have at least one rotation teammate out (V_min>0)")

# ---------------- fringe starters: player-seasons with a real sample of both, same definition as v1 ----------------
g = d.groupby(["personId", "personName", "season_year"])
agg = g.agg(n_start=("start", "sum"), n_bench=("start", lambda s: (~s).sum())).reset_index()
fringe = agg[(agg.n_start >= MIN_GAMES_EACH) & (agg.n_bench >= MIN_GAMES_EACH)]
print(f"{len(fringe):,} fringe-starter player-seasons")
d = d.merge(fringe[["personId", "season_year"]], on=["personId", "season_year"], how="inner")


def within_player_gap(sub, min_each=4):
    """mean (start - bench) for fp and minutes, within player-season, requiring >=min_each games in EACH cell of this subset."""
    by = sub.groupby(["personId", "season_year", "start"]).agg(fp_mean=("fp", "mean"), min_mean=("min", "mean"), n=("fp", "size")).reset_index()
    cnt = by.pivot(index=["personId", "season_year"], columns="start", values="n")
    ok = cnt[cnt.get(True, 0).fillna(0).ge(min_each) & cnt.get(False, 0).fillna(0).ge(min_each)].index
    piv_fp = by.pivot(index=["personId", "season_year"], columns="start", values="fp_mean")
    piv_min = by.pivot(index=["personId", "season_year"], columns="start", values="min_mean")
    d_fp = (piv_fp[True] - piv_fp[False]).reindex(ok).dropna()
    d_min = (piv_min[True] - piv_min[False]).reindex(ok).dropna()
    return d_fp, d_min


def report(name, diffs):
    m, sd, n = diffs.mean(), diffs.std(), len(diffs)
    se = sd / np.sqrt(n) if n else float("nan")
    print(f"  {name}: mean {m:+.2f} (SE {se:.2f}, n={n} player-seasons, t={m/se:.1f})" if n else f"  {name}: n=0, skipped")


print("\n=== ALL fringe-starter games (matches v1's headline number) ===")
fp_all, min_all = within_player_gap(d, min_each=10)
report("Fantasy points, start minus bench", fp_all)
report("Minutes, start minus bench", min_all)

print("\n=== split by whether a rotation teammate was OUT that game ===")
healthy = d[d.healthy]
hurt = d[~d.healthy]
print(f"(healthy-roster games: {len(healthy):,}; teammate-out games: {len(hurt):,}, within the fringe-starter sample)")
fp_h, min_h = within_player_gap(healthy, min_each=4)
report("HEALTHY ROSTER  -- fantasy points, start minus bench", fp_h)
report("HEALTHY ROSTER  -- minutes, start minus bench", min_h)
fp_t, min_t = within_player_gap(hurt, min_each=4)
report("TEAMMATE OUT    -- fantasy points, start minus bench", fp_t)
report("TEAMMATE OUT    -- minutes, start minus bench", min_t)

print("\n=== how often is a fringe starter's START specifically explained by a teammate being out? ===")
starts = d[d.start]
print(f"of {len(starts):,} start-games in the fringe sample, {  (starts.V_min>0).mean()*100:.1f}% had a rotation teammate out that game")
bench = d[~d.start]
print(f"of {len(bench):,} bench-games in the fringe sample,  {(bench.V_min>0).mean()*100:.1f}% had a rotation teammate out that game")
