"""Does starting vs coming off the bench change a player's OWN production, for players who genuinely do both? (Tommy, 2026-09-30)

The population that matters here is "fringe starters": guys who start some games and come off the bench in others within the same season
(a real swing role -- a starter's temporary injury, a slump-driven benching, a sixth-man-type role, etc.), NOT a player whose career simply
transitioned from starter to bench over the years (that's aging/role decline, a completely different, confounded question).

Data: regular_season_box_scores_2010_2024 (14 seasons, ~430k player-games, already pulled for the usage-flow study). `position` is non-null
only for starters (the same flag usage_flow_panel.py uses), so `start = position.notna()`. This is a WITHIN-PLAYER, paired comparison: for
every player-season where he has a real sample of BOTH start games and bench games (>=10 each), compare his own start-game production to
his own bench-game production. Differencing within player controls for talent/role entirely -- the only thing varying is whether that
specific game was a start or not.

Usage: uv run python research/start_bench_study.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
MIN_GAMES_EACH = 10   # need a real sample of both start and bench games in the same season to call this a genuine swing role


def mins(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, str) and ":" in x:
        a, b = x.split(":")
        return float(a) + float(b) / 60
    try:
        return float(x)
    except Exception:
        return 0.0


cols = ["season_year", "game_date", "gameId", "teamTricode", "personId", "personName", "position", "comment", "minutes", "points",
        "reboundsTotal", "assists", "steals", "blocks", "turnovers", "threePointersMade", "freeThrowsMade", "freeThrowsAttempted", "fieldGoalsAttempted"]
fs = [D / "game_logs" / f"regular_season_box_scores_2010_2024_part_{i}.csv" for i in (1, 2, 3)]
d = pd.concat([pd.read_csv(f, usecols=cols) for f in fs], ignore_index=True).drop_duplicates(["gameId", "personId"])
print(f"loaded {len(d):,} player-game rows, {d.season_year.nunique()} seasons")

d["min"] = d.minutes.map(mins)
d = d[d["min"] > 0].copy()   # actually played (drop DNPs/inactive/did-not-dress rows)
d["start"] = d.position.notna()
tds = ((d.points >= 10).astype(int) + (d.reboundsTotal >= 10) + (d.assists >= 10) + (d.steals >= 10) + (d.blocks >= 10)) >= 3
d["fp"] = d.points + 1.5 * d.reboundsTotal + 2 * d.assists + 3 * d.steals + 3 * d.blocks + d.threePointersMade + 2 * d.freeThrowsMade - d.freeThrowsAttempted - d.turnovers + 3 * tds.astype(float)
d["fp_per_min"] = d.fp / d["min"]
d["fga"] = d.fieldGoalsAttempted

print(f"{len(d):,} played games; {d.start.mean()*100:.1f}% were starts")

# ---------------- identify fringe starters: player-seasons with a real sample of both ----------------
g = d.groupby(["personId", "personName", "season_year"])
agg = g.agg(n_start=("start", "sum"), n_bench=("start", lambda s: (~s).sum())).reset_index()
fringe = agg[(agg.n_start >= MIN_GAMES_EACH) & (agg.n_bench >= MIN_GAMES_EACH)]
print(f"\n{len(fringe):,} player-seasons with >= {MIN_GAMES_EACH} starts AND >= {MIN_GAMES_EACH} bench games (out of {len(agg):,} total player-seasons)")
print(f"{fringe.personId.nunique():,} distinct players ever had such a season")

d = d.merge(fringe[["personId", "season_year"]], on=["personId", "season_year"], how="inner")
print(f"{len(d):,} player-games in the fringe-starter sample")

# ---------------- within-player paired comparison ----------------
by_role = d.groupby(["personId", "season_year", "start"]).agg(min_mean=("min", "mean"), fp_mean=("fp", "mean"), fpm_mean=("fp_per_min", "mean"), fga_mean=("fga", "mean"), n=("min", "size")).reset_index()
piv_min = by_role.pivot(index=["personId", "season_year"], columns="start", values="min_mean")
piv_fp = by_role.pivot(index=["personId", "season_year"], columns="start", values="fp_mean")
piv_fpm = by_role.pivot(index=["personId", "season_year"], columns="start", values="fpm_mean")
piv_fga = by_role.pivot(index=["personId", "season_year"], columns="start", values="fga_mean")

d_min = (piv_min[True] - piv_min[False]).dropna()
d_fp = (piv_fp[True] - piv_fp[False]).dropna()
d_fpm = (piv_fpm[True] - piv_fpm[False]).dropna()
d_fga = (piv_fga[True] - piv_fga[False]).dropna()


def report(name, diffs, unit=""):
    m, sd, n = diffs.mean(), diffs.std(), len(diffs)
    se = sd / np.sqrt(n)
    t = m / se
    pct_positive = (diffs > 0).mean() * 100
    print(f"{name}: mean +{m:.2f}{unit} when starting (SE {se:.2f}, t={t:.1f}, n={n} player-seasons, {pct_positive:.0f}% of players show a positive gap)")


print("\n=== within-player, start-night minus bench-night (same player, same season) ===")
report("Minutes", d_min, " min/g")
report("Fantasy points", d_fp, " fp/g")
report("Fantasy points per minute", d_fpm, " fp/min")
report("Field goal attempts", d_fga, " fga/g")

# ---------------- does the SIZE of the effect scale with anything -- e.g. is it bigger for guys who start ONLY occasionally? ----------------
fringe2 = fringe.set_index(["personId", "season_year"])
fringe2["start_share"] = fringe2.n_start / (fringe2.n_start + fringe2.n_bench)
joined = d_fp.rename("d_fp").to_frame().join(d_min.rename("d_min")).join(fringe2["start_share"])
joined["bucket"] = pd.cut(joined.start_share, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print("\n=== effect size by how often he started (share of games that were starts) ===")
print(joined.groupby("bucket", observed=True)[["d_fp", "d_min"]].agg(["mean", "count"]))

# ---------------- sanity check: is this just noise? compare to a placebo (odd vs even games, no real start/bench meaning) ----------------
d_sorted = d.sort_values(["personId", "season_year", "game_date"])
d_sorted["game_idx"] = d_sorted.groupby(["personId", "season_year"]).cumcount()
d_sorted["placebo"] = d_sorted.game_idx % 2 == 0
by_placebo = d_sorted.groupby(["personId", "season_year", "placebo"]).agg(fp_mean=("fp", "mean")).reset_index()
piv_pl = by_placebo.pivot(index=["personId", "season_year"], columns="placebo", values="fp_mean")
d_placebo = (piv_pl[True] - piv_pl[False]).dropna()
print("\n=== placebo check: same players, odd vs even game number (should be ~0 if the real effect above isn't just noise) ===")
report("Placebo fantasy points", d_placebo, " fp/g")

d.to_pickle(D / "start_bench_games.pkl")
fringe.to_pickle(D / "start_bench_fringe_seasons.pkl")
print("\nwrote data/start_bench_games.pkl, data/start_bench_fringe_seasons.pkl")
