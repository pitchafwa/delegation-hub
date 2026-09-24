"""Empirically choose the rookie model's target window, mirroring RUPI's own
discipline: the right window is the one where a KNOWN real signal (draft
position) predicts it most reliably and stably across real draft classes
(leave-one-class-out), not the one that "feels right." A window dominated by
short-term noise will show WEAKER, less stable draft-position correlation
than one that captures real, sustained talent -- that's the actual test.

Candidates:
  entry_3yr_best2   -- RUPI's own window, direct analog
  entry_5yr_best3   -- WRPI's window (longer runway, matches WR-style shelf life)
  entry_7yr_avg     -- full rookie scale + first extension
  age_24_28_avg     -- the real "prime years" window found in serious draft research
  age_22_29_best3   -- peak-anchored across the realistic age range, not tied to entry age
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent

season_base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
season_base["SEASON_YEAR"] = season_base["SEASON"].apply(lambda s: int(s.split("-")[0]))
season_base["FANTASY_PTS"] = (
    season_base["PTS"] + 1.5 * season_base["REB"] + 2 * season_base["AST"] + 3 * season_base["STL"]
    + 3 * season_base["BLK"] + season_base["FG3M"] + 2 * season_base["FTM"] - season_base["FTA"]
    - season_base["TOV"] + 3 * season_base["TD3"]
)
season_base["FANTASY_PPG"] = season_base["FANTASY_PTS"] / season_base["GP"].replace(0, np.nan)

bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
bio = bio.dropna(subset=["DRAFT_YEAR"]).copy()
bio["DRAFT_YEAR"] = pd.to_numeric(bio["DRAFT_YEAR"], errors="coerce")
bio["DRAFT_NUMBER"] = pd.to_numeric(bio["DRAFT_NUMBER"], errors="coerce")
bio = bio.dropna(subset=["DRAFT_YEAR", "DRAFT_NUMBER"])
bio = bio.rename(columns={"PERSON_ID": "PLAYER_ID"})[["PLAYER_ID", "DRAFT_YEAR", "DRAFT_NUMBER"]]
# season_base (real per-season data) only covers 2010-2026 -- pre-2010 draft
# classes would silently score ENTRY-window values as a false "0" (missing
# rookie-year coverage looks identical to a real bust otherwise), since
# entry_window() uses ppg_or_zero() which defaults to 0.0 for ANY missing
# (pid, year), including years before real coverage began. Confirmed real
# bug: 2,824 pre-2010 draftees were in this file before the fix.
#
# age_window_avg()/age_window_best() are NOT vulnerable to that same bug --
# they look up real season_base rows directly by AGE, never assume a zero
# for a year outside coverage, and already return NaN via rookie_age()/
# year_would_reach_hi if genuinely unresolvable. Applying the >=2010 floor
# to them too was needlessly conservative and silently zeroed out the
# "actual outcome" for real, resolvable 2008-2009 draftees whose age-22-29
# window mostly falls inside 2010-2026 real coverage (caught via a real
# user question: "why don't older prospects show an actual score?"). Use a
# separate, looser floor (2008, matching real college-data coverage start)
# for the age-anchored bio pool only.
bio_entry = bio[bio["DRAFT_YEAR"] >= 2010]
bio_age = bio[bio["DRAFT_YEAR"] >= 2008]

MAX_REAL_YEAR = season_base["SEASON_YEAR"].max()  # 2025 -- last real completed season

ppg_lookup = season_base.set_index(["PLAYER_ID", "SEASON_YEAR"])["FANTASY_PPG"]
age_lookup = season_base.set_index(["PLAYER_ID", "SEASON_YEAR"])["AGE"]


def ppg_or_zero(pid, year):
    return ppg_lookup.get((pid, year), 0.0)


def entry_window(pid, draft_year, n_seasons, best_k):
    years = [draft_year + k for k in range(n_seasons)]
    vals = [ppg_or_zero(pid, y) for y in years]
    return np.mean(sorted(vals, reverse=True)[:best_k])


def rookie_age(pid, draft_year):
    """Real age in the player's actual rookie season row, if one exists."""
    row = season_base[(season_base["PLAYER_ID"] == pid) & (season_base["SEASON_YEAR"] == draft_year)]
    if row.empty:
        # fall back to their earliest real row, whenever it is
        rows = season_base[season_base["PLAYER_ID"] == pid].sort_values("SEASON_YEAR")
        if rows.empty:
            return np.nan
        return rows.iloc[0]["AGE"] - (rows.iloc[0]["SEASON_YEAR"] - draft_year)
    return row.iloc[0]["AGE"]


def age_window_avg(pid, draft_year, age_lo, age_hi):
    r_age = rookie_age(pid, draft_year)
    if np.isnan(r_age):
        return np.nan
    # real calendar time needed for this player to have POSSIBLY reached age_hi
    year_would_reach_hi = draft_year + (age_hi - r_age)
    if year_would_reach_hi > MAX_REAL_YEAR:
        return np.nan  # too recent/young -- genuinely unresolved, not a bust
    player_rows = season_base[season_base["PLAYER_ID"] == pid]
    in_range = player_rows[(player_rows["AGE"] >= age_lo) & (player_rows["AGE"] <= age_hi)]
    n_seasons_expected = age_hi - age_lo + 1
    real_vals = in_range["FANTASY_PPG"].tolist()
    # pad with real zeros for seasons within the window they didn't play at all
    # (out of the league / hadn't caught back on) -- a real outcome, not missing.
    padded = real_vals + [0.0] * max(0, n_seasons_expected - len(real_vals))
    return float(np.mean(padded))


def age_window_best(pid, draft_year, age_lo, age_hi, best_k):
    r_age = rookie_age(pid, draft_year)
    if np.isnan(r_age):
        return np.nan
    year_would_reach_hi = draft_year + (age_hi - r_age)
    if year_would_reach_hi > MAX_REAL_YEAR:
        return np.nan
    player_rows = season_base[season_base["PLAYER_ID"] == pid]
    in_range = player_rows[(player_rows["AGE"] >= age_lo) & (player_rows["AGE"] <= age_hi)]
    real_vals = in_range["FANTASY_PPG"].tolist()
    padded = real_vals + [0.0] * max(0, best_k - len(real_vals))
    return float(np.mean(sorted(padded, reverse=True)[:best_k]))


results = []
for _, row in bio_age.iterrows():
    pid, dy = row["PLAYER_ID"], row["DRAFT_YEAR"]
    is_entry_safe = dy >= 2010  # entry_window() is unsafe before this (see note above)
    entry = {
        "PLAYER_ID": pid, "DRAFT_YEAR": dy, "DRAFT_NUMBER": row["DRAFT_NUMBER"],
        "entry_3yr_best2": entry_window(pid, dy, 3, 2) if is_entry_safe and dy + 2 <= MAX_REAL_YEAR else np.nan,
        "entry_5yr_best3": entry_window(pid, dy, 5, 3) if is_entry_safe and dy + 4 <= MAX_REAL_YEAR else np.nan,
        "entry_7yr_avg": entry_window(pid, dy, 7, 7) if is_entry_safe and dy + 6 <= MAX_REAL_YEAR else np.nan,
        "age_24_28_avg": age_window_avg(pid, dy, 24, 28),
        "age_22_29_best3": age_window_best(pid, dy, 22, 29, 3),
    }
    results.append(entry)

df = pd.DataFrame(results)

# A window is only evaluable for a given player if it's chronologically
# resolvable AND the age-anchored ones need the player to have actually
# reached that age range in real data already (not just theoretically old
# enough) -- age_24_28_avg/age_22_29_best3 are already NaN otherwise.
WINDOWS = ["entry_3yr_best2", "entry_5yr_best3", "entry_7yr_avg", "age_24_28_avg", "age_22_29_best3"]

print("=== Real coverage: how many of the drafted players (2010-2025 draft classes) have a resolvable value per window ===")
for w in WINDOWS:
    n = df[w].notna().sum()
    print(f"{w:20s} {n:5d} players ({n/len(df):.1%})")

print("\n=== Newest fully-resolvable draft class per window (chronological lag before a new class can be evaluated) ===")
for w in WINDOWS:
    resolved = df[df[w].notna()]
    print(f"{w:20s} newest draft class covered: {int(resolved['DRAFT_YEAR'].max())}")


def loco_cv_spearman(window_col):
    sub = df.dropna(subset=[window_col, "DRAFT_NUMBER"]).copy()
    classes = sub["DRAFT_YEAR"].unique()
    fold_rhos = []
    for held_out in classes:
        test = sub[sub["DRAFT_YEAR"] == held_out]
        if len(test) < 15:
            continue
        # draft position is a fixed, known signal -- no fitting needed, just
        # check its real rank correlation with the window's value in this
        # held-out class (negate pick so "good" is high on both sides).
        rho, _ = spearmanr(-test["DRAFT_NUMBER"], test[window_col])
        if np.isfinite(rho):
            fold_rhos.append(rho)
    return np.mean(fold_rhos), len(fold_rhos)


print("\n=== LOCO-CV: does real draft position predict this window reliably across real, held-out draft classes? ===")
print(f"{'window':20s} {'mean rho':>10s} {'n folds':>8s}")
for w in WINDOWS:
    rho, n_folds = loco_cv_spearman(w)
    print(f"{w:20s} {rho:10.4f} {n_folds:8d}")

df.to_csv(ROOT / "data" / "target_window_test.csv", index=False)
