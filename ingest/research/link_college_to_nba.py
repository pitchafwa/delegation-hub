"""Identity-link real college player-seasons to real NBA PLAYER_IDs.
Match on normalized name + draft-year proximity, then VALIDATE using each
player's REAL known NBA draft slot (from player_bio.csv, sourced from
nba_api directly) against torvik's own recorded 'pick' field where available
-- exactly the kind of stable-ID-first, name-matching-as-validated-fallback
discipline used throughout this whole project, applied to a new domain.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

college = pd.read_csv(ROOT / "data" / "college_player_season_2008_2025.csv", low_memory=False)
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")


def normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


bio["norm_name"] = bio["PLAYER_FIRST_NAME"].fillna("") .str.cat(bio["PLAYER_LAST_NAME"].fillna(""), sep=" ").apply(normalize_name)
bio_drafted = bio.dropna(subset=["DRAFT_YEAR"]).copy()
bio_drafted["DRAFT_YEAR"] = pd.to_numeric(bio_drafted["DRAFT_YEAR"], errors="coerce")
bio_drafted["DRAFT_NUMBER_NUM"] = pd.to_numeric(bio_drafted["DRAFT_NUMBER"], errors="coerce")
# nba_api uses DRAFT_NUMBER=0 (with DRAFT_ROUND=0) as its own sentinel for
# "undrafted", not a real pick -- treating it as a literal value made
# undrafted players look like the #1 overall pick (better than every real
# pick), massively inflating their post-draft score via the capital curve.
# Caught via a real user report (players "mistakenly showing as pick 0").
bio_drafted.loc[bio_drafted["DRAFT_NUMBER_NUM"] == 0, "DRAFT_NUMBER_NUM"] = np.nan

college["norm_name"] = college["player"].apply(normalize_name)
# a player's college "year" field = the college season; real draft happens
# in the calendar year the season ENDS (torvik year 2022 = 2021-22 season,
# draft summer 2022) -- matches nba_api DRAFT_YEAR directly.
#
# BUG (found via a real user question about international-player coverage,
# which led to auditing the "gap" population and finding real COLLEGE
# players missing too): grouping by norm_name here, with normalize_name()
# stripping Jr/Sr/II/III/IV suffixes, silently MERGES unrelated real people
# who share a base name -- e.g. "Anthony Davis" (Kentucky, 2012, the real
# NBA star) and "Anthony Davis Jr." (Fairfield, 2024, a completely
# different, unrelated player) both normalize to "anthony davis", and
# .tail(1) then picks whichever one's season is chronologically LATER --
# silently discarding the real target's actual college record. Confirmed
# 758 normalized-name groups contain multiple distinct real raw name
# spellings. Fixed by grouping on the RAW player string (each real
# name-as-written keeps its own identity and its own last season);
# normalize_name() is still used, but only later, to fuzzy-match a given
# raw college identity against bio_drafted -- never to collapse multiple
# college identities into one before that point.
#
# SECOND bug (same root cause class, worse): even grouping by the raw
# player string isn't enough -- torvik's own data has genuinely DIFFERENT
# real people sharing the exact same raw name with no suffix to distinguish
# them at all (three real "Devin Booker"s: Clemson 2010-13, Kentucky 2015 --
# the actual NBA star -- and George Mason 2026). Grouping by raw name still
# picks whichever is chronologically last (George Mason, wrong). The real
# fix: torvik's own "id" field is a stable per-career identity, already
# used elsewhere in this pipeline (build_breakout_age.py) for exactly this
# reason -- group by THAT, not by name, so each real college career (however
# it's spelled, however many namesakes share the string) keeps its own
# correct final season. The subsequent per-row bio year-gap match (and the
# PLAYER_ID dedup-by-smallest-gap below) then correctly resolves which of
# several same-named college careers is the real NBA draftee.
college_last_season = college.sort_values("year").groupby("id").tail(1)

# Build a name -> list of (PLAYER_ID, DRAFT_YEAR, DRAFT_NUMBER) map, since
# names aren't globally unique (the real father/son namesake risk the
# WRPI/RUPI doc warned about) -- resolved by draft-year proximity, not blind
# first-match.
bio_by_name = bio_drafted.groupby("norm_name")

matches = []
unmatched = []
ambiguous = []
for _, row in college_last_season.iterrows():
    name = row["norm_name"]
    if name not in bio_by_name.groups:
        unmatched.append(row["player"])
        continue
    candidates = bio_by_name.get_group(name)
    if len(candidates) == 1:
        cand = candidates.iloc[0]
        # BUG (found via dashboard review): a unique name match was accepted
        # with NO year-proximity check at all, unlike the multi-candidate
        # branch below. This let a 1970s-80s NBA player (the only "Alex
        # English"/"Amir Johnson" in bio_drafted) get silently matched to an
        # unrelated MODERN college season just because they shared a
        # normalized name and no OTHER bio candidate existed to force the
        # ambiguous-match check. Real Amir Johnson never played college at
        # all -- confirms this was a false match, not a real namesake.
        year_gap = abs(cand["DRAFT_YEAR"] - row["year"])
        if year_gap > 2:
            ambiguous.append((row["player"], row["year"], [[cand["DRAFT_YEAR"]]]))
            continue
        matches.append({
            "college_player": row["player"], "college_last_year": row["year"],
            "PLAYER_ID": cand["PERSON_ID"], "real_draft_year": cand["DRAFT_YEAR"],
            "real_draft_number": cand["DRAFT_NUMBER_NUM"], "torvik_pick": row["pick"],
            "match_method": "unique_name",
        })
    else:
        # multiple real NBA players share this normalized name -- pick the
        # one whose real draft year is closest to this college season ending
        # (handles the namesake-collision risk explicitly rather than
        # silently taking the first row).
        candidates = candidates.copy()
        candidates["year_gap"] = (candidates["DRAFT_YEAR"] - row["year"]).abs()
        best = candidates.sort_values("year_gap").iloc[0]
        if best["year_gap"] > 2:
            ambiguous.append((row["player"], row["year"], candidates[["DRAFT_YEAR"]].values.tolist()))
            continue
        matches.append({
            "college_player": row["player"], "college_last_year": row["year"],
            "PLAYER_ID": best["PERSON_ID"], "real_draft_year": best["DRAFT_YEAR"],
            "real_draft_number": best["DRAFT_NUMBER_NUM"], "torvik_pick": row["pick"],
            "match_method": f"namesake_resolved ({len(candidates)} candidates)",
        })

result = pd.DataFrame(matches)
# safety net: grouping by raw name (instead of normalized name) means two
# real season-rows with slightly different raw spellings of the SAME real
# person (e.g. "PJ Washington" one year, "P.J. Washington" another) could in
# principle both independently match the same bio PLAYER_ID -- dedupe,
# keeping the match whose college season is closest to that player's real
# draft year (most likely their actual final college season).
before = len(result)
result["_gap"] = (result["real_draft_year"] - result["college_last_year"]).abs()
result = result.sort_values("_gap").drop_duplicates(subset=["PLAYER_ID"], keep="first").drop(columns="_gap")
if before != len(result):
    print(f"Deduped {before - len(result)} duplicate PLAYER_ID matches (same real person, multiple raw-name spellings)")
print(f"Matched: {len(result)}  Unmatched: {len(unmatched)}  Ambiguous (skipped, >2yr draft-year gap): {len(ambiguous)}")

# Validation: where torvik recorded its own 'pick' field, does it agree with
# the REAL nba_api draft number for the player we matched to?
val = result.dropna(subset=["torvik_pick", "real_draft_number"])
val["agrees"] = (val["torvik_pick"] - val["real_draft_number"]).abs() <= 1
print(f"\nCross-validation against torvik's own recorded pick (n={len(val)}): "
      f"{val['agrees'].mean():.1%} agree within 1 slot of the real NBA draft number")
print(val[~val["agrees"]][["college_player", "torvik_pick", "real_draft_number", "match_method"]].head(10))

result.to_csv(ROOT / "data" / "college_to_nba_links.csv", index=False)
