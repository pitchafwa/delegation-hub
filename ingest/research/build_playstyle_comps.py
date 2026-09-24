"""Build stylistic ("what do they look like, what do they do on the
court") player comps, separate from the existing prospect-profile comps.
Real, disclosed limitation: "transition scorer" has no proxy in this data
-- torvik/BBR stats are season aggregates, not play-type-tagged, so it's
left out rather than faked with a weak signal.

College players get the full, rich version (torvik shot-location splits,
DBPM, "stops", usage-adjusted assist rate). International players get a
cruder version (basic box-score rates only, re-derived from the raw pull
since the final feature file didn't keep shot-volume columns) -- real,
disclosed, not hidden, same discipline as the rest of the international
pathway.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "rookie_model_dataset_unified.csv")

# ---------------------------------------------------------------------------
# Real, universal height/weight/position from player_bio.csv (covers both
# pathways; torvik's "inches" and combine WEIGHT are patchy/college-only or
# combine-only, this is close to 100% real coverage for any real NBA player).
# ---------------------------------------------------------------------------
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")


def parse_height(h):
    if not isinstance(h, str) or "-" not in h:
        return np.nan
    try:
        ft, inch = h.split("-")
        return int(ft) * 12 + int(inch)
    except ValueError:
        return np.nan


bio["height_in"] = bio["HEIGHT"].apply(parse_height)
bio_cols = bio[["PERSON_ID", "height_in", "WEIGHT", "POSITION"]].rename(
    columns={"PERSON_ID": "PLAYER_ID", "WEIGHT": "bio_weight", "POSITION": "bio_position"}
)
df = df.merge(bio_cols, on="PLAYER_ID", how="left")
print(f"Real height coverage: {df['height_in'].notna().sum()} / {len(df)}")
print(f"Real bio weight coverage: {df['bio_weight'].notna().sum()} / {len(df)}")

# ---------------------------------------------------------------------------
# Re-derive real shot-volume rates for international rows from the raw pull
# (the final features file only kept shooting PERCENTAGES, not attempts)
# ---------------------------------------------------------------------------
raw_intl = pd.read_csv(ROOT / "data" / "international_player_seasons.csv")
for c in ["FGA", "3PA", "2PA", "FTA"]:
    raw_intl[c] = pd.to_numeric(raw_intl[c], errors="coerce")
raw_intl["_is_final_year"] = raw_intl.groupby("PERSON_ID")["season_end_year"].transform("max") == raw_intl["season_end_year"]
final_intl = raw_intl[raw_intl["_is_final_year"]].sort_values("G", ascending=False).drop_duplicates(subset=["PERSON_ID"], keep="first")
intl_shots = final_intl[["PERSON_ID", "FGA", "3PA", "2PA"]].rename(columns={"PERSON_ID": "PLAYER_ID"})
df = df.merge(intl_shots, on="PLAYER_ID", how="left")

# ---------------------------------------------------------------------------
# Build the real, unified stylistic feature set
# ---------------------------------------------------------------------------
c_mask = df["data_source"] == "college"
i_mask = df["data_source"] == "international"

# lankiness: real wingspan minus real height (bigger = longer/lankier for
# their size, smaller/negative = bulkier/compact) -- only meaningful where
# real combine wingspan exists (both pathways, whoever tested)
df["lankiness"] = df["WINGSPAN"] - df["height_in"]

# floor spacing: 3PA rate. College: real three_a/fga. International:
# re-derived 3PA/FGA from the raw pull.
df["three_rate"] = np.nan
df.loc[c_mask, "three_rate"] = df.loc[c_mask, "three_a"] / df.loc[c_mask, "fga"].replace(0, np.nan)
df.loc[i_mask, "three_rate"] = df.loc[i_mask, "3PA"] / df.loc[i_mask, "FGA"].replace(0, np.nan)

# rim pressure / paint scoring: college only (real shot-location data);
# international has no equivalent split -- left NaN, a real disclosed gap
df["rim_rate"] = np.nan
df.loc[c_mask, "rim_rate"] = df.loc[c_mask, "rim_a"] / df.loc[c_mask, "fga"].replace(0, np.nan)

# playmaking: usage-adjusted assist rate + ball security. College: real
# ast/usg/ast_to. International: cruder per-game AST and AST/TOV from the
# basic box score (already have apg/tov from the earlier merge's renamed
# columns).
df["playmaking"] = np.nan
df.loc[c_mask, "playmaking"] = df.loc[c_mask, "ast"] * (df.loc[c_mask, "usg"] / 20.0)
df.loc[i_mask, "playmaking"] = df.loc[i_mask, "apg"] / (df.loc[i_mask, "tov"].replace(0, np.nan))

# defense: college real DBPM + "stops"; international only has basic
# steal+block rate (no real defensive advanced metric available)
df["defense_score"] = np.nan
df.loc[c_mask, "defense_score"] = df.loc[c_mask, "dbpm"]
df.loc[i_mask, "defense_score"] = (df.loc[i_mask, "spg"].fillna(0) + df.loc[i_mask, "bpg"].fillna(0))

# rebounding profile: college real oreb_rate/dreb_rate; international per-game rpg only
df["rebounding"] = np.nan
df.loc[c_mask, "rebounding"] = df.loc[c_mask, "oreb_rate"] + df.loc[c_mask, "dreb_rate"]
df.loc[i_mask, "rebounding"] = df.loc[i_mask, "rpg"]

STYLE_W = {
    "height_in": 2.0, "bio_weight": 1.6, "lankiness": 1.4,
    "three_rate": 1.6, "rim_rate": 1.3, "playmaking": 1.3, "defense_score": 1.2, "rebounding": 0.9,
}
_M = pd.DataFrame(index=df.index)
for c, wgt in STYLE_W.items():
    v = pd.to_numeric(df[c], errors="coerce")
    v = v.fillna(v.median())
    z = (v - v.mean()) / (v.std() + 1e-9)
    _M[c] = z * wgt
_Marr = _M.to_numpy()

players_arr = df["player"].to_numpy()
years_arr = df["real_draft_year"].to_numpy()
sources_arr = df["data_source"].to_numpy()
style_col = []
for i in range(len(_Marr)):
    dist = np.sqrt(((_Marr - _Marr[i]) ** 2).sum(1))
    order = np.argsort(dist)
    top = [j for j in order if j != i][:6]
    scale = np.median(dist[dist > 0]) if (dist > 0).any() else 1.0
    style_col.append([
        {"p": str(players_arr[j]), "y": int(years_arr[j]) if pd.notna(years_arr[j]) else None,
         "sim": round(float(100 * np.exp(-dist[j] / scale)), 0),
         "intl": sources_arr[j] == "international"}
        for j in top
    ])
df["style_similar"] = style_col

# NOTE: this script is for validating the approach only -- it does NOT
# overwrite rookie_model_dataset_unified.csv (a list-of-dicts column would
# serialize to a string through CSV and break other scripts that read that
# file expecting scalar columns). The validated logic gets ported directly
# into build_dashboard_data_unified.py instead, computed fresh in-memory.

# real face-validity spot check: known player archetypes should comp sensibly
for name in ["Victor Wembanyama", "Chet Holmgren", "Ja Morant" if "Ja Morant" in df["player"].values else "Trae Young"]:
    row = df[df["player"] == name]
    if not row.empty:
        print(f"\n{name} ({row.iloc[0]['pos']}, {row.iloc[0]['height_in']}in, three_rate={row.iloc[0]['three_rate']:.2f}, rim_rate={row.iloc[0]['rim_rate']}) style comps:")
        for s in row.iloc[0]["style_similar"]:
            print(f"   {s}")
    else:
        print(f"\n{name}: not found")
