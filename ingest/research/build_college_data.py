"""Build one unified, consistently-named college basketball dataset,
2008-2025 (18 seasons):
- 2008-2023: toRvik's archived, properly-named parquet files (free, GitHub-hosted)
- 2024-2025: barttorvik.com's own live getadvstats.php endpoint (free, no auth),
  field order reverse-engineered and VALIDATED against the 2023 archive for two
  real players with many distinctive non-zero values (see chat) -- not guessed.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "torvik_cache"
CACHE.mkdir(exist_ok=True)

ARCHIVE_YEARS = range(2008, 2024)
LIVE_YEARS = [2024, 2025, 2026]

# Validated by cross-referencing 2023 live vs archived data for two real
# players (Robby Carmody, Antoine Davis) using distinctive non-zero values.
# Index -> archive column name. Positions with no confident match are omitted.
LIVE_FIELD_MAP = {
    0: "player", 1: "team", 2: "conf", 3: "g", 4: "min", 5: "ortg", 6: "usg",
    7: "efg", 8: "ts", 9: "oreb_rate", 10: "dreb_rate", 11: "ast", 12: "to",
    13: "ftm", 14: "fta", 15: "ft_pct", 16: "two_m", 17: "two_a", 18: "two_pct",
    19: "three_m", 20: "three_a", 21: "three_pct", 22: "blk", 23: "stl",
    24: "ftr", 25: "exp", 26: "hgt", 27: "num", 28: "porpag", 29: "adj_oe",
    30: "pfr", 31: "year", 32: "id", 33: "hometown", 34: "rec", 35: "ast_to",
    36: "rim_m", 37: "rim_a", 38: "mid_m", 39: "mid_a", 40: "rim_pct",
    41: "mid_pct", 42: "dunk_m", 43: "dunk_a", 44: "dunk_pct", 46: "drtg",
    47: "adj_de", 48: "dporpag", 49: "stops", 53: "bpm", 54: "mpg",
    55: "obpm", 56: "dbpm", 57: "oreb", 58: "dreb", 59: "rpg", 60: "apg",
    61: "spg", 62: "bpg", 63: "ppg", 64: "pos", 66: "birthdate",
}


def height_to_inches(h):
    if not isinstance(h, str) or "-" not in h:
        return np.nan
    try:
        ft, inch = h.split("-")
        return int(ft) * 12 + int(inch)
    except ValueError:
        return np.nan


frames = []

for year in ARCHIVE_YEARS:
    path = CACHE / f"archive_{year}.parquet"
    if not path.exists():
        url = f"https://github.com/andreweatherman/toRvik-data/raw/main/player_season/{year}/all_{year}.parquet"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        path.write_bytes(r.content)
    frames.append(pd.read_parquet(path))
    print(f"  archive {year} loaded")

for year in LIVE_YEARS:
    path = CACHE / f"live_{year}.json"
    if not path.exists():
        r = requests.get(f"https://barttorvik.com/getadvstats.php?year={year}", timeout=30)
        r.raise_for_status()
        path.write_text(r.text, encoding="utf-8")
        time.sleep(1.0)
    rows = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    df = df.rename(columns={i: name for i, name in LIVE_FIELD_MAP.items()})
    df = df[[c for c in LIVE_FIELD_MAP.values()]]
    df["inches"] = df["hgt"].apply(height_to_inches)
    df["fgm"] = df["two_m"] + df["three_m"]
    df["fga"] = df["two_a"] + df["three_a"]
    df["fg_pct"] = df["fgm"] / df["fga"].replace(0, np.nan)
    for c in ["year", "g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    frames.append(df)
    print(f"  live {year} loaded ({len(df)} players)")

college = pd.concat(frames, ignore_index=True, sort=False)
college.to_csv(ROOT / "data" / "college_player_season_2008_2025.csv", index=False)
print(f"\nTotal: {len(college)} college player-seasons, years {sorted(college['year'].unique())}")
