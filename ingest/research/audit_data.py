"""Data audit -- run this on every source refresh, per the WRPI/RUPI lesson
that silent upstream truncation (40-75% of rows dropped in three files for
months, disproportionately hitting RBs, Saquon Barkley entirely missing) went
undetected until someone deliberately checked row counts and known-entity
presence. Cheap to run, catches expensive silent corruption.
"""
import re
import unicodedata
from pathlib import Path
import pandas as pd


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name, flags=re.I)
    name = re.sub(r"[^a-z ]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()

ROOT = Path(__file__).resolve().parent
base = pd.read_csv(ROOT / "data" / "player_season_base.csv")
adv = pd.read_csv(ROOT / "data" / "player_season_advanced.csv")
bio = pd.read_csv(ROOT / "data" / "player_bio.csv")

print("=== Row counts per season (Base) -- expect roughly 450-580/season ===")
counts = base.groupby("SEASON").size()
print(counts.to_string())
suspicious = counts[(counts < 400) | (counts > 650)]
if len(suspicious):
    print(f"\n!!! SUSPICIOUS row counts: {suspicious.to_dict()}")
else:
    print("\nOK -- no season outside the expected range.")

print("\n=== Fill rate for key columns ===")
for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3", "AGE", "GP"]:
    null_pct = base[col].isna().mean()
    flag = "  <-- NONZERO NULLS" if null_pct > 0 else ""
    print(f"{col:8s} null rate: {null_pct:.4%}{flag}")

adv_key_cols = ["USG_PCT", "TS_PCT", "PACE", "PIE"]
for col in adv_key_cols:
    null_pct = adv[col].isna().mean()
    flag = "  <-- NONZERO NULLS" if null_pct > 0 else ""
    print(f"{col:8s} null rate: {null_pct:.4%}{flag}")

print("\n=== Known must-be-present stars, spot-checked by season ===")
must_be_present = {
    "2015-16": ["Stephen Curry", "LeBron James"],
    "2018-19": ["Giannis Antetokounmpo", "James Harden"],
    "2022-23": ["Nikola Jokic", "Joel Embiid"],
    "2024-25": ["Nikola Jokic", "Shai Gilgeous-Alexander"],
}
all_ok = True
for season, names in must_be_present.items():
    present = {normalize_name(n) for n in base[base["SEASON"] == season]["PLAYER_NAME"]}
    for name in names:
        ok = normalize_name(name) in present
        all_ok &= ok
        print(f"  {season}  {name:28s} {'OK' if ok else '!!! MISSING (real, not just accent/encoding)'}")

print("\n=== Duplicate PLAYER_ID within a season (should be zero -- confirms trade aggregation) ===")
dupe_check = base.groupby("SEASON").apply(lambda d: d["PLAYER_ID"].duplicated().sum())
bad = dupe_check[dupe_check > 0]
print("OK -- no duplicates" if bad.empty else f"!!! DUPLICATES FOUND:\n{bad}")

print("\n=== Bio table sanity ===")
print(f"Total players in bio: {len(bio)}")
print(f"Players missing DRAFT_YEAR entirely (expected for old undrafted/ABA-era): {bio['DRAFT_YEAR'].isna().mean():.2%}")

print("\n=== FTMI sanity: FTA should always be >= FTM (missed free throws can't be negative) ===")
bad_ft = base[base["FTA"] < base["FTM"]]
print("OK" if bad_ft.empty else f"!!! {len(bad_ft)} rows with FTA < FTM:\n{bad_ft[['PLAYER_NAME','SEASON','FTA','FTM']]}")

print(f"\n{'PASSED' if all_ok and suspicious.empty and bad.empty and bad_ft.empty else 'FAILED -- investigate flagged items above'}")
