"""Merge real birthdates into the rookie dataset and compute precise
draft-day age -- a top predictor in real literature, only crudely
represented so far via the Fr/So/Jr/Sr categorical.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")
df = df.drop(columns=[c for c in ["BIRTHDATE", "draft_age"] if c in df.columns])
bd = pd.read_csv(ROOT / "data" / "birthdates_all.csv")
bd["BIRTHDATE"] = pd.to_datetime(bd["BIRTHDATE"])
bd = bd.rename(columns={"PERSON_ID": "PLAYER_ID"}).drop_duplicates(subset=["PLAYER_ID"])

df = df.merge(bd, on="PLAYER_ID", how="left")
print(f"Matched birthdates: {df['BIRTHDATE'].notna().sum()} / {len(df)}")

# draft day = June 26 of real_draft_year (real NBA draft date, close enough
# across years for age-in-years granularity)
draft_day = pd.to_datetime(df["real_draft_year"].astype("Int64").astype(str) + "-06-26", errors="coerce")
df["draft_age"] = (draft_day - df["BIRTHDATE"]).dt.days / 365.25

before = df["draft_age"].notna().sum()
print(f"Real draft_age computed for {before} / {len(df)} rows")

# 5 rows came back with physically impossible ages (e.g. -24yo, 27-29yo) --
# a real real_draft_year data-quality issue for those specific rows (their
# birthdates matched correctly, so this isn't a namesake ID collision).
# Null these out rather than let bad joins corrupt the feature; too small a
# fraction (5/1093 = 0.46%) to investigate the root cause further right now.
implausible = ~df["draft_age"].between(17, 26)
print(f"Nulling {implausible.sum()} implausible draft_age rows (outside 17-26yo)")
df.loc[implausible, "draft_age"] = pd.NA

print(df["draft_age"].describe())

df.to_csv(ROOT / "data" / "rookie_model_dataset.csv", index=False)
print("Saved with draft_age column added.")
