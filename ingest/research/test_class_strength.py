"""Tommy's idea: NBA draft classes vary in real strength year to year (2026
is widely regarded as elite; 2024 as weak -- Cameron Boozer going 3rd in
2026 would have gone 1st/2nd in many other years). Test whether a real,
non-circular "class strength" signal -- built from OUR OWN pre-draft talent
model, leave-one-out so a player's own score never inflates their own
class's strength reading -- helps the post-draft model beyond pick number
alone.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from fit_rookie_model_a2 import score, build_bounds

ROOT = Path(__file__).resolve().parent
frozen = json.loads((ROOT / "data" / "output_a_model.frozen.json").read_text())
PRE = frozen["pre_draft_model"]

# full, UNFILTERED population (fit_rookie_model_a2's own df is restricted to
# <=2018 at import time -- need every class, including 2019-2026, to read
# their real class strength even though only <=2018 has a resolved outcome
# to validate against).
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")
EXP_MAP = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}
df["exp_numeric"] = df["exp"].map(EXP_MAP)
df["exp_numeric"] = df["exp_numeric"].fillna(df["exp_numeric"].median())
df["rec_filled"] = df["rec"].fillna(0)
df["draft_age_filled"] = df["draft_age"].fillna(df["draft_age"].median())
df["LANE_AGILITY_TIME_PCTILE"] = df["LANE_AGILITY_TIME_PCTILE"].fillna(df["LANE_AGILITY_TIME_PCTILE"].median())

pre_score_all = score(PRE["params"], df, PRE["features"])
df["pre_draft_score_raw"] = pre_score_all

# leave-one-out class strength: for player i in class Y, average the
# PRE-DRAFT score of every OTHER player in class Y (top-14 picks only, and
# separately the whole real-drafted class), so a player's own talent score
# never contributes to their own class-strength reading.
df["is_lottery"] = df["real_draft_number"].notna() & (df["real_draft_number"] <= 14)
df["is_drafted"] = df["real_draft_number"].notna() & (df["real_draft_number"] <= 60)

def loo_class_strength(data, pool_mask, col_name):
    out = pd.Series(np.nan, index=data.index)
    for year, grp in data[pool_mask].groupby("real_draft_year"):
        n = len(grp)
        if n < 3:
            continue
        total = grp["pre_draft_score_raw"].sum()
        loo_avg = (total - grp["pre_draft_score_raw"]) / (n - 1)
        out.loc[grp.index] = loo_avg
    data[col_name] = out

loo_class_strength(df, df["is_lottery"], "class_strength_lottery_loo")
loo_class_strength(df, df["is_drafted"], "class_strength_overall_loo")

# fill for players outside the pool (can't compute LOO for them) with the
# real, non-LOO class average so every row still gets a value
for col, pool in [("class_strength_lottery_loo", df["is_lottery"]), ("class_strength_overall_loo", df["is_drafted"])]:
    class_avg = df[pool].groupby("real_draft_year")["pre_draft_score_raw"].transform("mean")
    df.loc[pool, col] = df.loc[pool, col].fillna(class_avg)
    overall_median = df[col].median()
    df[col] = df[col].fillna(overall_median)

print("Real class strength by year (lottery LOO avg, overall LOO avg):")
show = df[df["is_drafted"]].groupby("real_draft_year")[["class_strength_lottery_loo", "class_strength_overall_loo"]].mean()
print(show.to_string())

df.to_csv(ROOT / "data" / "rookie_model_dataset_with_class_strength.csv", index=False)
print(f"\nSaved with class-strength columns to rookie_model_dataset_with_class_strength.csv")
