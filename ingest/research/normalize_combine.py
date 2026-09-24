"""Position-normalize combine measurements to percentiles within broad
position groups, per the real research finding: raw combine numbers aren't
comparable across positions (a 7'0" wingspan means something different for a
center than a guard), and size measurements specifically carry more real
signal than raw athleticism testing once properly normalized.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "rookie_model_dataset.csv")

POSITION_GROUP = {
    "Pure PG": "Guard", "Scoring PG": "Guard", "Combo G": "Guard", "Wing G": "Guard",
    "Wing F": "Forward", "Stretch 4": "Forward",
    "PF/C": "Big", "C": "Big",
}
df["pos_group"] = df["pos"].map(POSITION_GROUP)

COMBINE_COLS = ["WINGSPAN", "STANDING_REACH", "STANDING_VERTICAL_LEAP", "MAX_VERTICAL_LEAP",
                 "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"]
LOWER_IS_BETTER = {"LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"}  # times -- faster is better

for col in COMBINE_COLS:
    pct_col = f"{col}_PCTILE"
    ranks = df.groupby("pos_group")[col].rank(pct=True, ascending=(col not in LOWER_IS_BETTER))
    # pandas groupby drops NaN group keys entirely, so any row with no real
    # position (torvik's archived 2008-2009 seasons have ZERO position
    # labels at all -- a genuine, disclosed source limitation, confirmed by
    # checking the raw archive files) got a NaN percentile even when the
    # real combine measurement itself was present -- silently discarding
    # real signal for ~74 players. Caught via a real user question ("did
    # they genuinely not participate?"). Fall back to an overall (not
    # position-normalized) percentile for those rows instead of dropping
    # them -- less precise, but real signal beats none.
    overall_rank = df[col].rank(pct=True, ascending=(col not in LOWER_IS_BETTER))
    df[pct_col] = ranks.fillna(overall_rank)

df.to_csv(ROOT / "data" / "rookie_model_dataset.csv", index=False)

from scipy.stats import spearmanr
print("=== Position-normalized combine percentiles vs both targets ===")
for col in COMBINE_COLS:
    pct_col = f"{col}_PCTILE"
    for target in ["age_22_29_best3", "rookie_PTS_per_min"]:
        sub = df.dropna(subset=[pct_col, target])
        if len(sub) < 20:
            continue
        rho, _ = spearmanr(sub[pct_col], sub[target])
        print(f"{col:24s} vs {target:20s} rho={rho:+.3f}  n={len(sub)}")
