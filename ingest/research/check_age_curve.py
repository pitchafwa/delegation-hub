import pandas as pd
df = pd.read_csv("research/data/valuation_dataset.csv")
df = df[df["FANTASY_PPG"] > 5]  # avoid divide-by-near-zero noise
df["GROWTH_RATIO"] = df["TARGET_FANTASY_PPG"] / df["FANTASY_PPG"]

by_age = df.groupby("AGE").agg(
    n=("GROWTH_RATIO", "size"),
    mean_growth_ratio=("GROWTH_RATIO", "mean"),
    median_growth_ratio=("GROWTH_RATIO", "median"),
    mean_current_ppg=("FANTASY_PPG", "mean"),
).reset_index()
by_age = by_age[by_age["n"] >= 15]
print(by_age.to_string(index=False))
