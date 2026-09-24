import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.argv = [sys.argv[0]]
from fit_stat_kalman_qualityweighted import objective, build_actual, HORIZONS, prep_horizon
import numpy as np

stat_col = "PTS"
is_direct = False
import pandas as pd
df = pd.read_csv("data/kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID","GAME_DATE"]).reset_index(drop=True)
raw_col = df[stat_col].to_numpy(dtype=float)
full_min = df["MIN"].to_numpy(dtype=float)
x0 = float(raw_col.sum()) / max(float(full_min.sum()),1.0)

preps_local = [prep_horizon(h) for h in HORIZONS]
for prep in preps_local:
    prep["actual"] = build_actual(prep["target_df"], stat_col, is_direct)

# fitted optimum for reference
fitted = [0.0, 31.9685, 25.4096, -0.047, -0.0678]
Q, R, peak_age, _, slope_down = fitted

print(f"PTS fitted: Q={Q} R={R} peak_age={peak_age} slope_down={slope_down}\n")
print("slope_up sweep (all else fixed at fitted optimum):")
for su in [-0.08, -0.05, -0.02, 0.0, 0.02, 0.05, 0.08]:
    params = [Q, R, peak_age, su, slope_down]
    score = -objective(params, preps_local, stat_col, is_direct, x0)
    print(f"  slope_up={su:+.3f}  mean multi-year Spearman={score:.4f}")
