"""Did preseason dynasty ranks HOLD? (Tommy's better target: how the same players are ranked later.)

A dynasty rank is a claim about future value, so the right check is whether the market's own later verdict confirms it.
If a youth premium were systematically too high, young players' ranks should DRIFT DOWN over time; if it's right (or too
low), they should hold or rise. Windows with real published lists:
  A: FantraxHQ preseason 2023-24 (top 100)  ->  RotoWire Aug 2025 (top 100)         [2 years later, different outlet]
  B: RotoWire Aug 2025 (top 100)            ->  Hashtag crowd Sept 2026 (top 200)   [1 year later, different outlet]
Different outlets => level differences, so we compare GROUPS (young vs rest) and use percentile-style rank change;
players missing from the later list are assigned just past its cutoff (censored, so drops are if anything UNDERSTATED).
"""
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
sys.argv = ["x"]
src = open(Path(__file__).resolve().parent / "expert_premium_check.py", encoding="utf-8").read().split("sb = pd.read_csv")[0]
exec(src)  # nn_, ROTO_2025, FANTRAX_2023

bio = pd.read_csv(D / "player_bio.csv")
bio["norm"] = (bio["PLAYER_FIRST_NAME"] + " " + bio["PLAYER_LAST_NAME"]).apply(nn_)
draft = bio.drop_duplicates("norm").set_index("norm")["DRAFT_YEAR"]

roto25 = pd.DataFrame([(i + 1, t.rsplit(" ", 1)[0], int(t.rsplit(" ", 1)[1])) for i, t in enumerate(ROTO_2025.split("|"))], columns=["rank", "player", "age"])
roto25["norm"] = roto25["player"].apply(nn_)
fan23 = pd.DataFrame([(i + 1, t) for i, t in enumerate(FANTRAX_2023.split("|"))], columns=["rank", "player"])
fan23["norm"] = fan23["player"].apply(nn_)
fan23["young"] = fan23["norm"].map(lambda n: (draft.get(n, 0) or 0) >= 2020)  # drafted 2020 or later = still developing in 2023
hash26 = pd.read_csv(D / "hashtag_dynasty_2026-09-24.csv")
hash26["norm"] = hash26["player"].apply(nn_)


def drift(earlier, later, later_cut, label, young_col):
    m = earlier.merge(later[["norm", "rank"]].rename(columns={"rank": "later"}), on="norm", how="left")
    m["dropped"] = m["later"].isna()
    m["later_f"] = m["later"].fillna(later_cut + 1)
    n0 = len(earlier)
    m["chg"] = (m["later_f"] - m["rank"])  # + = value fell (rank number rose)
    print(f"\n=== {label}: {n0} players; Spearman(earlier rank, later rank incl. dropped) = {spearmanr(m['rank'], m['later_f'])[0]:.3f}")
    for name, sub in [("young / still developing", m[m[young_col]]), ("established", m[~m[young_col]])]:
        held = ((sub["later_f"] <= sub["rank"] + 5)).mean()
        print(f"  {name:26s} n={len(sub):3d}  avg start rank {sub['rank'].mean():5.1f} -> later {sub['later_f'].mean():5.1f}   "
              f"median change {sub['chg'].median():+5.1f}   held or improved (within 5 spots): {held:.0%}   dropped off the later list: {sub['dropped'].mean():.0%}")
    return m


m1 = drift(fan23, roto25.rename(columns={"rank": "rank"})[["norm", "rank"]], 100, "A) FantraxHQ 2023 preseason -> RotoWire Aug 2025 (2 yrs later)", "young")
top = m1[m1["young"] & (m1["rank"] <= 100)].sort_values("rank")
print("   young players, start -> later:", ", ".join(f"{r.player} {int(r.rank)}->{'out' if r.dropped else int(r.later)}" for r in top.itertuples()))
roto25["young"] = roto25["age"] <= 22
m2 = drift(roto25, hash26[["norm", "rank"]], 200, "B) RotoWire Aug 2025 -> Hashtag Sept 2026 (1 yr later)", "young")
top = m2[m2["young"]].sort_values("rank")
print("   young players, start -> later:", ", ".join(f"{r.player} {int(r.rank)}->{'out' if r.dropped else int(r.later)}" for r in top.itertuples()))
# the very top prospects of each class specifically
print("\nTop prospects (the picks experts were most bullish on):")
for label, names in [("2023 class (Fantrax 2023 -> RotoWire 2025)", ["Victor Wembanyama", "Scoot Henderson", "Amen Thompson", "Ausar Thompson", "Brandon Miller", "Jalen Duren"]),
                     ("2025 class (RotoWire 2025 -> Hashtag 2026)", ["Cooper Flagg", "Dylan Harper", "Ace Bailey", "VJ Edgecombe", "Kon Knueppel", "Zaccharie Risacher"])]:
    mm = m1 if "2023" in label else m2
    print("  " + label + ": " + ", ".join(f"{r.player} {int(r.rank)}->{'out' if r.dropped else int(r.later)}" for r in mm[mm["norm"].isin([nn_(n) for n in names])].sort_values('rank').itertuples()))
