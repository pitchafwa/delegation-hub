"""Real post-mortem: where did our pre-draft talent read diverge most from
actual draft position (our best proxy for "consensus"), and which of those
divergences paid off vs. missed? Uses only the resolved (real_draft_year
<=2018, real outcome known) population -- no speculation on unresolved
recent classes.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT.parent.parent / "dashboard" / "rookie_scores.json").read_text(encoding="utf-8"))
scored = pd.DataFrame(d["scored"])
resolved = scored[scored["actual_fantasy_pctl"].notna()].copy()
print(f"Resolved population: {len(resolved)} players, classes {resolved['Year'].min()}-{resolved['Year'].max()}")

# real draft-position percentile (within this same population) as the
# "consensus" proxy -- lower pick number = higher consensus percentile
resolved["pick_for_rank"] = resolved["pick"].where(resolved["pick"] < 999, 61)
resolved["consensus_pctl"] = 1 - (resolved["pick_for_rank"].rank(pct=True))
resolved["model_vs_consensus"] = resolved["rpi_pre"] - resolved["consensus_pctl"]

print("\n=== TOP 15: model liked them MUCH more than consensus did ===")
liked_more = resolved.sort_values("model_vs_consensus", ascending=False).head(15)
print(liked_more[["Player", "Year", "pick", "rpi_pre", "consensus_pctl", "model_vs_consensus", "actual_fantasy_pctl"]].to_string(index=False))

print("\n=== TOP 15: model liked them MUCH LESS than consensus did ===")
liked_less = resolved.sort_values("model_vs_consensus", ascending=True).head(15)
print(liked_less[["Player", "Year", "pick", "rpi_pre", "consensus_pctl", "model_vs_consensus", "actual_fantasy_pctl"]].to_string(index=False))
