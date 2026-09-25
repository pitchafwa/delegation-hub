"""Panel for the usage-flow study: what happens to teammates' minutes and fantasy points when a rotation player is out?

Source: regular_season_box_scores_2010_2024 (every roster player for every game, INCLUDING did-not-play rows with the reason: coach's decision, injury/illness, not with
team, did not dress, ...; starters carry a position).  14 seasons (2010-11 to 2023-24), ~430k player-games.
Per player-game we compute (all from games BEFORE that game, so nothing leaks):
  mp15 / fp15 / usg15   minutes, league-scored fantasy points and usage proxy per game over his previous 15 games played
  st15                  share of those games he started
An ABSENCE is a roster player who did not play for a reason other than "coach's decision" (injury, illness, rest, personal, suspension, not with team, did not dress)
and whose mp15 is 12+ (a rotation player).  Team-game totals of what is vacated (minutes, fantasy points, usage) are attached to every teammate who played.
Each teammate also gets a FULL-STRENGTH baseline: his mean fantasy points / minutes over his previous 40 games in which no rotation teammate was absent (needs 5+ such games,
else his previous-15 mean).
Output: data/usage_flow_panel.pkl (teammate-game rows) and data/usage_flow_absences.pkl (absence rows).
Run from ingest/:  uv run python research/usage_flow_panel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
fs = [D / "game_logs" / f"regular_season_box_scores_2010_2024_part_{i}.csv" for i in (1, 2, 3)]
cols = ["season_year", "game_date", "gameId", "teamTricode", "personId", "personName", "position", "comment", "minutes", "points", "reboundsTotal", "assists", "steals", "blocks",
        "turnovers", "threePointersMade", "freeThrowsMade", "freeThrowsAttempted", "fieldGoalsAttempted"]
d = pd.concat([pd.read_csv(f, usecols=cols) for f in fs], ignore_index=True).drop_duplicates(["gameId", "personId"])


def mins(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, str) and ":" in x:
        a, b = x.split(":")
        return float(a) + float(b) / 60
    try:
        return float(x)
    except Exception:
        return 0.0


d["min"] = d.minutes.map(mins)
d["played"] = d["min"] > 0
for c in ["points", "reboundsTotal", "assists", "steals", "blocks", "turnovers", "threePointersMade", "freeThrowsMade", "freeThrowsAttempted", "fieldGoalsAttempted"]:
    d[c] = d[c].fillna(0)
cats = (d[["points", "reboundsTotal", "assists", "steals", "blocks"]] >= 10).sum(axis=1)
d["fp"] = (d.points + 1.5 * d.reboundsTotal + 2 * d.assists + 3 * d.steals + 3 * d.blocks + d.threePointersMade + 2 * d.freeThrowsMade - d.freeThrowsAttempted - d.turnovers + 3 * (cats >= 3)).where(d.played, 0.0)
d["usg"] = (d.fieldGoalsAttempted + 0.44 * d.freeThrowsAttempted + d.turnovers).where(d.played, 0.0)
d["gd"] = pd.to_datetime(d.game_date)
d["start"] = d.position.notna().astype(float)
cm = d.comment.fillna("").str.upper()
d["coach"] = cm.str.contains("COACH")
d["reason"] = np.where(~d.played & ~d.coach & (d.comment.notna()), np.where(cm.str.contains("INJUR|ILLNESS|ANKLE|KNEE|BACK|HAMSTRING|FOOT|CALF|SORE|SPRAIN|STRAIN|HEALTH"), "injury", np.where(cm.str.contains("REST"), "rest", np.where(cm.str.contains("PERSONAL|SUSPENS"), "personal", "other"))), "")
d = d.sort_values(["personId", "gd", "gameId"]).reset_index(drop=True)

# ---- rolling baselines from previous 15 GAMES PLAYED (across seasons)
pl = d[d.played].copy()
g = pl.groupby("personId")
for c, nm in (("min", "mp15"), ("fp", "fp15"), ("usg", "usg15"), ("start", "st15")):
    pl[nm] = g[c].transform(lambda s: s.shift(1).rolling(15, min_periods=5).mean())
base = pl[["personId", "gd", "gameId", "mp15", "fp15", "usg15", "st15"]]
# for every row (played or not) attach the baseline as of the most recent played game BEFORE this one
pl2 = pl[["personId", "gd", "min", "fp", "usg", "start"]].copy()
pl2["mp15n"] = g["min"].transform(lambda s: s.rolling(15, min_periods=5).mean())      # includes his own game: as of the END of that game
pl2["fp15n"] = g["fp"].transform(lambda s: s.rolling(15, min_periods=5).mean())
pl2["usg15n"] = g["usg"].transform(lambda s: s.rolling(15, min_periods=5).mean())
pl2["st15n"] = g["start"].transform(lambda s: s.rolling(15, min_periods=5).mean())
asof = pl2[["personId", "gd", "mp15n", "fp15n", "usg15n", "st15n"]].sort_values("gd")
d = pd.merge_asof(d.sort_values("gd"), asof, on="gd", by="personId", direction="backward", allow_exact_matches=False)
d = d.rename(columns={"mp15n": "mp15", "fp15n": "fp15", "usg15n": "usg15", "st15n": "st15"}).sort_values(["personId", "gd", "gameId"]).reset_index(drop=True)
print(f"{len(d):,} player-games, {d.played.mean():.3f} played; absence reasons among non-played:", d[~d.played].reason.value_counts().to_dict())

# ---- absences (rotation players who missed for a non-coach reason)
# The box-score comment is unreliable for WHY (the share labelled "coach's decision" for players averaging 12+ minutes is far above the injury-labelled share and the
# labels drift over the years), and for teammates the reason does not matter: a rotation player who does not play frees his minutes either way. So every rotation
# player who did not play counts; `known` marks the ones labelled injury/illness/rest/personal (the kind a report announces in advance).
ab = d[(~d.played) & (d.mp15 >= 12)].copy()
ab["known"] = (ab.reason != "").astype(float)
tg = ab.groupby(["gameId", "teamTricode"]).agg(V_min=("mp15", "sum"), V_fp=("fp15", "sum"), V_usg=("usg15", "sum"), n_abs=("personId", "size"), V_start=("st15", "sum"), V_known=("known", "sum")).reset_index()
print(f"{len(ab):,} rotation absences in {len(tg):,} team-games (of {d.groupby(['gameId', 'teamTricode']).ngroups:,})")

# ---- teammate rows
t = d[d.played & (d.mp15 >= 6)].merge(tg, on=["gameId", "teamTricode"], how="left")
for c in ("V_min", "V_fp", "V_usg", "n_abs", "V_start", "V_known"):
    t[c] = t[c].fillna(0.0)
t = t.sort_values(["personId", "gd", "gameId"]).reset_index(drop=True)
# full-strength baseline: mean over the previous 40 games (played) with NO rotation teammate absent, needs 5+; else previous-15 mean
gg = t.groupby("personId")
full = (t.V_min == 0)
t["fp_full"] = t.fp.where(full)
t["min_full"] = t["min"].where(full)
t["_n"] = full.astype(float)
num_fp = gg.fp_full.transform(lambda s: s.shift(1).rolling(40, min_periods=1).sum())
num_mn = gg.min_full.transform(lambda s: s.shift(1).rolling(40, min_periods=1).sum())
cnt = gg["_n"].transform(lambda s: s.shift(1).rolling(40, min_periods=1).sum())
t["b_fp"] = np.where(cnt >= 5, num_fp / cnt.replace(0, np.nan), t.fp15)
t["b_min"] = np.where(cnt >= 5, num_mn / cnt.replace(0, np.nan), t.mp15)
t["full_n"] = cnt
# share of the active team's baseline production this player represents
tot = t.groupby(["gameId", "teamTricode"]).b_fp.transform("sum")
t["w"] = t.b_fp / tot
tm = t.groupby(["gameId", "teamTricode"]).b_min.transform("sum")
t["wm"] = t.b_min / tm
t = t.dropna(subset=["b_fp", "b_min"])
t.to_pickle(D / "usage_flow_panel.pkl")
ab.to_pickle(D / "usage_flow_absences.pkl")
print(f"{len(t):,} teammate-game rows; {(t.V_min > 0).mean():.3f} have a rotation teammate absent; mean vacated minutes when any: {t[t.V_min > 0].V_min.mean():.1f}")
