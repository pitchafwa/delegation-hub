"""Short-term projection program, Phase 1: the as-of panel (RESEARCH_short_term_projection_plan.md).

One row per player per game he played (the "info point" is the morning after that game), 2015-16 .. today, with every feature computed only from
information available that morning, plus a long target table: his team's next 10 games within 14 days (days ahead, played, fantasy points, minutes,
started). Writes data/st_panel.parquet and data/st_targets.parquet.

Sources: box scores 2010-24 (data/game_logs/regular_season_box_scores_*.csv: full lines, starters, did-not-play reasons), NBA game logs 2024-25 on
(data/game_logs/nba_api_*.csv) with starters/reasons from data/game_logs/box_v3_*.csv when pulled (pull_box_v3.py), DELCO's belief after every game
(data/delco_pregame_all.csv.gz: season and career mode), and player ages (data/kalman_input.csv).
Run from ingest/:  .venv/Scripts/python research/st_panel.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
G = D / "game_logs"
FIRST = "2014-15"            # one season of history before the first info points (2015-16)


def mins(x):
    if isinstance(x, str) and ":" in x:
        m, s = x.split(":")[:2]
        return float(m) + float(s) / 60.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# ---------------- unified box scores
parts = []
for f in sorted(G.glob("regular_season_box_scores_2010_2024_part_*.csv")):
    b = pd.read_csv(f, dtype={"gameId": str})
    b = b[b["season_year"] >= FIRST]
    parts.append(pd.DataFrame({
        "pid": b["personId"], "name": b["personName"], "season": b["season_year"], "gid": b["gameId"].str.zfill(10), "date": b["game_date"],
        "team": b["teamTricode"], "min": b["minutes"].map(mins), "pts": b["points"], "reb": b["reboundsTotal"], "ast": b["assists"], "stl": b["steals"],
        "blk": b["blocks"], "tov": b["turnovers"], "fg3m": b["threePointersMade"], "fg3a": b["threePointersAttempted"], "fgm": b["fieldGoalsMade"],
        "fga": b["fieldGoalsAttempted"], "ftm": b["freeThrowsMade"], "fta": b["freeThrowsAttempted"], "pf": b["foulsPersonal"],
        "started": b["position"].notna() & (b["position"].astype(str).str.len() > 0), "comment": b["comment"]}))
for f in sorted(G.glob("nba_api_*.csv")):
    season = f.stem.replace("nba_api_", "")
    b = pd.read_csv(f, dtype={"GAME_ID": str})
    x = pd.DataFrame({
        "pid": b["PLAYER_ID"], "name": b["PLAYER_NAME"], "season": season, "gid": b["GAME_ID"].str.zfill(10), "date": b["GAME_DATE"], "team": b["TEAM_ABBREVIATION"],
        "min": b["MIN"].map(mins), "pts": b["PTS"], "reb": b["REB"], "ast": b["AST"], "stl": b["STL"], "blk": b["BLK"], "tov": b["TOV"], "fg3m": b["FG3M"],
        "fg3a": b["FG3A"], "fgm": b["FGM"], "fga": b["FGA"], "ftm": b["FTM"], "fta": b["FTA"], "pf": b["PF"], "started": np.nan, "comment": np.nan})
    v3 = G / f"box_v3_{season}.csv"
    if v3.exists():
        s3 = pd.read_csv(v3, dtype={"gameId": str}, usecols=["gameId", "personId", "position", "comment", "minutes", "teamTricode", "firstName", "familyName"])
        s3["gid"] = s3["gameId"].str.zfill(10)
        st = s3.set_index(["gid", "personId"])
        key = list(zip(x["gid"], x["pid"]))
        x["started"] = [bool(isinstance(st["position"].get(k), str) and len(st["position"].get(k)) > 0) if k in st.index else np.nan for k in key]
        # did-not-play rows (with reasons) exist only in the box score: add them so availability can be read the same way for every season
        dnp = s3[s3["minutes"].isna() | (s3["minutes"].astype(str).isin(["", "0:00", "00:00"]))]
        if len(dnp):
            dd = x.drop_duplicates("gid").set_index("gid")["date"]
            x = pd.concat([x, pd.DataFrame({"pid": dnp["personId"], "name": dnp["firstName"] + " " + dnp["familyName"], "season": season, "gid": dnp["gid"],
                                           "date": dnp["gid"].map(dd), "team": dnp["teamTricode"], "min": 0.0, "started": False, "comment": dnp["comment"]})], ignore_index=True)
    parts.append(x)
box = pd.concat(parts, ignore_index=True)
box = box[box["date"].notna()].copy()
box["date"] = pd.to_datetime(box["date"])
for c in ["pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "pf"]:
    box[c] = pd.to_numeric(box[c], errors="coerce").fillna(0.0)
dbl = sum((box[c] >= 10).astype(int) for c in ["pts", "reb", "ast", "stl", "blk"])
box["fp"] = box["pts"] + 1.5 * box["reb"] + 2 * box["ast"] + 3 * box["stl"] + 3 * box["blk"] + box["fg3m"] + 2 * box["ftm"] - box["fta"] - box["tov"] + 3 * (dbl >= 3)
box["played"] = box["min"] > 0
box = box.sort_values(["pid", "date"]).drop_duplicates(["pid", "gid"]).reset_index(drop=True)
print(f"box scores: {len(box)} rows ({box['played'].sum()} played), {box['pid'].nunique()} players, seasons {box['season'].min()}..{box['season'].max()}", flush=True)

# team schedule: every team game (date, gid) and a player's team on each date (his last game's team)
tgames = box.drop_duplicates(["team", "gid"])[["team", "gid", "date", "season"]].sort_values(["team", "date"]).reset_index(drop=True)
tgames["tg_idx"] = tgames.groupby("team").cumcount()
tg_key = {(t, g): i for t, g, i in zip(tgames["team"], tgames["gid"], tgames["tg_idx"])}
tg_by_team = {t: d.reset_index(drop=True) for t, d in tgames.groupby("team")}
tg_gid = {t: list(d["gid"]) for t, d in tg_by_team.items()}
tg_date = {t: list(d["date"]) for t, d in tg_by_team.items()}

# ---------------- features after each PLAYED game
P = box[box["played"]].copy().reset_index(drop=True)
gp = P.groupby("pid", sort=False)
for n in (1, 3, 5, 10, 15):
    P[f"fp_l{n}"] = gp["fp"].transform(lambda s: s.rolling(n, min_periods=n).mean())
    P[f"min_l{n}"] = gp["min"].transform(lambda s: s.rolling(n, min_periods=n).mean())
for hl in (2, 4, 8):
    P[f"fp_ew{hl}"] = gp["fp"].transform(lambda s: s.ewm(halflife=hl, min_periods=3).mean())
    P[f"min_ew{hl}"] = gp["min"].transform(lambda s: s.ewm(halflife=hl, min_periods=3).mean())
P["min_sd10"] = gp["min"].transform(lambda s: s.rolling(10, min_periods=5).std())
P["start_l1"] = P["started"].astype(float)
P["start_l5"] = gp["started"].transform(lambda s: s.astype(float).rolling(5, min_periods=3).mean())
P["start_prev10"] = gp["started"].transform(lambda s: s.astype(float).shift(5).rolling(10, min_periods=5).mean())
use = P["fga"] + 0.44 * P["fta"] + P["tov"]
P["use"] = use
P["use_l5"] = gp["use"].transform(lambda s: s.rolling(5, min_periods=5).sum()) / gp["min"].transform(lambda s: s.rolling(5, min_periods=5).sum()).clip(lower=1)
P["use_l40"] = gp["use"].transform(lambda s: s.rolling(40, min_periods=15).sum()) / gp["min"].transform(lambda s: s.rolling(40, min_periods=15).sum()).clip(lower=1)
for a, b_, n in (("fgm", "fga", "fg"), ("fg3m", "fg3a", "fg3"), ("ftm", "fta", "ft")):
    P[f"{n}_l5"] = gp[a].transform(lambda s: s.rolling(5, min_periods=5).sum()) / gp[b_].transform(lambda s: s.rolling(5, min_periods=5).sum()).replace(0, np.nan)
    P[f"{n}_l100"] = gp[a].transform(lambda s: s.rolling(100, min_periods=30).sum()) / gp[b_].transform(lambda s: s.rolling(100, min_periods=30).sum()).replace(0, np.nan)
P["pf_l3"] = gp["pf"].transform(lambda s: s.rolling(3, min_periods=3).mean())
P["rate_l5"] = gp["fp"].transform(lambda s: s.rolling(5, min_periods=5).sum()) / gp["min"].transform(lambda s: s.rolling(5, min_periods=5).sum()).clip(lower=1)
P["days_since_prev"] = gp["date"].diff().dt.days
P["gp_season"] = P.groupby(["pid", "season"]).cumcount() + 1
P["new_team"] = P["team"] != gp["team"].shift(1)
P["games_with_team"] = P.groupby(["pid", (P["new_team"] | (P["season"] != gp["season"].shift(1))).cumsum()]).cumcount() + 1
P["traded_in_season"] = (P["team"] != gp["team"].shift(1)) & (P["season"] == gp["season"].shift(1))
P["games_since_trade"] = P.groupby(["pid", P["traded_in_season"].cumsum()]).cumcount()
season_start = box.groupby("season")["date"].min()
P["season_day"] = (P["date"] - P["season"].map(season_start)).dt.days

# DELCO beliefs after each game (season and career mode) and age
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "after", "afterMIN", "career", "careerMIN"])
k["GAME_DATE"] = pd.to_datetime(k["GAME_DATE"])
P = P.merge(k.drop_duplicates(["PLAYER_ID", "GAME_DATE"]).rename(columns={"PLAYER_ID": "pid", "GAME_DATE": "date", "after": "delco", "afterMIN": "delco_min",
                                                                            "career": "delco_career", "careerMIN": "delco_career_min"}), on=["pid", "date"], how="left")
ages = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "GAME_DATE", "AGE_AT_GAME"])
ages["GAME_DATE"] = pd.to_datetime(ages["GAME_DATE"])
P = P.merge(ages.drop_duplicates(["PLAYER_ID", "GAME_DATE"]).rename(columns={"PLAYER_ID": "pid", "GAME_DATE": "date", "AGE_AT_GAME": "age"}), on=["pid", "date"], how="left")

# teammates who sat this game (rotation players by DELCO minutes belief 22+, last game for this team within 10 days) and their DELCO level
out_fp = {}
order = box.sort_values(["date"]).groupby(["team", "gid"], sort=False)
dl = {(a, b): (c, d) for a, b, c, d in zip(P["pid"], P["gid"], P["delco"], P["delco_min"])}
team_state = {}
for (team, gid), grp in order:
    stt = team_state.setdefault(team, {})
    date = grp["date"].iloc[0]
    here = set(grp.loc[grp["played"], "pid"])
    out_fp[(team, gid)] = sum(v[1] for p_, v in stt.items() if p_ not in here and v[2] >= 22 and (date - v[0]).days <= 10 and np.isfinite(v[1]))
    for p_ in here:
        r = dl.get((p_, gid))
        if r is not None and np.isfinite(r[0]):
            stt[p_] = (date, float(r[0]), float(r[1]) if np.isfinite(r[1]) else 0.0)
P["out_fp"] = [out_fp.get((t, g), 0.0) for t, g in zip(P["team"], P["gid"])]
P = P[P["season"] > FIRST].reset_index(drop=True)
P["info_id"] = np.arange(len(P))
print(f"panel: {len(P)} info points", flush=True)

# ---------------- targets: the team's next 10 games within 14 days
played_map = {(a, b): (c, d, e) for a, b, c, d, e in zip(box["pid"], box["gid"], box["min"], box["fp"], box["started"])}
rows = []
for r in P[["info_id", "pid", "team", "gid", "date"]].itertuples(index=False):
    tg = tg_by_team.get(r.team)
    i0 = tg_key.get((r.team, r.gid))
    if tg is None or i0 is None:
        continue
    for k_, (g2, d2) in enumerate(zip(tg_gid[r.team][i0 + 1:i0 + 11], tg_date[r.team][i0 + 1:i0 + 11]), start=1):
        da = (d2 - r.date).days
        if da > 14:
            break
        m = played_map.get((r.pid, g2))
        if m is not None:
            rows.append((r.info_id, k_, da, m[0] > 0, m[1] if m[0] > 0 else np.nan, m[0], m[2]))
        else:
            rows.append((r.info_id, k_, da, False, np.nan, 0.0, False))     # not in the box score: did not play (or left the team)
T = pd.DataFrame(rows, columns=["info_id", "k", "days_ahead", "played", "fp", "min", "started"])
P.drop(columns=["comment"]).to_parquet(D / "st_panel.parquet", index=False)
T.to_parquet(D / "st_targets.parquet", index=False)
print(f"targets: {len(T)} rows; played share {T['played'].mean():.1%}; wrote st_panel.parquet / st_targets.parquet")
