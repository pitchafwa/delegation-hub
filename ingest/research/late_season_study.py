"""Late-season motivation study (Tommy, 2026-10-25): in the stretch that matches our fantasy playoffs, how much less do players on teams with nothing to play for
(or a reason to lose) play, compared with teams still in a race?

Window: our fantasy playoffs (weeks 20-22) run 2027-03-08 .. 03-28 and the NBA regular season ends 2027-04-11, so the window is END-34 .. END-14 days of each season.
Team status at the window's start (standings rebuilt from every regular-season game; win % league-wide and by conference):
  bottom3      3 worst records in the league                    (the lottery's best odds until this season)
  bottom4_10   4th-10th worst
  race         conference seeds 6-12 not in the bottom 10   (playoff / play-in race)
  secure       conference seeds 1-5
Players: each team's "core" = its top 3 by fantasy points per game before the window (25+ games, 24+ minutes). Outcome per core player:
  availability  share of the team's window games he played, relative to his share before the window (Nov 1 .. window start - 7 days)
  minutes       minutes per game played in the window vs before
  points        fantasy points per TEAM game in the window vs before (availability x level: what a fantasy manager gets)
Also young players (age 23 or younger, 10+ games before) on each team: minutes in the window vs before (the 'kids get run' effect).
Regimes: 2012-13..2018-19 (steep lottery: worst team 25%), 2020-21..2025-26 (2019 flattening: bottom 3 at 14% each, play-in from 2021). 2019-20 (bubble) excluded.
A falling number for bottom teams after 2019, when their lottery edge shrank, is direct evidence that losing incentives drive the rest.
Run from ingest/:  .venv/Scripts/python research/late_season_study.py   (writes data/late_season_study.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
SEASONS_OLD = [f"{y}-{str(y + 1)[2:]}" for y in range(2012, 2019)]
SEASONS_NEW = [f"{y}-{str(y + 1)[2:]}" for y in range(2020, 2026)]
EAST = {"ATL", "BOS", "BKN", "NJN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}

g = pd.read_csv(D / "game_logs_unified.csv", usecols=["PLAYER_ID", "SEASON", "GAME_ID", "GAME_DATE", "TEAM", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "TD3"])
g = g[g["SEASON"].isin(SEASONS_OLD + SEASONS_NEW)].copy()
g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
g["fp"] = g["PTS"] + 1.5 * g["REB"] + 2 * g["AST"] + 3 * g["STL"] + 3 * g["BLK"] + g["FG3M"] + 2 * g["FTM"] - g["FTA"] - g["TOV"] + 3 * g["TD3"]
ages = pd.read_csv(D / "kalman_input.csv", usecols=["PLAYER_ID", "SEASON", "AGE"]).drop_duplicates(["PLAYER_ID", "SEASON"]).set_index(["PLAYER_ID", "SEASON"])["AGE"]

rows, young = [], []
for season, s in g.groupby("SEASON"):
    end = s["GAME_DATE"].max()
    w0, w1 = end - pd.Timedelta(days=34), end - pd.Timedelta(days=14)
    pre0, pre1 = pd.Timestamp(f"{season[:4]}-11-01"), w0 - pd.Timedelta(days=7)
    # team results per game
    tg = s.groupby(["GAME_ID", "TEAM", "GAME_DATE"])["PTS"].sum().reset_index()
    opp = tg.merge(tg, on=["GAME_ID", "GAME_DATE"], suffixes=("", "_o"))
    opp = opp[opp["TEAM"] != opp["TEAM_o"]]
    opp["W"] = (opp["PTS"] > opp["PTS_o"]).astype(int)
    st = opp[opp["GAME_DATE"] < w0].groupby("TEAM")["W"].agg(["sum", "count"])
    st["pct"] = st["sum"] / st["count"]
    st = st.sort_values("pct")
    lg_rank = {t: i + 1 for i, t in enumerate(st.index)}                       # 1 = worst
    status = {}
    for conf in (True, False):
        cs = st[[(t in EAST) == conf for t in st.index]].sort_values("pct", ascending=False)
        for i, t in enumerate(cs.index):
            seed = i + 1
            r = lg_rank[t]
            status[t] = "bottom3" if r <= 3 else ("bottom4_10" if r <= 10 else ("secure" if seed <= 5 else "race"))
    team_games = opp.groupby("TEAM")["GAME_DATE"].apply(list).to_dict()
    for team, gd in team_games.items():
        gd = pd.Series(gd)
        n_pre, n_win = int(((gd >= pre0) & (gd <= pre1)).sum()), int(((gd >= w0) & (gd <= w1)).sum())
        if n_pre < 20 or n_win < 5:
            continue
        ps = s[(s["TEAM"] == team)]
        pre = ps[(ps["GAME_DATE"] >= pre0) & (ps["GAME_DATE"] <= pre1) & (ps["MIN"] > 0)]
        win = ps[(ps["GAME_DATE"] >= w0) & (ps["GAME_DATE"] <= w1) & (ps["MIN"] > 0)]
        agg = pre.groupby("PLAYER_ID").agg(gp=("fp", "size"), fpg=("fp", "mean"), mpg=("MIN", "mean"))
        core = agg[(agg["gp"] >= 25) & (agg["mpg"] >= 24)].sort_values("fpg", ascending=False).head(3)
        for pid, r in core.iterrows():
            wg = win[win["PLAYER_ID"] == pid]
            avail_pre, avail_win = r["gp"] / n_pre, len(wg) / n_win
            rows.append({"season": season, "team": team, "status": status.get(team, "race"), "regime": "old" if season in SEASONS_OLD else "new",
                         "avail_ratio": avail_win / avail_pre, "min_ratio": (wg["MIN"].mean() / r["mpg"]) if len(wg) else np.nan,
                         "pts_ratio": (wg["fp"].sum() / n_win) / (r["fpg"] * avail_pre), "fpg": r["fpg"]})
        for pid, r in agg[agg["gp"] >= 10].iterrows():
            a = ages.get((pid, season))
            if a is None or a > 23:
                continue
            wg = win[win["PLAYER_ID"] == pid]
            young.append({"season": season, "status": status.get(team, "race"), "regime": "old" if season in SEASONS_OLD else "new",
                          "min_pre": r["mpg"], "min_win": wg["MIN"].mean() if len(wg) else 0.0, "pts_pre": r["fpg"] * r["gp"] / n_pre, "pts_win": wg["fp"].sum() / n_win})

R = pd.DataFrame(rows)
Y = pd.DataFrame(young)
ORDER = ["secure", "race", "bottom4_10", "bottom3"]
out = {}
print(f"core players: {len(R)} (team-seasons {R.groupby(['season', 'team']).ngroups})\n")
print("Core players (top 3 per team): fantasy-playoff window vs before.  'points' = fantasy points per team game (what a manager gets)")
for reg in ("old", "new"):
    print(f"\n  {'2012-19 steep lottery' if reg == 'old' else '2020-26 flatter lottery (2019 reform)'}")
    print(f"    {'status':<11} {'n':>4}  {'availability':>12}  {'minutes':>8}  {'points':>7}   points vs race teams")
    base = R[(R["regime"] == reg) & (R["status"] == "race")]["pts_ratio"].mean()
    for stt in ORDER:
        x = R[(R["regime"] == reg) & (R["status"] == stt)]
        out[f"{reg}_{stt}"] = {"n": len(x), "avail": x["avail_ratio"].mean(), "min": x["min_ratio"].mean(), "pts": x["pts_ratio"].mean(), "pts_vs_race": x["pts_ratio"].mean() / base}
        print(f"    {stt:<11} {len(x):>4}  {x['avail_ratio'].mean():>12.2f}  {x['min_ratio'].mean():>8.2f}  {x['pts_ratio'].mean():>7.2f}   {x['pts_ratio'].mean() / base:>6.2f}")
print("\nYoung players (23 or younger): fantasy points per team game, window vs before (minutes in brackets)")
for reg in ("old", "new"):
    line = []
    for stt in ORDER:
        y = Y[(Y["regime"] == reg) & (Y["status"] == stt)]
        line.append(f"{stt} {y['pts_win'].sum() / max(1e-9, y['pts_pre'].sum()):.2f} ({y['min_win'].mean():.1f} vs {y['min_pre'].mean():.1f} min, n={len(y)})")
        out[f"young_{reg}_{stt}"] = {"n": len(y), "pts": float(y["pts_win"].sum() / max(1e-9, y["pts_pre"].sum()))}
    print(f"  {reg}: " + " | ".join(line))
# bootstrap: is the bottom-team gap smaller after the 2019 reform?
rng = np.random.default_rng(7)
def gap(df):
    return df[df["status"].isin(["bottom3", "bottom4_10"])]["pts_ratio"].mean() - df[df["status"] == "race"]["pts_ratio"].mean()
bo, bn = R[R["regime"] == "old"], R[R["regime"] == "new"]
diffs = []
for _ in range(2000):
    so = bo.groupby("season", group_keys=False).apply(lambda d: d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))))
    sn = bn.groupby("season", group_keys=False).apply(lambda d: d.sample(len(d), replace=True, random_state=int(rng.integers(1e9))))
    diffs.append(gap(sn) - gap(so))
lo, hi = np.percentile(diffs, [5, 95])
print(f"\nBottom-10 gap vs race teams: old {gap(bo):+.2f}, new {gap(bn):+.2f}; change {gap(bn) - gap(bo):+.2f} (90% range {lo:+.2f} .. {hi:+.2f})")
out["gap_old"], out["gap_new"], out["gap_change_ci"] = float(gap(bo)), float(gap(bn)), [float(lo), float(hi)]
json.dump(out, open(D / "late_season_study.json", "w"), indent=1, default=float)
