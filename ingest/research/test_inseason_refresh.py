"""PLUMBING TEST of the in-season refresh chain. Builds a synthetic 2026-27 game-log file (the first 20 days of 2025-26 shifted forward
364 days), runs refresh_all.py --full --no-push with the in-season switch forced, and reports how projections moved. Backs up and
then RESTORES every data file it touches, so it leaves the real model untouched.  Takes ~15 min.
Usage (from ingest/):  uv run python research/test_inseason_refresh.py"""
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
D = R / "data"
HUB = R.parent.parent / "dashboard"
TOUCH = [D / "game_logs_unified.csv", D / "kalman_input.csv", D / "asset_value.csv", D / "unified_paths.csv", D / "ceiling_paths.csv", D / "prospect_context.csv",
         D / "kalman_vor_all_players.csv", D / "current_player_trajectories.csv", D / "refresh_status.json", HUB / "hub_data.json", HUB / "breakout_history.json"]
TOUCH = [p for p in TOUCH if p.exists()]
BK = R / "_inseason_test_backup"
BK.mkdir(exist_ok=True)
for p in TOUCH:
    shutil.copy2(p, BK / p.name)
synth = D / "game_logs" / "nba_api_2026-27.csv"
try:
    g = pd.read_csv(D / "game_logs" / "nba_api_2025-26.csv")
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    first = g["GAME_DATE"].min()
    g = g[g["GAME_DATE"] < first + timedelta(days=20)].copy()
    g["GAME_DATE"] = (g["GAME_DATE"] + timedelta(days=364)).dt.strftime("%Y-%m-%d")
    g["SEASON_ID"] = "22026"
    g.to_csv(synth, index=False)
    print(f"synthetic 2026-27 logs: {len(g)} player-games over 20 days")
    before = json.load(open(HUB / "hub_data.json", encoding="utf-8"))
    env = dict(os.environ, REFRESH_FORCE_INSEASON="1")
    p = subprocess.run([sys.executable, str(R / "refresh_all.py"), "--full", "--no-push"], cwd=R.parent, env=env)
    print("refresh exit code", p.returncode)
    after = json.load(open(HUB / "hub_data.json", encoding="utf-8"))
    A = {x["id"]: x for x in before["players"]}
    moved = []
    for x in after["players"]:
        y = A.get(x["id"])
        if y and y.get("trajectory") and x.get("trajectory"):
            moved.append((x["player"], y["year0_ppg"], x["year0_ppg"], (y["asset_k"] or [0] * 6)[5], (x["asset_k"] or [0] * 6)[5]))
    d = pd.DataFrame(moved, columns=["player", "year0_before", "year0_after", "asset5_before", "asset5_after"])
    d["d_year0"] = d.year0_after - d.year0_before
    print("players compared:", len(d), " year-0 pts/g change: mean %.2f, |mean| %.2f, max |%.1f|" % (d.d_year0.mean(), d.d_year0.abs().mean(), d.d_year0.abs().max()))
    print(d.reindex(d.d_year0.abs().sort_values(ascending=False).index).head(8).round(1).to_string(index=False))
finally:
    if synth.exists():
        synth.unlink()
    for p in TOUCH:
        shutil.copy2(BK / p.name, p)
    shutil.rmtree(BK, ignore_errors=True)
    print("restored all touched data files; removed synthetic logs")
