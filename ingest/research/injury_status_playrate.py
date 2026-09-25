"""How often does each official NBA injury designation actually play?  (replaces the guessed 55% for day-to-day)

Inputs: data/nba_injury_status.csv (nba_injury_reports.py history: the 06:00 AM ET and 05:30 PM ET report for every 2024-25 and 2025-26
game date) and the NBA game logs (played = appears in that day's box scores, i.e. logged minutes).
For every (game date, player) listed on the report we ask: did he play?  Reported by designation, by report time, and by the player's
recent level (rotation player vs deep bench), plus how often a morning 'Questionable' resolves in each direction by evening.
Writes data/injury_playrates.json (used by the weekly plan) with the rotation-player rates for the 06:00 AM report.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"


def key(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"[^a-z]", "", n.lower())


inj = pd.read_csv(D / "nba_injury_status.csv")
logs = pd.concat([pd.read_csv(D / "game_logs" / f"nba_api_{s}.csv").assign(S=s) for s in ("2024-25", "2025-26")])
logs["key"] = logs.PLAYER_NAME.map(key)
logs["date"] = pd.to_datetime(logs.GAME_DATE).dt.strftime("%Y-%m-%d")
logs["fp"] = logs.PTS + 1.5 * logs.REB + 2 * logs.AST + 3 * logs.STL + 3 * logs.BLK + logs.FG3M + 2 * logs.FTM - logs.FTA - logs.TOV
played = set(zip(logs.key, logs.date))
# recent level (before the date) to separate rotation players from deep bench
lv = {}
for k, sub in logs.sort_values("date").groupby("key"):
    lv[k] = (sub.date.to_numpy(), sub.fp.to_numpy(), sub.MIN.to_numpy())


def recent(k, d):
    if k not in lv:
        return np.nan, np.nan
    ds, fp, mn = lv[k]
    i = np.searchsorted(ds, d)
    if i < 6:
        return np.nan, np.nan
    return float(fp[max(0, i - 15):i].mean()), float(mn[max(0, i - 15):i].mean())


inj = inj.drop_duplicates(["slot", "game_date", "key"])
inj["played"] = [int((k, d) in played) for k, d in zip(inj.key, inj.game_date)]
rl = [recent(k, d) for k, d in zip(inj.key, inj.game_date)]
inj["fp15"], inj["min15"] = [r[0] for r in rl], [r[1] for r in rl]
inj["rot"] = np.where(inj.min15 >= 20, "rotation (20+ mpg)", np.where(inj.min15.notna(), "bench (<20 mpg)", "unknown"))
print(f"{len(inj):,} listed player-days; {inj.game_date.nunique()} game dates; seasons {sorted(inj.season.unique())}")
order = ["Available", "Probable", "Questionable", "Doubtful", "Out"]
for slot in ("6:00AM", "5:30PM"):
    s = inj[inj.slot == slot]
    print(f"\n=== {slot} ET report: share who played, by designation (all listed players)")
    print(s.groupby("status").played.agg(["mean", "size"]).reindex(order).round(3).to_string())
    print(f"--- rotation players (20+ mpg over last 15 games) only")
    r = s[s.rot.str.startswith("rotation")]
    print(r.groupby("status").played.agg(["mean", "size"]).reindex(order).round(3).to_string())
    print(f"--- bench players (<20 mpg) only")
    b = s[s.rot.str.startswith("bench")]
    print(b.groupby("status").played.agg(["mean", "size"]).reindex(order).round(3).to_string())
# how morning statuses resolve by evening
m = inj[inj.slot == "6:00AM"].set_index(["game_date", "key"]).status
e = inj[inj.slot == "5:30PM"].set_index(["game_date", "key"]).status
j = pd.concat([m.rename("morning"), e.rename("evening")], axis=1).dropna(subset=["morning"])
j["evening"] = j.evening.fillna("(not listed)")
print("\nmorning Questionable -> evening status (share):")
print(j[j.morning == "Questionable"].evening.value_counts(normalize=True).round(3).to_string())
# 'not listed at all' baseline for rotation players is measured in rest_absence_test.py / availability_by_level.py
rates = {}
s = inj[(inj.slot == "6:00AM") & inj.rot.str.startswith("rotation")]
for st in order:
    x = s[s.status == st].played
    if len(x) >= 30:
        rates[st] = {"p_play": round(float(x.mean()), 3), "n": int(len(x))}
allb = inj[(inj.slot == "6:00AM")]
for st in order:
    x = allb[allb.status == st].played
    if len(x) >= 30:
        rates.setdefault(st, {})["p_play_all"] = round(float(x.mean()), 3)
json.dump({"source": "NBA official injury reports 06:00 AM ET vs box scores, 2024-25 and 2025-26", "rotation_players_min20": rates}, open(D / "injury_playrates.json", "w"), indent=1)
print("\nsaved injury_playrates.json:", rates)
