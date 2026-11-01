"""Laptop-free test, part 1 (2026-11-01, Tommy): do ESPN's box scores (pull_espn_box.py) match the NBA site's (pull_recent_gamelogs.py / pull_box_v3.py)?
Matches player-games on (date, team, NBA player id) and reports: games and player-games found on each side, every stat that differs, fantasy points
differences, did-not-play reasons, and whether the rotation status (healthy players out of the rotation) comes out the same.
Prints a summary and returns it as a dict (cloud_test.py puts it on the scorecard).
Run from ingest/:  python research/compare_espn_nba.py [season] [--out FILE]
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
GL = R / "data" / "game_logs"
FP = dict(PTS=1, REB=1.5, AST=2, STL=3, BLK=3, FG3M=1, FTM=2, FTA=-1, TOV=-1)
STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "FTM", "FTA", "FGM", "FGA", "OREB", "DREB", "PF"]


def bucket(c):
    c = str(c) if pd.notna(c) else ""
    if c == "":
        return "played"
    if re.search("Injury|Illness", c, re.I):
        return "injury"
    if re.search("G League|Two-Way", c, re.I):
        return "g_league"
    if re.search("Not With Team|Suspen|Personal", c, re.I):
        return "away"
    return "coach"


def norm(n):
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    n = re.sub(r"(jr|sr|ii|iii|iv)\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def rotation(b):
    """build_rotation_status.py's rule: of each player's last 5 healthy team games, how many he played."""
    c = b.comment.fillna("")
    b = b.assign(played=c.eq("") & b.minutes.notna())
    b = b[~c.str.contains("Injury|Illness|Not With Team|Suspen|Personal", case=False)].sort_values(["game_date", "gameId"])
    b = b.assign(nm=(b.firstName.astype(str) + " " + b.familyName.astype(str)).map(norm))     # by name, as build_rotation_status.py keys its output
    return {nm: (int(s.tail(5).played.sum()), len(s.tail(5))) for nm, s in b.groupby("nm")}


def compare(season):
    nba = pd.read_csv(GL / f"nba_api_{season}.csv")
    esp = pd.read_csv(GL / f"espn_nba_api_{season}.csv")
    last = min(nba.GAME_DATE.max(), esp.GAME_DATE.max())          # both sides through the same night
    nba, esp = nba[nba.GAME_DATE <= last], esp[esp.GAME_DATE <= last]
    k = ["GAME_DATE", "TEAM_ABBREVIATION", "PLAYER_ID"]
    for df in (nba, esp):
        df["fp"] = sum(df[c] * w for c, w in FP.items())
    m = nba.merge(esp, on=k, how="outer", suffixes=("_n", "_e"), indicator=True)
    both = m[m._merge == "both"]
    only_n, only_e = m[m._merge == "left_only"], m[m._merge == "right_only"]
    diffs = {s: int((both[f"{s}_n"] != both[f"{s}_e"]).sum()) for s in STATS}
    fpd = (both.fp_e - both.fp_n).abs()
    out = {"through": last, "games_nba": int(nba.groupby(["GAME_DATE", "MATCHUP"]).ngroups // 2), "games_espn": int(esp.GAME_ID.nunique()),
           "player_games_nba": len(nba), "player_games_espn": len(esp), "matched": len(both),
           "only_nba": [f"{r.GAME_DATE} {r.PLAYER_NAME_n} {r.TEAM_ABBREVIATION} ({r.MIN_n} min)" for r in only_n.itertuples()][:15], "only_nba_n": len(only_n),
           "only_espn": [f"{r.GAME_DATE} {r.PLAYER_NAME_e} {r.TEAM_ABBREVIATION} ({r.MIN_e} min)" for r in only_e.itertuples()][:15], "only_espn_n": len(only_e),
           "stat_diffs": diffs, "fp_exact": round(float((fpd < 0.01).mean()), 4), "fp_within_2": round(float((fpd <= 2).mean()), 4),
           "fp_mean_abs": round(float(fpd.mean()), 3), "fp_max": round(float(fpd.max()), 1),
           "fp_worst": [f"{r.GAME_DATE} {r.PLAYER_NAME_n}: NBA {r.fp_n:.1f}, ESPN {r.fp_e:.1f}" for r in both.assign(d=fpd).nlargest(5, "d").itertuples() if r.d > 0.01]}
    # did-not-play reasons and rotation status
    bn = pd.read_csv(GL / f"box_v3_{season}.csv", dtype={"gameId": str})
    be = pd.read_csv(GL / f"espn_box_v3_{season}.csv", dtype={"gameId": str})
    bn, be = bn[bn.game_date.astype(str).str[:10] <= last], be[be.game_date <= last]
    bn["b"], be["b"] = bn.comment.map(bucket), be.comment.map(bucket)
    bn["game_date"] = bn.game_date.astype(str).str[:10]
    kk = ["game_date", "teamTricode", "personId"]
    mb = bn[kk + ["b", "familyName"]].merge(be[kk + ["b"]], on=kk, how="outer", suffixes=("_n", "_e"))
    mb = mb[mb.set_index(["game_date", "teamTricode"]).index.isin(bn.set_index(["game_date", "teamTricode"]).index)]   # games the NBA file has
    mb = mb.fillna({"b_n": "not listed", "b_e": "not listed"})
    out["reasons"] = {f"{a} -> {b}": int(n) for (a, b), n in mb.groupby(["b_n", "b_e"]).size().items() if a != b}
    out["reasons_same"] = int((mb.b_n == mb.b_e).sum())
    be = be[be.set_index(["game_date", "teamTricode"]).index.isin(bn.set_index(["game_date", "teamTricode"]).index)]
    rn, re_ = rotation(bn), rotation(be)
    out_n = {p for p, (pl, n) in rn.items() if n >= 3 and pl <= 1}
    out_e = {p for p, (pl, n) in re_.items() if n >= 3 and pl <= 1}
    out["rotation_out_nba"], out["rotation_out_espn"] = len(out_n), len(out_e)
    out["rotation_only_nba"] = sorted(out_n - out_e)
    out["rotation_only_espn"] = sorted(out_e - out_n)
    return out


if __name__ == "__main__":
    from season import current_season_str
    from datetime import datetime
    from zoneinfo import ZoneInfo
    a = sys.argv[1:]
    out = Path(a[a.index("--out") + 1]) if "--out" in a else None
    season = next((x for x in a if re.fullmatch(r"\d{4}-\d{2}", x)), current_season_str())
    res = compare(season)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if out:                                    # the laptop's daily copy for the cloud test's scorecard (cloud_test.py reads it from main)
        res = {"date": datetime.now(ZoneInfo("America/New_York")).date().isoformat(), **res}
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
