"""DARKO projections, logged daily for the Dec 14 blend study (2026-11-09, Tommy). Logged only: nothing on the site uses them yet.

Source: darko.app "Fantasy Lab" (https://www.darko.app/projections), the page's own data (projections/__data.json): per player, projected minutes, pace and
every box-score stat per 100 possessions, including free-throw attempts and percentages, so THIS league's scoring is applied exactly (the page's
"Custom" scoring does the same arithmetic in the browser):
    per game = per-100 x (minutes / 48) x (pace / 100);  3PM = 3PA x 3P%;  FTM = FTA x FT%;  REB = OREB + DREB
    fantasy points = PTS + 1.5 REB + 2 AST + 3 STL + 3 BLK + 3PM + 2 FTM - FTA - TOV        (triple-doubles not projected)
INJURED PLAYERS: DARKO's minutes for a player who is out or questionable are cut by his chance of playing (Kon Knueppel 14.3, Jimmy Butler 19.3 on 11-09).
The study compares projections for games a player PLAYS, so each row also has `min_full` = minutes if he plays:
    ACTIVE per ESPN -> DARKO's minutes (and remembered as his last healthy DARKO minutes)
    otherwise       -> his last healthy DARKO minutes (data/darko_healthy_minutes.json), else season-mode DELCO's expected minutes, else DARKO's
`fp_full` uses min_full; `fp_raw` uses DARKO's minutes as published.
Writes data/darko_daily/<date>.csv (data/ is private: not in the public repo; the cloud test carries it in its own storage). Non-critical step in refresh_all.py.
  python research/pull_darko.py        (from ingest/)
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
OUT = D / "darko_daily"
MEM = D / "darko_healthy_minutes.json"
URL = "https://www.darko.app/projections/__data.json"
SCORING = {"pts": 1.0, "reb": 1.5, "ast": 2.0, "stl": 3.0, "blk": 3.0, "fg3m": 1.0, "ftm": 2.0, "fta": -1.0, "tov": -1.0}


def resolve(data, i, depth=0):
    """SvelteKit's compact data format: values are stored once and referenced by index"""
    v = data[i]
    if depth > 6:
        return v
    if isinstance(v, list):
        return [resolve(data, x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {k: resolve(data, x, depth + 1) for k, x in v.items()}
    return v


def main():
    r = requests.get(URL, headers={"User-Agent": "delegation-hub projection log (personal fantasy league tool)"}, timeout=60)
    r.raise_for_status()
    node = next(n for n in r.json()["nodes"] if n and n.get("type") == "data" and isinstance(n.get("data"), list) and n["data"] and isinstance(n["data"][0], dict) and "players" in n["data"][0])
    data = node["data"]
    root = data[0]
    players = [resolve(data, i) for i in data[root["players"]]]
    as_of = resolve(data, root["asOf"]) if "asOf" in root else None
    status = {}
    try:
        for row in csv.DictReader(open(D / "espn_positions.csv", encoding="utf-8")):
            status[int(row["PLAYER_ID"])] = (row.get("injury_status") or "ACTIVE").upper()
    except Exception:
        pass
    delco_min = {}
    try:
        for row in csv.DictReader(open(D / "delco_pregame.csv", encoding="utf-8")):
            delco_min[int(row["PLAYER_ID"])] = float(row["afterMIN"])          # rows are in date order: the last one per player wins
    except Exception:
        pass
    mem = json.loads(MEM.read_text(encoding="utf-8")) if MEM.exists() else {}
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    rows = []
    for p in players:
        try:
            pid = int(p["nba_id"])
            mins, pace = float(p["x_minutes"]), float(p["x_pace"])
        except (KeyError, TypeError, ValueError):
            continue
        st = status.get(pid, "ACTIVE")

        def per_game(k, m):
            return float(p.get(k) or 0.0) * (m / 48.0) * (pace / 100.0)
        if st == "ACTIVE":
            mfull, src = mins, "darko"
            mem[str(pid)] = {"min": round(mins, 2), "date": today}
        elif str(pid) in mem:
            mfull, src = mem[str(pid)]["min"], "darko_last_healthy"
        elif pid in delco_min:
            mfull, src = delco_min[pid], "delco"
        else:
            mfull, src = mins, "darko"

        def line(m):
            s = {"pts": per_game("x_pts_100", m), "reb": per_game("x_orb_100", m) + per_game("x_drb_100", m), "ast": per_game("x_ast_100", m),
                 "stl": per_game("x_stl_100", m), "blk": per_game("x_blk_100", m), "tov": per_game("x_tov_100", m),
                 "fg3m": per_game("x_fg3a_100", m) * float(p.get("x_fg3_pct") or 0.0), "fta": per_game("x_fta_100", m)}
            s["ftm"] = s["fta"] * float(p.get("x_ft_pct") or 0.0)
            return s, sum(SCORING[k] * s[k] for k in SCORING)
        full, fp_full = line(mfull)
        _, fp_raw = line(mins)
        rows.append({"date": today, "as_of": as_of, "nba_id": pid, "name": p.get("player_name"), "team": p.get("team_name"), "status": st,
                     "x_minutes": round(mins, 2), "min_full": round(mfull, 2), "min_src": src, "pace": round(pace, 2),
                     **{k: round(v, 2) for k, v in full.items()}, "fp_full": round(fp_full, 2), "fp_raw": round(fp_raw, 2),
                     "fp_per36": round(fp_full / mfull * 36, 2) if mfull > 0 else None})
    if not rows:
        raise SystemExit("DARKO: no players parsed (page format changed?)")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{today}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    MEM.write_text(json.dumps(mem, separators=(",", ":")), encoding="utf-8")
    adj = [r_ for r_ in rows if r_["min_src"] != "darko"]
    ex = ", ".join(f"{a['name']} {a['x_minutes']}->{a['min_full']}" for a in adj[:4])
    print(f"DARKO as of {as_of}: {len(rows)} players -> darko_daily/{today}.csv; minutes restored for {len(adj)} not-active players (e.g. {ex})")


if __name__ == "__main__":
    main()
