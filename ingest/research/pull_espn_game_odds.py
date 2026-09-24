"""Pull pre-game betting lines (spread, total, moneylines) for every NBA game from ESPN's free core API.
Used to test whether Vegas-implied game environment (blowout risk, pace/total, team implied total) adds
information about players' fantasy output beyond what their own recent averages already say.

Writes data/game_odds.csv: one row per game (date, home, away, provider, spread [home-signed: negative = home favoured],
total, home/away moneyline). Resumable; polite (0.25s between calls).
Usage: uv run python research/pull_espn_game_odds.py 2024 2025 2026   (seasonIds = year the season ends)
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parent / "data" / "game_odds.csv"
PREF = ["DraftKings", "ESPN BET", "Caesars", "Caesars Sportsbook", "BetMGM", "Bet365", "FanDuel"]
S = requests.Session()


def get(url, **kw):
    for i in range(3):
        try:
            r = S.get(url, timeout=30, **kw)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.5)
    return None


def pick(items):
    good = [x for x in items if x.get("spread") is not None and x.get("overUnder") and "Live" not in x["provider"]["name"]]
    if not good:
        return None
    for p in PREF:
        for x in good:
            if x["provider"]["name"].startswith(p):
                return x
    return good[0]


done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
seen = set(done["event_id"]) if len(done) else set()
rows = done.to_dict("records") if len(done) else []
for yr in map(int, sys.argv[1:]):
    d, end = date(yr - 1, 10, 15), date(yr, 4, 20)
    while d <= end:
        sb = get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", params={"dates": d.strftime("%Y%m%d")})
        for ev in (sb or {}).get("events", []):
            eid = int(ev["id"])
            if eid in seen:
                continue
            comp = ev["competitions"][0]
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            if "abbreviation" not in home["team"] or "abbreviation" not in away["team"]:
                continue  # exhibition / non-NBA opponent
            od = get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds")
            best = pick((od or {}).get("items", []))
            seen.add(eid)
            if best is None:
                continue
            fav_home = (best.get("homeTeamOdds") or {}).get("favorite")
            sp = abs(best["spread"]) * (-1 if fav_home else 1) if fav_home is not None else best["spread"]
            rows.append(dict(event_id=eid, date=d.isoformat(), season=yr, home=home["team"]["abbreviation"], away=away["team"]["abbreviation"],
                             provider=best["provider"]["name"], spread_home=sp, total=best["overUnder"],
                             ml_home=(best.get("homeTeamOdds") or {}).get("moneyLine"), ml_away=(best.get("awayTeamOdds") or {}).get("moneyLine"),
                             home_score=float(home.get("score") or 0), away_score=float(away.get("score") or 0)))
            time.sleep(0.2)
        if d.day == 1 or d.day == 15:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(yr, d, len(rows), flush=True)
        d += timedelta(days=1)
        time.sleep(0.15)
pd.DataFrame(rows).to_csv(OUT, index=False)
print("finished", len(rows))
