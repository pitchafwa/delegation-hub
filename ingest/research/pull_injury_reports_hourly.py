"""Official NBA injury reports WITH the reason text, five seasons (2021-22 to 2025-26).

nba_injury_reports.py only saw the newer 15-minute file names (available from Dec 2025), which limited it to about 100 game days of one season and
threw the reason away.  Older reports are named by the hour (Injury-Report_2022-01-20_05PM.pdf) and exist back to 2021-22, so this pulls the late-morning
(11AM) and evening (05PM, published 5:30 PM ET) report for every game date and keeps every column:
    report_date, slot, game_date, matchup, team, player_key, player_name, status, reason
Layout is read from word positions (the PDFs wrap long reasons over several lines and, in newer files, drop the spaces), assigning each cell to its
row by vertical position.  Output: data/injury_reports_hourly.csv (gitignored, regenerable).
Run from ingest/:  uv run python research/pull_injury_reports_hourly.py [test]
"""
import io
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{s}.pdf"
UA = {"User-Agent": "Mozilla/5.0"}
SEASONS = {"2021-22": (date(2021, 10, 19), date(2022, 4, 10)), "2022-23": (date(2022, 10, 18), date(2023, 4, 9)), "2023-24": (date(2023, 10, 24), date(2024, 4, 14)),
           "2024-25": (date(2024, 10, 22), date(2025, 4, 13)), "2025-26": (date(2025, 10, 21), date(2026, 4, 12)), "2026-27": (date(2026, 10, 20), date(2027, 4, 11))}
SLOTS = ("11AM", "05PM")
STATUS = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
SUFFIX = re.compile(r"(Jr\.|Sr\.|II|III|IV)$")
DATE_RE = re.compile(r"^\d\d/\d\d/\d{4}$")
_T = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
      "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "LA Clippers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
      "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
      "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards", "Charlotte Bobcats", "New Orleans Hornets", "Seattle SuperSonics"]
TEAMS = {re.sub(r"[^a-z]", "", t.lower()): t for t in _T}


def key_of(name):
    """'Hardaway Jr., Tim' / 'HardawayJr.,Tim' -> 'timhardaway'"""
    if "," not in name:
        return re.sub(r"[^a-z]", "", name.lower())
    last, first = name.split(",", 1)
    last = SUFFIX.sub("", last.replace(" ", ""))
    return re.sub(r"[^a-z]", "", (first + last).lower())


def parse(content):
    import pdfplumber
    rows = []
    ctx = {"game_date": None, "matchup": None, "team": None}
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        cols = None
        for page in pdf.pages:
            ws = page.extract_words(x_tolerance=1.5)
            hdr = {}
            rtop = min((w["top"] for w in ws if w["text"] == "Reason"), default=None)
            if rtop is None:
                hdr_words = []
            else:
                hdr_words = [w for w in ws if abs(w["top"] - rtop) < 3]
            for w in hdr_words:
                t = w["text"]
                if w["text"] in ("Game", "GameDate") and "date" not in hdr:
                    hdr["date"] = w["x0"]
                elif t in ("Matchup",):
                    hdr["matchup"] = w["x0"]
                elif t == "Team":
                    hdr["team"] = w["x0"]
                elif t in ("Player", "PlayerName"):
                    hdr["player"] = w["x0"]
                elif t in ("Current", "CurrentStatus"):
                    hdr["status"] = w["x0"]
                elif t == "Reason":
                    hdr["reason"] = w["x0"]
            if len(hdr) == 6:
                cols = hdr
            if not cols:
                continue
            bounds = sorted((x - 4, k) for k, x in cols.items())
            # the time column sits between date and matchup: treat as part of 'date'

            def col_of(x0):
                c = None
                for bx, k in bounds:
                    if x0 >= bx:
                        c = k
                return c
            top_hdr = rtop if rtop is not None else 0
            body = [w for w in ws if w["top"] > top_hdr + 3 and w["text"] not in ("Page",) and not re.match(r"^\d+of\d+$", w["text"])]
            body.sort(key=lambda w: (round(w["top"]), w["x0"]))
            # anchors = player-name cells (a name has a comma); group consecutive words in the player column that share a line
            players, ctxs, team_lines, team_buf = [], [], [], None
            for w in body:
                c = col_of(w["x0"])
                if c == "date":
                    if DATE_RE.match(w["text"]):
                        m, d, y = w["text"].split("/")
                        ctx = dict(ctx, game_date=f"{y}-{m}-{d}")
                        ctxs.append((w["top"], ctx))
                elif c == "matchup":
                    if "@" in w["text"]:
                        ctx = dict(ctx, matchup=w["text"], team=None)
                        ctxs.append((w["top"], ctx))
                elif c == "team":
                    if team_buf and abs(w["top"] - team_buf[0]) < 3:
                        team_buf[1].append(w["text"])
                    else:
                        team_buf = [w["top"], [w["text"]]]
                        team_lines.append(team_buf)
                    joined = re.sub(r"[^a-z]", "", "".join(team_buf[1]).lower())
                    if joined in TEAMS:
                        ctx = dict(ctx, team=TEAMS[joined])
                        ctxs.append((w["top"], ctx))
                    elif len(team_lines) > 1:            # a team name wrapped over two lines
                        j2 = re.sub(r"[^a-z]", "", ("".join(team_lines[-2][1]) + "".join(team_buf[1])).lower())
                        if j2 in TEAMS:
                            ctx = dict(ctx, team=TEAMS[j2])
                            ctxs.append((w["top"], ctx))
                elif c == "player":
                    if players and abs(w["top"] - players[-1]["top"]) < 3:
                        players[-1]["name"] += " " + w["text"]
                    else:
                        players.append({"top": w["top"], "name": w["text"], "status": None, "reason": [], "ctx": ctx})
            if not players:
                continue
            for p in players:                                   # the game/team context in force at this row
                best = None
                for top, c in ctxs:
                    if top <= p["top"] + 3:
                        best = c
                p["ctx"] = best or p["ctx"]
            tops = [p["top"] for p in players]
            for w in body:
                c = col_of(w["x0"])
                if c not in ("status", "reason"):
                    continue
                j = min(range(len(tops)), key=lambda i: abs(tops[i] - w["top"]))
                if c == "status" and w["text"] in STATUS:
                    players[j]["status"] = w["text"]
                elif c == "reason":
                    players[j]["reason"].append((round(w["top"]), w["x0"], w["text"]))
            for p in players:
                if p["status"] and "," in p["name"]:
                    reason = " ".join(t for _, _, t in sorted(p["reason"]))
                    reason = re.sub(r"(NOT\s*(YET)?\s*SUBMITTED\s*(YET)?)+", "", reason).strip()
                    rows.append((p["ctx"]["game_date"], p["ctx"]["matchup"], p["ctx"]["team"], key_of(p["name"]), p["name"], p["status"], reason))
    return rows


def job(args):
    d, slot = args
    for s in ([slot] if slot != "05PM" else ["05PM", "05_30PM"]):
        try:
            r = requests.get(URL.format(d=d.isoformat(), s=s), headers=UA, timeout=40)
        except Exception:
            continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            try:
                return [(d.isoformat(), slot) + row for row in parse(r.content)]
            except Exception as ex:
                return [("ERR", str(ex)[:80])]
    return []


def main():
    test = len(sys.argv) > 1 and sys.argv[1] == "test"
    update = len(sys.argv) > 1 and sys.argv[1] == "update"      # in season: only report dates after the newest one already saved
    old = None
    start_after = None
    if update:
        old = pd.read_csv(D / "injury_reports_hourly.csv")
        start_after = date.fromisoformat(old.report_date.max())
    jobs = []
    for season, (a, b) in SEASONS.items():
        if test and season != "2024-25":
            continue
        d = a + timedelta(days=30) if test else a
        if update:
            d = max(a, start_after + timedelta(days=1))
            b = min(b, date.today())
        while d <= b:
            for s in SLOTS:
                jobs.append((d, s))
            d += timedelta(days=1)
            if test and len(jobs) > 12:
                break
    print(len(jobs), "report slots to try")
    out, err = [], 0
    with ProcessPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(job, jobs, chunksize=4)):
            for r in res:
                if r[0] == "ERR":
                    err += 1
                else:
                    out.append(r)
            if i % 200 == 0:
                print(i, len(out), "rows", err, "errors", flush=True)
    if update and not out:
        print("no new reports")
        return
    df = pd.DataFrame(out, columns=["report_date", "slot", "game_date", "matchup", "team", "player_key", "player_name", "status", "reason"])
    if update:
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(["report_date", "slot", "game_date", "player_key", "status", "reason"])
    df.to_csv(D / ("injury_reports_hourly_test.csv" if test else "injury_reports_hourly.csv"), index=False)
    print("rows", len(df), "report-days", df.groupby(["report_date", "slot"]).ngroups, "errors", err)
    print(df.head(12).to_string())


if __name__ == "__main__":
    main()
