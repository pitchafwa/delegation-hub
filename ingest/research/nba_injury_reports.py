"""Official NBA injury reports (public PDFs at ak-static.cms.nba.com/referee/injury/) -> tidy player statuses.

Statuses are the league's own participation designations: Available, Probable, Questionable, Doubtful, Out. Reports are published every 15
minutes (content changes as teams update). Used two ways:
  * history:  python nba_injury_reports.py history 2025-26 2024-25   -> data/nba_injury_status.csv  (06:00 AM and 05:30 PM ET report for every
              game date), to measure how often each designation actually plays
  * live:     latest_statuses(now) -> {player key: status} for the day's games (used by the weekly plan)
Player key = first+last name, lowercase letters only, suffix (Jr/Sr/II/III/IV) removed, so it matches norm(name).replace(' ', '') for our data.
"""
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
D = Path(__file__).resolve().parent / "data"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{h}_{m}{ap}.pdf"
STATUSES = "Out|Doubtful|Questionable|Probable|Available"
ROW = re.compile(rf"([A-Z][A-Za-z'.\-]*),([A-Za-z'.\-]+)\s+({STATUSES})\b")
DATE_HDR = re.compile(r"^(\d\d)/(\d\d)/(\d{4})\s")
SUFFIX = re.compile(r"(Jr\.|Sr\.|II|III|IV)$")
SESSION = requests.Session()


def key_of(last, first):
    last = SUFFIX.sub("", last)
    return re.sub(r"[^a-z]", "", (first + last).lower())


def parse_pdf(content):
    import pdfplumber
    rows, cur = [], None
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").split("\n"):
                m = DATE_HDR.match(line)
                if m:
                    cur = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
                if cur is None:
                    continue
                for last, first, status in ROW.findall(line):
                    rows.append((cur.isoformat(), key_of(last, first), status))
    return rows


def fetch(d, hour12, minute, ap):
    for i in range(3):
        try:
            r = SESSION.get(URL.format(d=d.isoformat(), h=f"{hour12:02d}", m=f"{minute:02d}", ap=ap), timeout=30)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return r.content
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1.5)
    return None


def slot_for(now):
    """the most recent 15-minute report slot at or before `now` (ET)"""
    now = now.astimezone(ET).replace(second=0, microsecond=0)
    now -= timedelta(minutes=now.minute % 15)
    h = now.hour % 12 or 12
    return now.date(), h, now.minute, "AM" if now.hour < 12 else "PM"


def latest_statuses(now=None, lookback_slots=12):
    """{(game date iso, player key): status} from the newest published report at/before now; walks back a few slots if one is missing."""
    now = now or datetime.now(ET)
    for back in range(lookback_slots):
        d, h, m, ap = slot_for(now - timedelta(minutes=15 * back))
        content = fetch(d, h, m, ap)
        if content:
            rows = parse_pdf(content)
            if rows:
                return {(g, k): s for g, k, s in rows}, f"{d} {h}:{m:02d}{ap}"
    return {}, None


def history(seasons):
    ranges = {"2025-26": (date(2025, 10, 21), date(2026, 4, 12)), "2024-25": (date(2024, 10, 22), date(2025, 4, 13))}
    jobs = []
    for s in seasons:
        a, b = ranges[s]
        d = a
        while d <= b:
            jobs.append((s, d, 6, 0, "AM"))
            jobs.append((s, d, 5, 30, "PM"))
            d += timedelta(days=1)

    def work(j):
        s, d, h, m, ap = j
        c = fetch(d, h, m, ap)
        time.sleep(0.15)
        if not c:
            return []
        return [(s, d.isoformat(), f"{h}:{m:02d}{ap}", g, k, st) for g, k, st in parse_pdf(c) if g == d.isoformat()]

    out = []
    with ThreadPoolExecutor(4) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            out += r
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} reports, {len(out)} rows", flush=True)
    df = pd.DataFrame(out, columns=["season", "date", "slot", "game_date", "key", "status"])
    df.to_csv(D / "nba_injury_status.csv", index=False)
    print(f"saved {len(df)} rows")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 2 and sys.argv[1] == "history":
        history(sys.argv[2:])
    else:
        st, at = latest_statuses()
        print(f"{len(st)} statuses from report {at}")
        print(pd.Series(list(st.values())).value_counts().to_string())
