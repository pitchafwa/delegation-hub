"""Pull Hashtag Basketball's crowdsourced dynasty rankings (the MKT column) from https://hashtagbasketball.com/keeper.

The rankings are a public static HTML table (robots.txt only blocks SemrushBot). We fetch it once a day with an honest User-Agent,
keep a dated copy (data/hashtag_dynasty_YYYY-MM-DD.csv) and a stable data/hashtag_dynasty_latest.csv that the model reads.
Refuses to overwrite `latest` with a short/garbled table (a bad pull must never blank the MKT column).
"""
import re
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
URL = "https://hashtagbasketball.com/keeper"

r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (personal fantasy basketball tool; contact tommy.valtin@gmail.com)"}, timeout=60)
r.raise_for_status()
html = r.text
tables = pd.read_html(StringIO(html), attrs={"id": "ContentPlaceHolder1_GridView1"})
t = tables[0].rename(columns=str.lower)
t = t.rename(columns={"#": "rank"})
t["rank"] = pd.to_numeric(t["rank"], errors="coerce")
t = t.dropna(subset=["rank"]).copy()
t["rank"] = t["rank"].astype(int)
votes = re.search(r"([\d,]{6,}) votes", html)
if len(t) < 500 or int(t["rank"].min()) != 1 or t["player"].isna().mean() > 0.02:
    sys.exit(f"refusing to overwrite: table looks wrong ({len(t)} rows, min rank {t['rank'].min() if len(t) else None})")
t["asof"] = date.today().isoformat()
t.to_csv(D / f"hashtag_dynasty_{date.today().isoformat()}.csv", index=False)
t.to_csv(D / "hashtag_dynasty_latest.csv", index=False)
print(f"{len(t)} rows, top: {', '.join(t.player.head(3))}; votes {votes.group(1) if votes else '?'}")
