"""Laptop-free test, part 3 (2026-11-01): does the cloud's DFF monitor (cloud-test-dff.yml) catch the same DailyFantasyFuel projection changes and
ESPN injury changes as the laptop's, and as quickly? Fetches the cloud's logs from Cloudflare R2 with wrangler (the laptop's own login), then for
yesterday (New York time) matches each change by what changed and compares when each side first saw it.
Writes dashboard/cloud_test_dff.json (summary only; the logs themselves stay private). Run from ingest/:  python research/compare_dff.py
"""
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
OUT = R.parent.parent / "dashboard" / "cloud_test_dff.json"
ET = ZoneInfo("America/New_York")
day = (datetime.now(ET) - timedelta(days=1)).date()

tmp = Path(tempfile.mkdtemp())
p = subprocess.run(f'npx --yes wrangler r2 object get delegation-hub-data/cloud/dfs.tar.gz --file "{tmp / "dfs.tar.gz"}" --remote',
                   shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=tmp)
if p.returncode != 0 or not (tmp / "dfs.tar.gz").exists():
    OUT.write_text(json.dumps({"date": datetime.now(ET).date().isoformat(), "error": "could not fetch the cloud's DFF logs: " + (p.stderr or p.stdout)[-200:]}), encoding="utf-8")
    sys.exit(0)
with tarfile.open(tmp / "dfs.tar.gz") as t:
    t.extractall(tmp, filter="data")


def load(d, f, keys):
    x = pd.read_csv(d / f, dtype=str, keep_default_na=False)
    x["t"] = pd.to_datetime(x.fetched_utc, utc=True)
    x = x[x.t.dt.tz_convert(ET).dt.date == day]
    return x.groupby(keys).t.min()


res = {"date": datetime.now(ET).date().isoformat(), "day": day.isoformat()}
for name, f, keys in (("dff", "log.csv", ["game_date", "name", "dk_proj", "inj"]), ("injuries", "inj_log.csv", ["name", "status", "espn_time"])):
    a, b = load(R / "data" / "dfs", f, keys), load(tmp / "dfs", f, keys)
    both = a.index.intersection(b.index)
    lag = ((b[both] - a[both]).dt.total_seconds() / 60) if len(both) else pd.Series(dtype=float)
    res[name] = {"laptop": len(a), "cloud": len(b), "matched": len(both), "only_laptop": len(a.index.difference(b.index)), "only_cloud": len(b.index.difference(a.index)),
                 "cloud_minus_laptop_median_min": round(float(lag.median()), 1) if len(lag) else None,
                 "cloud_later_by_15min_plus": int((lag > 15).sum()) if len(lag) else 0}
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
