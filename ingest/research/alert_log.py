"""Alert log for the site (2026-11-25, Tommy: "I haven't gotten any ntfy notifications in a bit: are there none, or are alerts failing?").
Writes ingest/live_out/alert_log.json: the last 30 days of phone alerts from the ledger (ledger/alerts-YYYY-MM.jsonl) and when this check ran and whether it worked.
alerts.yml publishes it to the live-data branch every run; the site's More > Alerts page reads it. Never includes the ntfy topic.
Run from ingest/:  python research/alert_log.py   (env RUN_OK = the Run step's outcome)
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
now = datetime.now(timezone.utc)
rows = []
for f in sorted((ROOT / "ledger").glob("alerts-*.jsonl"))[-2:]:
    for line in open(f, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") != "alert" or not r.get("ts"):
            continue
        ts = datetime.fromisoformat(r["ts"])
        if now - ts > timedelta(days=30):
            continue
        rows.append({"ts": r["ts"], "title": r.get("title", ""), "body": r.get("body", ""), **({"ok": r["ok"]} if "ok" in r else {})})
rows.sort(key=lambda r: r["ts"], reverse=True)
outcome = os.environ.get("RUN_OK", "")
out = {"check": {"at": now.isoformat(timespec="seconds"), "ok": outcome in ("success", "")}, "alerts": rows[:150]}
d = ROOT / "ingest" / "live_out"
d.mkdir(exist_ok=True)
(d / "alert_log.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"alert log: {len(rows)} alerts in 30 days, check {'ok' if out['check']['ok'] else 'FAILED'}")
