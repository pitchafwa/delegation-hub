"""How did waivers and free-agent adds really behave in this league last season (2025-26, ESPN seasonId 2026)? Pulls every transaction (view mTransactions2, one call per scoring
period) into data/league_transactions_2026.json, then answers: which adds were plain free-agent adds vs waiver claims, how long after a drop did the same player get added,
and at what hour of the day did adds take effect.   Usage: uv run python research/waiver_history_check.py   (from ingest/; add --cached to reuse the saved file)
"""
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
ET = ZoneInfo("America/New_York")
OUT = Path(__file__).resolve().parent / "data" / "league_transactions_2026.json"

if "--cached" in sys.argv and OUT.exists():
    tx = json.load(open(OUT))
else:
    import config
    from espn_api.basketball import League
    lg = League(league_id=config.LEAGUE_ID, year=2026, espn_s2=config.ESPN_S2, swid=config.SWID)
    byid = {}
    for d in range(1, 161):
        try:
            raw = lg.espn_request.league_get(params={"view": "mTransactions2", "scoringPeriodId": d})
        except Exception as e:
            print(d, "error", repr(e)[:80])
            continue
        for t in raw.get("transactions", []):
            byid[t["id"]] = t
        time.sleep(0.25)
    tx = list(byid.values())
    json.dump(tx, open(OUT, "w"))
print("unique transactions:", len(tx))

rows = []
for t in tx:
    ts = t.get("processDate") or t.get("proposedDate")
    for it in t.get("items", []):
        rows.append({"ts": ts, "ttype": t.get("type"), "itype": it.get("type"), "pid": it.get("playerId"), "status": t.get("status"), "exe": t.get("executionType")})
c = collections.Counter((r["ttype"], r["itype"], r["status"]) for r in rows)
print("(transaction type, item type, status): count")
for k, v in sorted(c.items(), key=lambda x: -x[1])[:16]:
    print("  ", v, k)

rows = [r for r in rows if r["ts"]]
rows.sort(key=lambda r: r["ts"])
last_drop, res = {}, collections.defaultdict(list)
for r in rows:
    if r["itype"] == "DROP":
        last_drop[r["pid"]] = r["ts"]
    elif r["itype"] == "ADD" and r["status"] == "EXECUTED":
        d = last_drop.get(r["pid"])
        res[r["ttype"]].append(((r["ts"] - d) / 3.6e6 if d else None, datetime.fromtimestamp(r["ts"] / 1000, ET)))
for k, lst in res.items():
    gaps = sorted(g for g, _ in lst if g is not None)
    print("\n" + f"{k} ADDS executed: {len(lst)}; the player had been dropped earlier in the log for {len(gaps)} of them")
    if gaps:
        print(f"   hours between that drop and the add: min {gaps[0]:.1f}, 10th pct {gaps[len(gaps) // 10]:.1f}, median {gaps[len(gaps) // 2]:.1f}; adds within 24h of the drop: {sum(1 for g in gaps if g < 24)}")
    hrs = collections.Counter(t.hour for _, t in lst)
    print("   hour of day (ET) the add took effect:", dict(sorted(hrs.items())))
