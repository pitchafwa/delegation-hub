"""Lock-in-early study (2026-10-27): can a next-day move be made the night before (after the day's first tip, when roster moves only take effect the next day)
without the next morning's DailyFantasyFuel number overturning it?
  A. When does DFF first post a day's numbers, and how far do they move from first post / the night before to the last number before tip?
  B. How far is our own number (projection_log: st when logged, else the plan's level) from DFF's last number for the same player and game?
  C. How often does a planned add/drop in the last plan run of a night survive into the next day's plans?
Run from the repo root:  ingest/.venv/Scripts/python ingest/research/lock_early_study.py
"""
import csv
import json
import re
import statistics as S
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DFF = ROOT / "ingest/research/data/dfs/log.csv"
PLOG = ROOT / "dashboard/projection_log.csv"
LEDGER = ROOT / "ledger/plan-2026-10.jsonl"
ET = timezone(timedelta(hours=-4))


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summ(label, xs):
    if not xs:
        print(f"  {label}: none")
        return
    a = [abs(x) for x in xs]
    print(f"  {label}: n={len(xs)}  mean {S.mean(xs):+.1f}  typical gap (median abs) {S.median(a):.1f}  80th pct abs {pct(a, .8):.1f}  90th {pct(a, .9):.1f}")


# ---- A. DFF timeline per (game date, player)
hist = defaultdict(list)          # (date, name) -> [(fetched ET, league_proj, inj)]
for r in csv.DictReader(open(DFF, encoding="utf-8")):
    t = datetime.fromisoformat(r["fetched_utc"]).astimezone(ET)
    hist[(r["game_date"], norm(r["name"]))].append((t, float(r["league_proj"] or 0), r["inj"]))
first_post = defaultdict(list)
for (d, _), h in hist.items():
    first_post[d].append(min(t for t, _, _ in h))
print("A. When DFF first posts a day's numbers (median first-seen time per player, ET):")
for d in sorted(first_post):
    ts = sorted(first_post[d])
    print(f"  {d}: {ts[len(ts) // 2]:%a %m-%d %I:%M%p}")


def val_at(h, cutoff):
    """last DFF value fetched at or before cutoff (the monitor logs a row only when a number changes)"""
    v = [x for x in sorted(h) if x[0] <= cutoff]
    return v[-1] if v else None


night_move, first_move = [], []
final = {}
for (d, n), h in hist.items():
    h = sorted(h)
    tip = datetime.fromisoformat(d).replace(hour=19, tzinfo=ET)            # the first tip; numbers after it are mid-slate
    last = val_at(h, tip + timedelta(hours=4))
    if not last or last[1] <= 0:                                           # ruled out late or not playing: a different question (status, not projection)
        continue
    final[(d, n)] = last[1]
    nb = val_at(h, tip - timedelta(hours=24) + timedelta(hours=3))          # what the plan could see at 10pm the night before
    if nb and nb[1] > 0:
        night_move.append(last[1] - nb[1])
    first_move.append(last[1] - h[0][1] if h[0][1] > 0 else 0)
print("\nDFF's own movement to its last pre-tip number (players who played):")
summ("from its 10pm-the-night-before number", night_move)
summ("from its first posted number", first_move)

# ---- B. our number vs DFF's last number
ours = []
by_kind = defaultdict(list)
for r in csv.DictReader(open(PLOG, encoding="utf-8")):
    k = (r["date"], norm(r["name"]))
    o = r.get("st") or r.get("level") or r.get("model")
    if k in final and o:
        diff = final[k] - float(o)
        ours.append(diff)
        by_kind["free agents" if r.get("is_fa") == "1" else "rostered"].append(diff)
print("\nB. DFF's last number minus our number (same player, same game):")
summ("all", ours)
for k, v in by_kind.items():
    summ(k, v)

# ---- C. do a night's planned moves survive into the next day?
plans = [json.loads(l) for l in open(LEDGER, encoding="utf-8")]
plans = [p for p in plans if p["kind"] == "plan"]
by_day = defaultdict(list)
for p in plans:
    by_day[datetime.fromisoformat(p["ts"]).astimezone(ET).date()].append(p)
print("\nC. Planned adds in the last run before midnight, checked against the next day's last plan before 7pm:")
kept = total = 0
for d in sorted(by_day):
    nxt = by_day.get(d + timedelta(days=1))
    if not nxt:
        continue
    late = by_day[d][-1]
    nx = [p for p in nxt if datetime.fromisoformat(p["ts"]).astimezone(ET).hour < 19] or nxt[:1]
    later = {(m["add"], m["drop"]) for m in nx[-1]["sequence"]}
    later_adds = {m["add"] for m in nx[-1]["sequence"]}
    for m in late["sequence"]:
        total += 1
        same = (m["add"], m["drop"]) in later
        kept += same
        tag = "kept" if same else ("same add, other drop" if m["add"] in later_adds else "GONE")
        print(f"  {d} {m['add']:<24} for {m['drop']:<24} gain {m['gain']:5.1f}  -> {tag}")
print(f"  kept {kept} of {total}")
