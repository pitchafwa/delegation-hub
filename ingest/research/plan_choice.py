"""Keep / pass choices from the Do now card (2026-11-09, Tommy): "keep X" means the plan may not drop X, "pass" means the plan may not add him. Both end when the matchup ends.
dashboard/plan_choices.json: {"keep": [{"id", "name", "until", "at"}], "pass": [...]}. build_week_plan.py reads it; the site lists it under More and can undo an entry.
Run by .github/workflows/plan-choice.yml:  python research/plan_choice.py --action keep|pass|undo --id 123 --name "Player Name" --until 2026-11-15
"""
import argparse
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

F = Path(__file__).resolve().parent.parent.parent / "dashboard" / "plan_choices.json"


def load(path=F):
    try:
        c = json.load(open(path, encoding="utf-8"))
    except Exception:
        c = {}
    return {"keep": list(c.get("keep") or []), "pass": list(c.get("pass") or [])}


def active(today, path=F):
    """(entries still in force, {"keep": {id: name}, "pass": {id: name}})"""
    c = load(path)
    t = today.isoformat()
    c = {k: [e for e in v if str(e.get("until") or "") >= t] for k, v in c.items()}
    return c, {k: {int(e["id"]): e.get("name") for e in v} for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=["keep", "pass", "undo"])
    ap.add_argument("--id", required=True, type=int)
    ap.add_argument("--name", default="")
    ap.add_argument("--until", default="")
    a = ap.parse_args()
    now = datetime.now(ZoneInfo("America/New_York"))
    until = a.until if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.until or "") else (now.date() + timedelta(days=7)).isoformat()
    date.fromisoformat(until)
    name = re.sub(r"[^\w .'\-]", "", a.name, flags=re.UNICODE)[:60]
    c = load()
    today = now.date().isoformat()
    for k in c:                                    # one entry per player; expired ones fall off here
        c[k] = [e for e in c[k] if int(e["id"]) != a.id and str(e.get("until") or "") >= today]
    if a.action != "undo":
        c[a.action].append({"id": a.id, "name": name, "until": until, "at": now.isoformat(timespec="minutes")})
    F.write_text(json.dumps(c, indent=1), encoding="utf-8")
    print(a.action, a.id, name, "->", json.dumps(c))


if __name__ == "__main__":
    main()
