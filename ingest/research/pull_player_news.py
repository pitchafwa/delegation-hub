"""Recent news per player for the player card (dashboard/player_news.json, keyed by ESPN player id).

Source: ESPN's public fantasy player-news feed (the same items ESPN's own fantasy app shows). Most are RotoWire's player-news blurbs, which name
the reporter or outlet they came from ("per Casey Holdahl of the Trail Blazers' official site"), so beat-writer reports on injuries, minutes, role
and coach comments come through; No cookies needed, so this also runs in GitHub Actions.

Players covered: everyone on the redraft board plus everyone on a league roster. Box-score recap blurbs ("recorded 27 points (9-15 FG)...") are
dropped, since the card already shows the game log. The page loads this file only when a player card is opened.
Run:  uv run python research/pull_player_news.py   (from ingest/)
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
URL = "https://site.api.espn.com/apis/fantasy/v2/games/fba/news/players"
KEEP, MAX_AGE_DAYS, STORY_CHARS = 5, 120, 420
RECAP = re.compile(r"\b(recorded|produced|registered|finished with|scored|posted|totaled|tallied|logged|ended)\b.*\d+\s+(points|rebounds|assists)|\(\d+-\d+ FG", re.I)


def ids():
    out = set()
    for fn, getter in (("redraft_data.json", lambda d: [p["id"] for p in d["players"]]),
                       ("league_rosters.json", lambda d: [r["espn_id"] for t in d["teams"] for r in t["roster"]])):
        f = HUB / fn
        if f.exists():
            out.update(int(x) for x in getter(json.load(open(f, encoding="utf-8"))) if x)
    return sorted(out)


def fetch(pid):
    for attempt in range(3):
        try:
            r = requests.get(URL, params={"playerId": pid, "limit": 25}, timeout=25)
            if r.status_code == 200:
                return pid, r.json().get("feed", [])
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return pid, None


def clean(item, cutoff):
    if item.get("type") != "Rotowire":      # the other items are generic ESPN stories/listicles, not player-specific reports
        return None
    pub = item.get("published") or ""
    if not pub or pub < cutoff:
        return None
    head = (item.get("headline") or "").strip()
    story = (item.get("story") or item.get("description") or "").strip()
    if not head or RECAP.search(head):
        return None
    story = re.sub(r"<[^>]+>", "", story)
    story = story if story != head else ""
    if len(story) > STORY_CHARS:
        story = story[:STORY_CHARS].rsplit(" ", 1)[0] + "…"
    return {"t": pub, "h": head, "s": story, "src": "RotoWire"}


if __name__ == "__main__":
    pids = ids()
    print(f"fetching news for {len(pids)} players")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    news, failed = {}, 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pid, feed in ex.map(fetch, pids):
            if feed is None:
                failed += 1
                continue
            items = [c for c in (clean(i, cutoff) for i in feed) if c]
            items.sort(key=lambda c: c["t"], reverse=True)
            if items:
                news[str(pid)] = items[:KEEP]
    if failed > len(pids) * 0.2:
        sys.exit(f"refusing to write: {failed}/{len(pids)} fetches failed")
    out = {"generated": datetime.now(timezone.utc).isoformat(), "players": news}
    (HUB / "player_news.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote player_news.json: {len(news)} players with news, {failed} failed, {round((HUB / 'player_news.json').stat().st_size / 1024)} KB")
