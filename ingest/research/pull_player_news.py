"""Recent news per player for the player card (dashboard/player_news.json, keyed by ESPN player id).

Source: ESPN's public fantasy player-news feed (the same items ESPN's own fantasy app shows). Most are RotoWire's player-news blurbs, which name
the reporter or outlet they came from ("per Casey Holdahl of the Trail Blazers' official site"), so beat-writer reports on injuries, minutes, role
and coach comments come through; No cookies needed, so this also runs in GitHub Actions.

Players covered: everyone on the redraft board plus everyone on a league roster. RotoWire's game recaps are kept (2026-11-15, Tommy: about half
carry role, minutes or injury notes under the stat line, and ESPN's card shows them); ESPN's highlight clips and general articles are dropped.
Also Bluesky (2026-11-15, Tommy: why a player's projection moved -- a teammate back, a new starter, a growing role): Rotoworld (NBC's own writers,
a second analysis voice beside RotoWire), SportsEthos (fast injury/lineup notes citing beat writers) and Basketball Monster's status alerts, read
from Bluesky's free public API. Posts are matched to every player they name in full (a lineup post lands on each starter), near-duplicates of the
same news are merged (the fuller version wins), and each item links back to the original.
The page loads this file only when a player card is opened.
Run:  uv run python research/pull_player_news.py   (from ingest/)
"""
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")
HUB = Path(__file__).resolve().parent.parent.parent / "dashboard"
URL = "https://site.api.espn.com/apis/fantasy/v2/games/fba/news/players"
KEEP, MAX_AGE_DAYS, STORY_CHARS = 10, 120, 420


BSKY = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
BSKY_SOURCES = {"rotoworld-hoops.bsky.social": "Rotoworld", "ethosfantasynba.bsky.social": "SportsEthos", "bbmalerts.bsky.social": "Basketball Monster"}
PRIORITY = {"RotoWire": 0, "Rotoworld": 1, "SportsEthos": 2, "Basketball Monster": 3}     # which version of the same news to keep
BSKY_PAGES = 4                                                                              # x100 posts per account: about 3-4 weeks back
STOP = {"the", "a", "an", "to", "for", "on", "in", "vs", "at", "of", "and", "with", "is", "will", "be", "sunday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "game", "nba", "fantasybasketball"}


def norm_name(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[.'’]", "", t)
    t = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def name_index():
    """normalized full name -> espn id, for everyone the news covers (names shared by two players are dropped)"""
    seen = {}
    for fn, getter in (("redraft_data.json", lambda d: [(p["id"], p["name"]) for p in d["players"]]),
                       ("league_rosters.json", lambda d: [(r["espn_id"], r.get("name")) for t in d["teams"] for r in t["roster"]])):
        f = HUB / fn
        if f.exists():
            for pid, nm in getter(json.load(open(f, encoding="utf-8"))):
                if pid and nm and len(norm_name(nm).split()) >= 2:
                    seen.setdefault(norm_name(nm), set()).add(int(pid))
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def bluesky_items(names, cutoff):
    """{espn id: [items]} from the Bluesky accounts in BSKY_SOURCES"""
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")
    out, n_posts = {}, 0
    for handle, src in BSKY_SOURCES.items():
        cursor = None
        for _ in range(BSKY_PAGES):
            try:
                r = requests.get(BSKY, params={"actor": handle, "limit": 100, "filter": "posts_no_replies", **({"cursor": cursor} if cursor else {})}, timeout=25)
                j = r.json() if r.status_code == 200 else {}
            except (requests.RequestException, ValueError):
                j = {}
            feed = j.get("feed", [])
            for x in feed:
                post = x.get("post", {})
                if post.get("author", {}).get("handle") != handle:          # reposts of other accounts
                    continue
                rec, emb = post.get("record", {}), post.get("embed") or {}
                pub = (rec.get("createdAt") or "")[:19] + "Z"
                if pub < cutoff:
                    continue
                ext = emb.get("external") or {}
                text = re.sub(r"\s*#\w+", "", rec.get("text") or "").strip()
                head = (ext.get("title") or "").strip() if src == "Rotoworld" and ext.get("title") else text
                head = re.sub(r"\s+", " ", head)
                if not head or len(head) < 12:
                    continue
                if src == "Rotoworld" and re.match(r"^[A-Z][A-Za-z0-9. ]{1,24} starts? [^,]+(, [^,]+){3,}$", head):          # routine five-man lineups (changes get their own post)
                    continue
                if src == "Basketball Monster" and re.match(r"^[A-Z][\w .'-]+ - Starting - [^:]+:", head):     # whole-team lineup posts: one on every starter, every game
                    continue
                story = (ext.get("description") or "").strip() if src == "Rotoworld" else ""
                story = "" if story == head else story
                if len(story) > STORY_CHARS:
                    story = story[:STORY_CHARS].rsplit(" ", 1)[0] + "…"
                rkey = (post.get("uri") or "").rsplit("/", 1)[-1]
                url = ext.get("uri") if src == "Rotoworld" and ext.get("uri") else f"https://bsky.app/profile/{handle}/post/{rkey}"
                hits = {names[m] for m in pat.findall(norm_name(head + " " + story))}
                n_posts += 1
                for pid in hits:
                    out.setdefault(pid, []).append({"t": pub, "h": head, "s": story, "src": src, "u": url})
            cursor = j.get("cursor")
            if not cursor or not feed or (feed[-1].get("post", {}).get("record", {}).get("createdAt") or "") < cutoff:
                break
    print(f"bluesky: {n_posts} posts, matched to {len(out)} players")
    return out


STATUS = [("out", r"(?<!\w)(ruled out|will not play|won't play|not play|out for|- out -|downgraded to out|sidelined|will miss)(?!\w)"),
          ("gtd", r"(?<!\w)(questionable|doubtful|game-time decision|probable)(?!\w)"),
          ("in", r"(?<!\w)(available|will play|set to play|cleared to play|- playing -|upgraded to available|good to go|active)(?!\w)"),
          ("game", r"(?<!\w)(\d+ points|\d+-\d+-\d+|\d+/\d+/\d+)(?!\w)"),      # one game's stat line, from any outlet
          ("start", r"(?<!\w)(will start|to start|starting|starts|spot start|first start|enter the starting)(?!\w)")]


def status_of(it):
    t = it["h"].lower()                     # the headline only: RotoWire's paragraph often recaps the earlier status
    for k, rx in STATUS:
        if re.search(rx, t):
            return k
    return None


def words(h):
    return {w for w in norm_name(h).split() if w not in STOP and len(w) > 1}


def merge(items):
    """drop near-duplicates of the same news (headlines sharing most words within 12 hours), keeping the fullest source"""
    kept = []
    for it in sorted(items, key=lambda c: (PRIORITY.get(c["src"], 9), -len(c.get("s") or ""))):
        w, st = words(it["h"]), status_of(it)
        dup = False
        for k in kept:
            gap = abs((datetime.fromisoformat(k["t"].replace("Z", "+00:00")) - datetime.fromisoformat(it["t"].replace("Z", "+00:00"))).total_seconds())
            if st and gap <= 8 * 3600 and status_of(k) == st:          # the same status update from another outlet
                dup = True
                break
            if gap <= 12 * 3600:
                kw = words(k["h"])
                if w and kw and len(w & kw) / min(len(w), len(kw)) >= 0.6:
                    dup = True
                    break
        if not dup:
            kept.append(it)
    return sorted(kept, key=lambda c: c["t"], reverse=True)


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
    if not head:
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
    try:
        bsky = bluesky_items(name_index(), cutoff)
    except Exception as ex:                                   # Bluesky is extra: never block RotoWire's news
        print("bluesky unavailable:", ex)
        bsky = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pid, feed in ex.map(fetch, pids):
            if feed is None:
                failed += 1
                continue
            items = merge([c for c in (clean(i, cutoff) for i in feed) if c] + bsky.get(pid, []))
            if items:
                news[str(pid)] = items[:KEEP]
    if failed > len(pids) * 0.2:
        sys.exit(f"refusing to write: {failed}/{len(pids)} fetches failed")
    out = {"generated": datetime.now(timezone.utc).isoformat(), "players": news}
    (HUB / "player_news.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote player_news.json: {len(news)} players with news, {failed} failed, {round((HUB / 'player_news.json').stat().st_size / 1024)} KB")
