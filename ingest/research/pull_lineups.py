"""Pull today's NBA starting lineups (confirmed + projected) from RotoWire's free lineups page -- the standard industry source for this, and
the only free one that has it at all (ESPN's own API has nothing before a game starts; checked live 2026-09-30).

IMPORTANT, READ BEFORE TRUSTING THIS: as of 2026-09-30 (writing this), RotoWire's free page does not cover preseason games at all ("there are
no games on the NBA schedule today" even on a day ESPN confirms a real preseason game exists), and next-day lineups are paywalled ("tomorrow's
schedule is reserved for subscribers") -- so this can only ever tell you TODAY's lineups, once posted (usually a couple hours before tip), and
it has NEVER been tested against a real page with real games on it. The parser below is a best-effort guess at RotoWire's structure. Every run
saves the raw HTML to data/lineups_debug/ and prints exactly how many games/players it found, specifically so a wrong guess is obvious and
fixable the first time this runs on a real game day, rather than silently producing empty or wrong data forever. DO NOT wire this into
projections with real weight until its first live run has been manually spot-checked against the actual page.

Output: dashboard/lineups_today.json -- {"generated":..., "has_data": bool, "teams": {"BOS": {"player key": {"starting": true, "confirmed":
true, "pct_note": "..."}}}}. "has_data": false means no games were found (either a real off day, or a parse failure) -- callers must treat that
as "no information," not "confirmed nobody starts."
Usage: uv run python research/pull_lineups.py
"""
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
HUB = ROOT.parent.parent / "dashboard"
DEBUG = ROOT / "data" / "lineups_debug"
DEBUG.mkdir(parents=True, exist_ok=True)
URL = "https://www.rotowire.com/basketball/nba-lineups.php"
HDR = {"User-Agent": "Mozilla/5.0"}


def norm(name):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())


def fetch():
    try:
        r = requests.get(URL, headers=HDR, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as ex:
        print(f"fetch failed: {ex}")
        return None


def parse(html):
    """Best-effort parse of RotoWire's lineup boxes. Returns {team_abbrev: {norm_name: {"starting": bool, "confirmed": bool, "pct_note": str}}}.
    RotoWire's real markup has never been confirmed against a live game -- this tries a few plausible, well-known-pattern selectors and reports
    what it finds; if the site's real structure differs, this will find 0 games and say so loudly rather than guess wrong silently."""
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    # RotoWire's lineup pages conventionally wrap each game in a "lineup" box with two "lineup__list" columns (away/home) of "lineup__player"
    # rows; a player still questionable for tonight's game usually carries an "is-pct" (or similar) class noting his position in the color-bar
    # legend (0/25/50/75/100% chance), and the starting two-way vs confirmed/expected wording lives in a nearby status label.
    boxes = soup.select(".lineup") or soup.select("[class*='lineup-list']") or soup.select("[class*='lineup__box']")
    for box in boxes:
        team_els = box.select("[class*='lineup__abbr'], [class*='lineup__team']")
        lists = box.select("[class*='lineup__list']")
        if len(team_els) < 2 or len(lists) < 2:
            continue
        for team_el, plist in zip(team_els, lists):
            team = team_el.get_text(strip=True).upper()
            players = {}
            for i, p_el in enumerate(plist.select("[class*='lineup__player']")):
                a = p_el.select_one("a") or p_el
                name = a.get_text(strip=True)
                if not name:
                    continue
                cls = " ".join(p_el.get("class", []))
                confirmed = "is-pct" not in cls  # RotoWire's own legend: a player with NO color bar is expected to play as listed
                # raw name kept alongside the key so a caller can re-normalize with its OWN name-matching convention (e.g. build_week_plan.py's
                # norm() strips Jr/Sr/III suffixes that this file's simpler norm() does not -- storing both avoids two normalizers disagreeing)
                players[norm(name)] = {"name": name, "starting": i < 5, "confirmed": confirmed, "pct_note": p_el.get("title", "")}
            if players:
                out[team] = players
    return out


if __name__ == "__main__":
    html = fetch()
    now = datetime.now(timezone.utc).isoformat()
    if html is None:
        result = {"generated": now, "has_data": False, "reason": "fetch failed", "teams": {}}
    else:
        debug_path = DEBUG / f"{now[:10]}.html"
        debug_path.write_text(html, encoding="utf-8")
        teams = parse(html)
        n_players = sum(len(v) for v in teams.values())
        print(f"found {len(teams)} teams, {n_players} players" + (f" (raw HTML saved to {debug_path.name} for inspection)" if not teams else ""))
        if "no games on the nba schedule" in html.lower():
            print("RotoWire itself says there are no games today (real off day, or preseason -- not a parse failure)")
        result = {"generated": now, "has_data": bool(teams), "teams": teams}
    (HUB / "lineups_today.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"wrote dashboard/lineups_today.json (has_data={result['has_data']})")
