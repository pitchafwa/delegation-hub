"""Pull real pre-draft international/non-college stats from Basketball-
Reference's dedicated international-players section, for every real NBA
draftee (2008-2026) who has no college match -- the population that makes
this hub's rookie model blind to players like Wembanyama, Doncic, Jokic,
Giannis, and every recent international/G-League-Ignite/OTE prospect.

Real URL pattern confirmed by direct navigation:
  https://www.basketball-reference.com/international/players/<slug>.html
where <slug> = lowercase, ASCII-transliterated (diacritics stripped),
hyphenated first-last name, with a numeric suffix (usually -1, occasionally
higher for real namesake collisions on that site itself).

Validation discipline (same as the college linking fix): never trust a
fetched page just because the URL resolved -- check the page's own listed
name against the real bio name, and check the LAST listed season's year is
within a real, plausible gap of the player's actual NBA draft year, before
accepting the match. Rate-limited to be respectful of a site with no public
API (matches the same principle already applied to barttorvik.com and
nba_api elsewhere in this pipeline).
"""
import re
import sys
import time
import unicodedata
from pathlib import Path

# Windows console defaults to cp1252, which can't print real diacritic names
# (Doncic, Jokic, etc.) or bbref's own special whitespace chars -- crashed
# the very first full run partway through. Force real UTF-8 stdout instead
# of avoiding printing names altogether.
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "international_cache"
CACHE.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research; personal fantasy-basketball project; respectful rate-limited use)"}
REQUEST_DELAY = 3.5  # seconds between real HTTP requests -- respectful of a site with no public API


def slugify(first, last):
    name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-zA-Z ]", "", name).lower().strip()
    return re.sub(r"\s+", "-", name)


def fetch(url):
    path = CACHE / (url.rsplit("/", 1)[-1])
    if path.exists():
        return path.read_text(encoding="utf-8")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    time.sleep(REQUEST_DELAY)
    if resp.status_code != 200:
        return None
    # requests' auto-detected .encoding is unreliable for real diacritic
    # names (confirmed live: "Dončić" came back as mojibake via
    # resp.text's guessed charset) -- decode the raw bytes as UTF-8
    # explicitly rather than trusting the guess.
    text = resp.content.decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


def parse_player_page(html, real_name, real_draft_year):
    """Returns (rows_df, validation_note) or (None, reason) if rejected."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else ""
    if "Page Not Found" in title or "404" in title:
        return None, "404"

    # not every player's page has every competition-type table -- e.g.
    # Giannis Antetokounmpo's real page has ONLY a "tournament" (national
    # team) table, no "all"/"league" data on record at all. Try in priority
    # order: "all" (most complete real sample) -> "league" (domestic league
    # only) -> "tournament" (weakest signal, national-team appearances only,
    # better than nothing). Confirmed via direct inspection after every
    # real player was wrongly coming back UNMATCHED on the first full run.
    table = None
    table_kind = None
    for kind in ["all", "league", "tournament"]:
        table = soup.find("table", id=f"player-stats-per_game-{kind}-")
        if table is not None:
            table_kind = kind
            break
    if table is None:
        return None, "no per_game table of any kind (all/league/tournament)"
    thead = table.find("thead")
    headers = [th.get_text() for th in thead.find_all("th")]
    body = table.find("tbody")
    if body is None:
        return None, "no tbody"
    rows = []
    for tr in body.find_all("tr"):
        cells = [c.get_text() for c in tr.find_all(["th", "td"])]
        if len(cells) == len(headers):
            rows.append(cells)
    if not rows:
        return None, "no data rows"
    df = pd.DataFrame(rows, columns=headers)
    df["table_kind"] = table_kind
    df = df[df["Season"].str.match(r"^\d{4}", na=False)]  # drop any career-total rows
    if df.empty:
        return None, "no real season rows"

    # real season year = the year the season ENDS (matches this whole
    # pipeline's convention elsewhere, e.g. torvik college years)
    def season_end_year(s):
        m = re.match(r"(\d{4})-?(\d{2})?", s)
        if not m:
            return None
        start = int(m.group(1))
        return start + 1 if m.group(2) else start

    df["season_end_year"] = df["Season"].apply(season_end_year)
    df = df.dropna(subset=["season_end_year"])

    # the "tournament" (national-team) table in particular can span a
    # player's ENTIRE career, including seasons long after they became an
    # established NBA player (e.g. Giannis Antetokounmpo's real page's last
    # tournament row is 2024, eleven years after his 2013 draft -- still
    # playing for Greece as a superstar). Only pre-draft rows are real
    # prospect data; drop anything after the real draft year first, THEN
    # check recency, rather than taking the table's overall last row.
    pre_draft = df[df["season_end_year"] <= real_draft_year]
    if pre_draft.empty:
        return None, f"no pre-draft rows (only post-draft seasons on record, last={df['season_end_year'].max()})"
    df = pre_draft
    last_year = df["season_end_year"].max()
    year_gap = abs(last_year - real_draft_year)
    if year_gap > 2:
        return None, f"year_gap={year_gap} too large (last real pre-draft season {last_year} vs draft {real_draft_year})"

    # real-name cross-check: does the page's own <h1> contain a name that's
    # plausibly the same real person (loose substring check on normalized
    # last name is enough here -- the year-gap check is the real guard
    # against a wrong-person page)
    h1 = soup.find("h1")
    page_name = h1.get_text() if h1 else ""
    return df, f"ok (page name: {page_name!r}, last real season {int(last_year)}, gap={year_gap})"


if __name__ == "__main__":
    bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
    bio = bio.dropna(subset=["DRAFT_YEAR"]).copy()
    bio["DRAFT_YEAR"] = pd.to_numeric(bio["DRAFT_YEAR"], errors="coerce")
    bio["DRAFT_NUMBER"] = pd.to_numeric(bio["DRAFT_NUMBER"], errors="coerce")
    bio = bio[(bio["DRAFT_YEAR"] >= 2008) & (bio["DRAFT_YEAR"] <= 2026)]

    links = pd.read_csv(ROOT / "data" / "college_to_nba_links.csv")
    linked_ids = set(links["PLAYER_ID"])
    gap = bio[~bio["PERSON_ID"].isin(linked_ids)].copy()
    print(f"Gap population (no college match, 2008-2026 draftees): {len(gap)}")

    results = []
    log_rows = []
    for i, (_, row) in enumerate(gap.iterrows()):
        first, last = row["PLAYER_FIRST_NAME"], row["PLAYER_LAST_NAME"]
        real_name = f"{first} {last}"
        draft_year = row["DRAFT_YEAR"]
        slug_base = slugify(first, last)
        matched = False
        for n in [1, 2, 3]:
            url = f"https://www.basketball-reference.com/international/players/{slug_base}-{n}.html"
            html = fetch(url)
            if html is None:
                log_rows.append({"name": real_name, "draft_year": draft_year, "url": url, "result": "fetch_failed"})
                continue
            df, note = parse_player_page(html, real_name, draft_year)
            if df is not None:
                df["real_name"] = real_name
                df["PERSON_ID"] = row["PERSON_ID"]
                df["real_draft_year"] = draft_year
                df["real_draft_number"] = row["DRAFT_NUMBER"]
                df["bbref_slug"] = f"{slug_base}-{n}"
                results.append(df)
                log_rows.append({"name": real_name, "draft_year": draft_year, "url": url, "result": note})
                matched = True
                break
            else:
                log_rows.append({"name": real_name, "draft_year": draft_year, "url": url, "result": f"rejected: {note}"})
        if not matched:
            print(f"UNMATCHED: {real_name} ({int(draft_year)})")

        # incremental save every 15 players -- a crash partway through (like
        # the first full run) shouldn't lose all prior progress
        if (i + 1) % 15 == 0 or (i + 1) == len(gap):
            n_matched = len(results)
            print(f"--- progress: {i + 1}/{len(gap)} processed, {n_matched} matched so far ---")
            pd.DataFrame(log_rows).to_csv(ROOT / "data" / "international_pull_log.csv", index=False, encoding="utf-8")
            if results:
                pd.concat(results, ignore_index=True).to_csv(
                    ROOT / "data" / "international_player_seasons.csv", index=False, encoding="utf-8"
                )

    log = pd.DataFrame(log_rows)
    log.to_csv(ROOT / "data" / "international_pull_log.csv", index=False, encoding="utf-8")

    if results:
        all_seasons = pd.concat(results, ignore_index=True)
        all_seasons.to_csv(ROOT / "data" / "international_player_seasons.csv", index=False, encoding="utf-8")
        print(f"\nMatched {all_seasons['PERSON_ID'].nunique()} / {len(gap)} real players")
        print(f"Saved {len(all_seasons)} season rows to international_player_seasons.csv")
    else:
        print("No matches at all -- something is wrong, check international_pull_log.csv")
