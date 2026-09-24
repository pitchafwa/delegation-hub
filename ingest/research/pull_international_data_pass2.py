"""Second pass for players whose direct URL guess failed in
pull_international_data.py. Direct guessing can't handle real quirks in
Basketball-Reference's own slug generation (confirmed: Thanasis
Antetokounmpo's real slug is "antetokuonmpo", a typo on BBR's own site, not
predictable from correct spelling). Use BBR's own site search instead --
more robust than guessing, since it finds whatever the real URL actually
is, whatever it's spelled like.
"""
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import requests

from pull_international_data import parse_player_page, CACHE, HEADERS, REQUEST_DELAY

ROOT = Path(__file__).resolve().parent


def search_bbref(name):
    """Returns list of candidate /international/players/... paths from
    BBR's own site search -- the REAL slug, whatever it's spelled like."""
    path = CACHE / f"search_{re.sub(r'[^a-z0-9]', '', name.lower())}.html"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        resp = requests.get(
            "https://www.basketball-reference.com/search/search.fcgi",
            params={"search": name}, headers=HEADERS, timeout=20,
        )
        time.sleep(REQUEST_DELAY)
        if resp.status_code != 200:
            return []
        text = resp.content.decode("utf-8")
        path.write_text(text, encoding="utf-8")
    return re.findall(r'href="(/international/players/[^"]+)"', text)


if __name__ == "__main__":
    log = pd.read_csv(ROOT / "data" / "international_pull_log.csv")
    bio = pd.read_csv(ROOT / "data" / "player_bio.csv")
    bio["DRAFT_YEAR"] = pd.to_numeric(bio["DRAFT_YEAR"], errors="coerce")
    bio["DRAFT_NUMBER"] = pd.to_numeric(bio["DRAFT_NUMBER"], errors="coerce")

    already_matched = set(log[log["result"].str.startswith("ok", na=False)]["name"])
    all_attempted_names = set(log["name"])
    still_unmatched = sorted(all_attempted_names - already_matched)
    print(f"Still unmatched after pass 1: {len(still_unmatched)}")

    results = []
    log_rows = []
    for i, name in enumerate(still_unmatched):
        bio_row = bio[(bio["PLAYER_FIRST_NAME"].fillna("") + " " + bio["PLAYER_LAST_NAME"].fillna("")) == name]
        if bio_row.empty:
            continue
        bio_row = bio_row.iloc[0]
        draft_year = bio_row["DRAFT_YEAR"]
        candidates = search_bbref(name)
        matched = False
        for cand_path in candidates[:3]:  # real BBR search rarely returns more than 1-2 real intl matches for a specific name
            url = f"https://www.basketball-reference.com{cand_path}"
            fetch_path = CACHE / (url.rsplit("/", 1)[-1])
            if fetch_path.exists():
                html = fetch_path.read_text(encoding="utf-8")
            else:
                resp = requests.get(url, headers=HEADERS, timeout=20)
                time.sleep(REQUEST_DELAY)
                if resp.status_code != 200:
                    log_rows.append({"name": name, "draft_year": draft_year, "url": url, "result": "fetch_failed"})
                    continue
                html = resp.content.decode("utf-8")
                fetch_path.write_text(html, encoding="utf-8")
            df, note = parse_player_page(html, name, draft_year)
            if df is not None:
                df["real_name"] = name
                df["PERSON_ID"] = bio_row["PERSON_ID"]
                df["real_draft_year"] = draft_year
                df["real_draft_number"] = bio_row["DRAFT_NUMBER"]
                df["bbref_slug"] = url.rsplit("/", 1)[-1]
                results.append(df)
                log_rows.append({"name": name, "draft_year": draft_year, "url": url, "result": note})
                matched = True
                break
            else:
                log_rows.append({"name": name, "draft_year": draft_year, "url": url, "result": f"rejected: {note}"})
        if not matched:
            print(f"STILL UNMATCHED: {name} ({int(draft_year) if pd.notna(draft_year) else '?'}) -- {len(candidates)} search candidates tried")

        if (i + 1) % 15 == 0 or (i + 1) == len(still_unmatched):
            print(f"--- pass 2 progress: {i + 1}/{len(still_unmatched)}, {len(results)} newly matched ---")
            pd.DataFrame(log_rows).to_csv(ROOT / "data" / "international_pull_log_pass2.csv", index=False, encoding="utf-8")
            if results:
                pd.concat(results, ignore_index=True).to_csv(
                    ROOT / "data" / "international_player_seasons_pass2.csv", index=False, encoding="utf-8"
                )

    print(f"\nPass 2: matched {len(results)} / {len(still_unmatched)} previously-unmatched players")
