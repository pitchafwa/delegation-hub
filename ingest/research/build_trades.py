"""Trade finder v1 -> dashboard/trade_ideas.json.   (First pass: logic is transparent, NOT backtested; see notes below.)

The idea: in a league where each team keeps only K players, the price of a player is not the same for everybody.
  * MARKET value (what an ordinary owner thinks): Hashtag's crowd dynasty ranking value. A trade is *acceptable* to the other side if they
    get about as much market value as they give up.
  * PERSONAL value (what a player is worth to THIS roster): (a) his contribution to this season's lineup, and (b) his keeper value, which
    only exists if he would make that team's top-K assets. A star who is a team's 7th-best asset is worth little to it as a keeper; the
    same star could be a top-3 keeper for a team that lacks them. That gap is what a mutually good trade lives in.
So we search 1-for-1, 2-for-1 and 1-for-2 trades between every pair of teams and keep the ones that (1) raise MY personal value,
(2) look fair to the other side in MARKET terms, and (3) usually help them personally too (win-win).

Personal value of a roster = season points + keeper points, each computed from what is on the roster after the trade:
  season  = (sum of the best 10 players' expected points/g x availability + 0.25 x the next 5) x 3.3 games/week x weeks left
            (rosters are padded to 15 with replacement-level players, so trading depth away costs what it really costs)
  keeper  = sum of the top-K dynasty assets (asset value at K keepers); the site converts it to points with a weight the user can change.
Both parts are stored separately so the page can re-weight instantly. Ratio to remember: 1 asset point = one point per game for one
season = about 73 points of output; the default weight (0.10) heavily discounts the ten-year horizon.
Inputs: hub_data.json (asset values), league_rosters.json + week_plan.json (rosters, ESPN-first levels, statuses), Hashtag CSV (market value).
Run from ingest/:  uv run python research/build_trades.py
"""
import itertools
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent
HUB = R.parent.parent / "dashboard"
K = 5
GAMES_PER_WEEK, WEEKS_LEFT = 3.3, 20
REPLACEMENT_LEVEL, AVAIL = 22.0, 0.90
ASSET_TO_PTS = 73.0
MEANINGFUL_RANK, MEANINGFUL_LEVEL = 260, 24.0      # only players who matter to someone are traded (keeps the search small)
WEIGHTS = (0.05, 0.10, 0.30)                        # keeper weights used to pre-select the ideas kept in the file


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


hub = json.load(open(HUB / "hub_data.json", encoding="utf-8"))
hubp = {p["id"]: p for p in hub["players"]}
lr = json.load(open(HUB / "league_rosters.json", encoding="utf-8"))
wp = json.load(open(HUB / "week_plan.json", encoding="utf-8"))
lvl = {}
for t in wp["teams"]:
    for p in t["roster"]:
        lvl[p["id"]] = (p["level"], p["status"], p["ir"])
hk = pd.read_csv(R / "data" / "hashtag_dynasty_latest.csv")
# Hashtag's VALUE is compressed (rank 1 = 2530, rank 100 = 1336, rank 430 = 998): most of it is a shared baseline. Trade "price" is the part above
# a replaceable player (about rank 420 = 1000), so a bench guy is worth ~0 and a star is worth many mid players.
MARKET_BASELINE = 1000.0
mkt = {norm(r.player): max(0.0, float(r.value) - MARKET_BASELINE) for r in hk.itertuples()}


def mk_player(r):
    hp = hubp.get(r["id"]) if r.get("id") else None
    level, status, ir_planned = lvl.get(r["espn_id"], (hp["year0_ppg"] if hp else REPLACEMENT_LEVEL, "ACTIVE", r.get("slot") == "IR"))
    asset = hp["asset_k"][K] if hp and hp.get("asset_k") else 0.0
    avail = 0.0 if status in ("OUT", "INJURY_RESERVE", "SUSPENSION") else AVAIL
    return {"id": r["espn_id"], "name": r["name"], "asset": float(asset), "level": float(level), "avail": avail, "mkt": mkt.get(norm(r["name"]), 0.0),
            "ir": bool(ir_planned), "age": hp["age"] if hp else None}


REPL = {"id": -1, "name": "(replacement)", "asset": 0.0, "level": REPLACEMENT_LEVEL, "avail": AVAIL, "mkt": 0.0, "ir": False, "age": None}


def team_parts(players):
    """(season points, keeper asset points) for a roster; pads to 15 with replacement level"""
    ps = list(players)
    while len([p for p in ps if not p["ir"]]) < 15:
        ps.append(REPL)
    ef = sorted((p["level"] * p["avail"] for p in ps if not p["ir"]), reverse=True)
    season = (sum(ef[:10]) + 0.25 * sum(ef[10:15])) * GAMES_PER_WEEK * WEEKS_LEFT
    keeper = sum(sorted((p["asset"] for p in ps), reverse=True)[:K])
    return season, keeper


def keeper_names(players):
    return [p["name"] for p in sorted(players, key=lambda p: -p["asset"])[:K] if p["asset"] > 0]


teams = []
for t in lr["teams"]:
    ps = [mk_player(r) for r in t["roster"]]
    teams.append({"id": t["id"], "abbrev": t["abbrev"], "name": t["name"], "players": ps})
base = {t["id"]: team_parts(t["players"]) for t in teams}


def meaningful(p):
    return p["asset"] > 0 and p["asset"] >= 3.0 or p["level"] >= MEANINGFUL_LEVEL


out_teams = []
for A in teams:
    a_pool = [p for p in A["players"] if meaningful(p)]
    ideas = []
    for B in teams:
        if B["id"] == A["id"]:
            continue
        b_pool = [p for p in B["players"] if meaningful(p)]
        combos = [(g, r) for g in itertools.combinations(a_pool, 1) for r in itertools.combinations(b_pool, 1)]
        combos += [(g, r) for g in itertools.combinations(a_pool, 2) for r in itertools.combinations(b_pool, 1)]
        combos += [(g, r) for g in itertools.combinations(a_pool, 1) for r in itertools.combinations(b_pool, 2)]
        for give, get in combos:
            give_ids, get_ids = {p["id"] for p in give}, {p["id"] for p in get}
            a_after = [p for p in A["players"] if p["id"] not in give_ids] + list(get)
            b_after = [p for p in B["players"] if p["id"] not in get_ids] + list(give)
            sa, ka = team_parts(a_after)
            sb, kb = team_parts(b_after)
            dsa, dka = sa - base[A["id"]][0], ka - base[A["id"]][1]
            dsb, dkb = sb - base[B["id"]][0], kb - base[B["id"]][1]
            give_mkt, get_mkt = sum(p["mkt"] for p in give), sum(p["mkt"] for p in get)
            # B receives `give`, gives `get`.  ratio = market value B receives / market value B gives
            ratio = (give_mkt / get_mkt) if get_mkt > 0 else (9.9 if give_mkt > 0 else 1.0)
            ideas.append({"p": B["id"], "give": [p["name"] for p in give], "get": [p["name"] for p in get], "me": [round(dsa), round(dka, 1)], "them": [round(dsb), round(dkb, 1)],
                          "mk": [round(give_mkt), round(get_mkt)], "ratio": round(min(ratio, 9.9), 2), "kbefore": keeper_names(A["players"]), "kafter": keeper_names(a_after)})
    # keep the ideas that are good for me at any of several keeper weights and not absurd for the market
    keep = {}
    for w in WEIGHTS:
        scored = sorted(((i["me"][0] + w * ASSET_TO_PTS * i["me"][1], n) for n, i in enumerate(ideas) if 0.8 <= i["ratio"] <= 4.0 and i["me"][0] + w * ASSET_TO_PTS * i["me"][1] > 100), reverse=True)
        for _, n in scored[:40]:
            keep[n] = ideas[n]
    ks, kk = base[A["id"]]
    out_teams.append({"id": A["id"], "abbrev": A["abbrev"], "name": A["name"], "keepers": keeper_names(A["players"]), "keeper_asset": round(kk, 1),
                      "ideas": list(keep.values())})
    print(f"{A['abbrev']:5s} {len(ideas):6d} candidate trades -> {len(keep)} kept", flush=True)

meta = {"generated": datetime.now(timezone.utc).isoformat(), "K": K, "weeks_left": WEEKS_LEFT, "asset_to_points": ASSET_TO_PTS, "default_keeper_weight": 0.10,
        "teams": {t["id"]: {"abbrev": t["abbrev"], "name": t["name"]} for t in out_teams}}
(HUB / "trade_ideas.json").write_text(json.dumps({"meta": meta, "teams": out_teams}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote trade_ideas.json", round((HUB / "trade_ideas.json").stat().st_size / 1024), "KB")
