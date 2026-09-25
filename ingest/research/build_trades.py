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
# HOW OWNERS IN THIS LEAGUE VALUE PLAYERS.  The Hashtag crowd ranks dynasty value (youth-heavy), but this league drafts a redraft-style ADP board with
# keepers, so owners also weigh current production. We blend the two: blended rank = average of Hashtag dynasty rank and ESPN ADP (players with no
# ADP use the Hashtag rank alone). Rank is then converted to a trade value with Hashtag's own rank->value curve, taking only the part above a
# replaceable player (VALUE is compressed: rank 1 = 2530, rank 100 = 1336, rank 430 = 998; baseline 1000), so a bench guy is worth ~0.
MARKET_BASELINE = 1000.0
import numpy as np
_hk = hk.sort_values("rank")
_ranks = _hk["rank"].to_numpy(dtype=float)
_m = np.maximum(_hk["value"].to_numpy(dtype=float) - MARKET_BASELINE, 0.0)
_m = np.minimum.accumulate(_m)                         # make the curve non-increasing
h_rank = {norm(r.player): float(r.rank) for r in hk.itertuples()}
adp = {norm(p["player"]): p["adp"] for p in hub["players"] if p.get("adp")}


def market_value(name):
    n = norm(name)
    h = h_rank.get(n)
    a = adp.get(n)
    if h is None and a is None:
        return 0.0
    r = h if a is None else (a if h is None else 0.5 * h + 0.5 * a)
    return float(np.interp(r, _ranks, _m, right=0.0))


mkt = {}
# KeepTradeCut-style "package adjustment": people prefer one elite player to several mid ones ("four quarters don't equal a dollar"), so a raw price
# sum is wrong. KTC compares the sums of RAW ADJUSTMENT values, each a player's value times a share that rises steeply with how close he is to the best
# asset in the trade (t) and in the league (v) (published range 10%-42.4% of value; the exact constants are not public). Our own version, softer than the
# first attempt (which made a star worth more than a slightly lesser star plus a second player):
#     raw(p) = p * (0.10 + s * (0.20*(p/t)^4 + 0.124*(p/v)^1.3)),   s = star-premium strength (page slider, default 0.5; 0 = plain sum)
V_MAX = float(_m[0])
S_DEFAULT = 0.5


def raw_adj(p, t, s=S_DEFAULT):
    if p <= 0:
        return 0.0
    return p * (0.10 + s * (0.20 * (p / t) ** 4 + 0.124 * (p / V_MAX) ** 1.3))


def package_ratio(recv, give, s=S_DEFAULT):
    """market ratio for the side that RECEIVES `recv` and gives up `give`: >1 means they come out ahead"""
    t = max([p["mkt"] for p in recv] + [p["mkt"] for p in give] + [1.0])
    a = sum(raw_adj(p["mkt"], t, s) for p in recv)
    b = sum(raw_adj(p["mkt"], t, s) for p in give)
    return (a / b) if b > 0 else (9.9 if a > 0 else 1.0)


# ---------- who values what: team tendencies (draft profiles), competitive window (roster strength), and last-season anchoring
SIGS = json.load(open(HUB / "regression_signals.json", encoding="utf-8")) if (HUB / "regression_signals.json").exists() else {"players": {}, "study": {}}
SIG = SIGS["players"]
PROF = json.load(open(HUB / "team_profiles.json", encoding="utf-8")) if (HUB / "team_profiles.json").exists() else {"teams": []}


def _z_axes():
    ts = [t for t in PROF["teams"] if t["metrics"].get("picks", 0) >= 8]
    def col(k):
        v = pd.Series([t["metrics"].get(k) for t in ts], dtype=float)
        return v.mean(), (v.std() or 1.0)
    st = {k: col(k) for k in ("age_mean", "age_early", "rookie_share", "injury_gamble")}
    out = {}
    for t in ts:
        m = t["metrics"]
        z = lambda k: 0.0 if m.get(k) is None else (m[k] - st[k][0]) / st[k][1]
        out[t["id"]] = {"youth": -(z("age_mean") + z("age_early")) / 2, "rookie": z("rookie_share"), "injury": z("injury_gamble"), "seasons": t["seasons"],
                        "tags": [g["label"] for g in t["tags"][:2]]}
    return out


TRAITS = _z_axes()


def mk_player(r):
    hp = hubp.get(r["id"]) if r.get("id") else None
    level, status, ir_planned = lvl.get(r["espn_id"], (hp["year0_ppg"] if hp else REPLACEMENT_LEVEL, "ACTIVE", r.get("slot") == "IR"))
    asset = hp["asset_k"][K] if hp and hp.get("asset_k") else 0.0
    avail = 0.0 if status in ("OUT", "INJURY_RESERVE", "SUSPENSION") else AVAIL
    hid = r.get("id")
    return {"id": r["espn_id"], "name": r["name"], "asset": float(asset), "level": float(level), "avail": avail, "mkt": market_value(r["name"]),
            "ir": bool(ir_planned), "age": hp["age"] if hp else None, "hid": hid, "status": status, "rookie": bool(hp and hp.get("kind") == "prospect"),
            "surprise": (SIG.get(hid) or {}).get("surprise", 0.0) if hid else 0.0}


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
_rank = sorted(teams, key=lambda t: -base[t["id"]][0])
WINDOW = {}
for i, t in enumerate(_rank):
    c = 1 - i / max(len(_rank) - 1, 1)                       # 1 = strongest roster (contender), 0 = weakest (rebuilder)
    WINDOW[t["id"]] = {"c": c, "w": (c - 0.5) * 2, "label": "contender" if c >= 0.7 else ("rebuilding" if c <= 0.3 else "in the middle"), "rank": i + 1}
    print(f"   window {t['abbrev']:5s} strength rank {i + 1:2d} -> {WINDOW[t['id']]['label']}")
def buy_sell(A):
    """flags from regression_signals: players on A's roster that look overvalued (sell-high) and players on other rosters that look undervalued (buy-low)"""
    nice = {"spike_or_slump": "played well above what last season and normal aging predicted", "minutes_change": "a minutes change that tends to partly revert",
            "games_missed": "an injury-shortened season", "shooting": "shooting luck versus his own career"}
    sell, buy = [], []
    for T in teams:
        for p in T["players"]:
            g = SIG.get(p["hid"]) if p["hid"] else None
            if not g or not g.get("label") or max(g["fpg"], g.get("hist") or 0) < 28:
                continue
            row = {"name": p["name"], "team": T["abbrev"], "fpg": g["fpg"], "hist": g.get("hist"), "extra": g["pred_delta"], "surprise": g.get("surprise"),
                   "why": [nice.get(d["k"], d["k"]) for d in g["drivers"][:2]], "label": g["label"], "market_rank": (hubp.get(p["hid"]) or {}).get("market_rank")}
            if T["id"] == A["id"] and g["label"] == "sell_high":
                sell.append(row)
            elif T["id"] != A["id"] and g["label"] in ("buy_low", "sell_high"):
                (buy if g["label"] == "buy_low" else []).append(row)
    sell.sort(key=lambda r: r["extra"])
    buy.sort(key=lambda r: -r["extra"])
    return sell[:8], buy[:8]


_perc = {}


def perceived(B, p):
    """what team B thinks player p is worth (market value adjusted for B's tendencies, B's window, and anchoring on last season) + short reasons"""
    key = (B["id"], p["id"])
    if key in _perc:
        return _perc[key]
    tr = TRAITS.get(B["id"], {"youth": 0.0, "rookie": 0.0, "injury": 0.0, "seasons": 0, "tags": []})
    w = WINDOW[B["id"]]["w"]
    m = p["mkt"]
    mult, why = 1.0, []
    age = p["age"] if p["age"] is not None else 26.0
    youthness = max(-1.0, min(1.0, (26.0 - age) / 6.0))
    prod = max(-0.5, min(1.0, (p["level"] - 30.0) / 25.0))
    if tr["seasons"] >= 2:
        a = 0.05 * tr["youth"] * youthness
        if abs(a) >= 0.02:
            mult += a
            why.append(("likes young players" if tr["youth"] > 0 else "prefers veterans") if a > 0 else ("prefers veterans" if tr["youth"] > 0 else "likes young players"))
        if p["rookie"] and abs(tr["rookie"]) > 0.5:
            mult += 0.04 * tr["rookie"]
            why.append("loves rookies" if tr["rookie"] > 0 else "avoids rookies")
        if p["status"] in ("OUT", "INJURY_RESERVE") and abs(tr["injury"]) > 0.5:
            mult += 0.04 * tr["injury"]
            why.append("takes injury gambles" if tr["injury"] > 0 else "avoids injured players")
    win = 0.12 * w * prod - 0.10 * w * youthness            # contenders pay for production now; rebuilders pay for youth
    if abs(win) >= 0.02:
        mult += win
        why.append("contending: wants production" if (w > 0 and prod > 0) else ("contending: less use for youth" if w > 0 else ("rebuilding: wants youth" if youthness > 0 else "rebuilding: discounts older production")))
    if p["status"] in ("OUT", "INJURY_RESERVE"):
        pen = 0.25 * (w + 1) / 2                              # a contender heavily discounts a player who is out; a rebuilder barely does
        mult -= pen
        if pen >= 0.05:
            why.append("out injured (hurts a contender)")
    anchor = max(-0.12, min(0.12, 0.4 * p["surprise"] / max(p["level"], 20.0)))     # owners anchor on last season's numbers
    if abs(anchor) >= 0.03:
        mult += anchor
        why.append("last season's spike inflates his price" if anchor > 0 else "last season's slump depresses his price")
    mult = max(0.6, min(1.4, mult))
    _perc[key] = (m * mult, mult, why)
    return _perc[key]



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
            pg = [perceived(B, p) for p in give]         # how B values what it would receive
            pr = [perceived(B, p) for p in get]          # ... and what it would give up
            give_mkt, get_mkt = sum(x[0] for x in pg), sum(x[0] for x in pr)
            # B receives `give`, gives `get`.  ratio = B's perceived value received / given, with the elite-player premium
            ratio = package_ratio([{"mkt": x[0]} for x in pg], [{"mkt": x[0]} for x in pr])
            ratio_lin = (give_mkt / get_mkt) if get_mkt > 0 else (9.9 if give_mkt > 0 else 1.0)   # plain sum, for comparison
            reasons = []
            for p, x in zip(give, pg):
                if x[1] >= 1.03 and x[2]:
                    reasons.append(f"{B['abbrev']} values {p['name']} about {round((x[1] - 1) * 100)}% more ({', '.join(x[2][:2])})")
            for p, x in zip(get, pr):
                if x[1] <= 0.97 and x[2]:
                    reasons.append(f"{B['abbrev']} values {p['name']} about {round((1 - x[1]) * 100)}% less ({', '.join(x[2][:2])})")
            ideas.append({"p": B["id"], "give": [p["name"] for p in give], "get": [p["name"] for p in get], "me": [round(dsa), round(dka, 1)], "them": [round(dsb), round(dkb, 1)],
                          "mk": [round(give_mkt), round(get_mkt)], "mkg": [round(x[0]) for x in pg], "mkr": [round(x[0]) for x in pr], "why": reasons[:3], "ratio": round(min(ratio, 9.9), 2), "ratio_lin": round(min(ratio_lin, 9.9), 2), "kbefore": keeper_names(A["players"]), "kafter": keeper_names(a_after)})
    # keep the ideas that are good for me at several keeper weights, chosen SEPARATELY for fairness bands so unfair-but-great-for-me trades
    # cannot crowd out the fair ones (the page lets the user move the fairness and premium sliders)
    keep = {}
    for lo, top in ((0.85, 40), (0.65, 20), (0.5, 10)):
        for w in WEIGHTS:
            scored = sorted(((i["me"][0] + w * ASSET_TO_PTS * i["me"][1], n) for n, i in enumerate(ideas)
                             if lo <= i["ratio"] <= 4.0 and i["me"][0] + w * ASSET_TO_PTS * i["me"][1] > 100), reverse=True)
            for _, n in scored[:top]:
                keep[n] = ideas[n]
    ks, kk = base[A["id"]]
    sell, buy = buy_sell(A)
    out_teams.append({"id": A["id"], "abbrev": A["abbrev"], "name": A["name"], "keepers": keeper_names(A["players"]), "keeper_asset": round(kk, 1),
                      "window": WINDOW[A["id"]]["label"], "sell_high": sell, "buy_low": buy, "ideas": list(keep.values())})
    print(f"{A['abbrev']:5s} {len(ideas):6d} candidate trades -> {len(keep)} kept", flush=True)

meta = {"generated": datetime.now(timezone.utc).isoformat(), "K": K, "weeks_left": WEEKS_LEFT, "asset_to_points": ASSET_TO_PTS, "default_keeper_weight": 0.10, "default_premium": S_DEFAULT, "v_max": V_MAX,
        "teams": {t["id"]: {"abbrev": t["abbrev"], "name": t["name"], "window": WINDOW[t["id"]]["label"], "tags": (TRAITS.get(t["id"]) or {}).get("tags", []), "seasons": (TRAITS.get(t["id"]) or {}).get("seasons", 0)} for t in out_teams},
        "signal_study": SIGS.get("study", {})}
(HUB / "trade_ideas.json").write_text(json.dumps({"meta": meta, "teams": out_teams}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote trade_ideas.json", round((HUB / "trade_ideas.json").stat().st_size / 1024), "KB")
