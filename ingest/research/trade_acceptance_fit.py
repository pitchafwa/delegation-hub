"""How do owners in THIS league value players in trades? (2023-24 to 2025-26)

ESPN only shows trade proposals that involve your own team, so the league-wide evidence is the trades that actually HAPPENED: both owners agreed, so under the right
definition of 'perceived value' a completed trade should look roughly even. Completed trades are rebuilt from the daily rosters (a player moving between two league
teams on the same day, with no add or drop). Your own declined proposals are the second test: under the right definition they should look worse for the other owner
than the completed trades did.

Candidate values per player (only what was knowable that day):
  prod  : production above replacement (this season's points per game once he has 5+ games, else last season's), raised to a power (stars worth more than linearly)
  draft : what this league paid for him that season (his draft or keeper slot), convex in the pick number; undrafted = 0
  blend : weighted mix;  + age: younger players (25 and under) scaled up (the league's rookie and youth premium)
Score for a definition: how even completed trades look (median |log value ratio|), and how many of your declined offers it calls clearly short for the other owner.
Writes data/trade_acceptance_fit.json (chosen definition, fair band) for the trade finder and analyzer.   Run: uv run python research/trade_acceptance_fit.py   (from ingest/)
"""
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
YEARS = [2024, 2025, 2026]
MY_TEAM = 12
REPL = 25.0


def key(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


dh = pd.read_csv(D / "draft_history_2021_2026.csv")
base = pd.read_csv(D / "player_season_base.csv")
base["fp"] = (base.PTS + 1.5 * base.REB + 2 * base.AST + 3 * base.STL + 3 * base.BLK + base.FG3M + 2 * base.FTM - base.FTA - base.TOV + 3 * base.TD3) / base.GP.replace(0, np.nan)
base["y"] = base.SEASON.str[:4].astype(int) + 1
base["k"] = base.PLAYER_NAME.map(key)
prior = {(r.k, r.y): r.fp for r in base.itertuples() if r.GP >= 15}
ages = {(r.k, r.y): r.AGE for r in base.itertuples()}


def season(year):
    tx = json.load(open(D / f"league_transactions_{year}.json"))
    days = json.load(open(D / f"league_days_{year}.json"))
    name, pts, team_on = {}, defaultdict(list), defaultdict(dict)
    for d, recs in days.items():
        for r in recs:
            for e in r["e"]:
                name[e[0]] = e[1]
                team_on[e[0]][int(d)] = r["team"]
                if e[3] and e[4] is not None:
                    pts[e[0]].append((int(d), e[4]))
    moved = set()                                          # (pid, period) adds/drops: not trades
    for t in tx:
        if t.get("type") in ("FREEAGENT", "WAIVER") and t.get("status") == "EXECUTED":
            for it in t.get("items", []):
                moved.add((it.get("playerId"), t.get("scoringPeriodId")))
    dslot = {key(r.name): (r.n, bool(r.keeper), r.age) for r in dh[dh.year == year].itertuples()}

    def feat(pid, sp):
        nm = name.get(pid, str(pid))
        g = [p for d, p in pts.get(pid, []) if d < sp]
        fp = float(np.mean(g)) if len(g) >= 5 else prior.get((key(nm), year - 1), 18.0)
        n, kept, age = dslot.get(key(nm), (None, False, None))
        if age is None or (isinstance(age, float) and math.isnan(age)):
            age = ages.get((key(nm), year - 1), 27)
        return {"pid": pid, "name": nm, "fp": fp, "pick": n, "kept": kept, "age": age}

    # completed trades from the daily rosters
    trades = defaultdict(lambda: defaultdict(list))
    for pid, byd in team_on.items():
        ds = sorted(byd)
        for a, b in zip(ds, ds[1:]):
            if byd[a] != byd[b] and b == a + 1 and (pid, b) not in moved and (pid, a) not in moved:
                trades[(b, frozenset((byd[a], byd[b])))][byd[b]].append(pid)
    done = []
    for (d, pair), side in trades.items():
        if len(side) == 2:
            t1, t2 = list(side)
            done.append({"year": year, "sp": d, "a": [feat(p, d) for p in side[t1]], "b": [feat(p, d) for p in side[t2]]})
    # your declined proposals (only ones visible to you)
    ids = {t["id"]: t for t in tx}
    mine = []
    for t in tx:
        if t.get("type") == "TRADE_DECLINE" and t.get("relatedTransactionId") in ids:
            pr = ids[t["relatedTransactionId"]]
            if pr.get("teamId") != MY_TEAM:
                continue
            responder = t.get("teamId")
            sp = pr.get("scoringPeriodId") or 1
            give = [it["playerId"] for it in pr.get("items", []) if it.get("type") == "TRADE" and it.get("fromTeamId") == responder]
            get = [it["playerId"] for it in pr.get("items", []) if it.get("type") == "TRADE" and it.get("toTeamId") == responder]
            if give and get:
                mine.append({"year": year, "give": [feat(p, sp) for p in give], "get": [feat(p, sp) for p in get]})
    return done, mine


def v(f, spec):
    p, q, w, young = spec
    prod = max(0.0, (f["fp"] or 0) - REPL) ** p
    dr = (max(0.0, 150.0 - f["pick"]) / 150.0) ** q * (60.0 - REPL) ** p if f["pick"] is not None else 0.0
    val = w * prod + (1 - w) * dr
    if f["age"] is not None and f["age"] <= 25:
        val *= 1 + young
    return val


def lr(xs, ys, spec):
    return math.log((sum(v(f, spec) for f in xs) + 1) / (sum(v(f, spec) for f in ys) + 1))


done, mine = [], []
for y in YEARS:
    d_, m_ = season(y)
    done += d_
    mine += m_
print(f"completed trades rebuilt: {len(done)} ({', '.join(str(sum(1 for t in done if t['year'] == y)) + ' in ' + str(y) for y in YEARS)}); your declined proposals with both sides: {len(mine)}")
for t in done[:12]:
    print("   ", t["year"], [f["name"] for f in t["a"]], "<->", [f["name"] for f in t["b"]])
res = []
for p in (1.0, 1.5, 2.0):
    for q in (1.0, 2.0, 3.0):
        for w in (1.0, 0.7, 0.5, 0.3, 0.0):
            for young in (0.0, 0.25, 0.5):
                spec = (p, q, w, young)
                even = [abs(lr(t["a"], t["b"], spec)) for t in done]
                decl = [lr(m["get"], m["give"], spec) for m in mine]                   # for the OTHER owner: in / out
                band = float(np.percentile(even, 80)) if even else 1.0
                caught = float(np.mean([x < -band * 0.5 for x in decl])) if decl else 0.0  # share of declined offers it calls short for them
                res.append((float(np.median(even)) - 0.15 * caught, float(np.median(even)), caught, band, spec))
res.sort()
print("best definitions (score, median |log ratio| of completed trades, share of your declined offers called short, 80th pct band, (prod power, draft power, weight on production, youth premium)):")
for r in res[:8]:
    print("  ", [round(x, 3) if isinstance(x, float) else x for x in r])
best = res[0]
spec = best[4]
print("chosen:", spec, "-> completed trades sit within a value ratio of", round(math.exp(best[3]), 2), "(80%)")
for m in mine:
    print("   declined:", [f["name"] for f in m["give"]], "for", [f["name"] for f in m["get"]], "ratio for them", round(math.exp(lr(m["get"], m["give"], spec)), 2))
(D / "trade_acceptance_fit.json").write_text(json.dumps({"spec": {"prod_power": spec[0], "draft_power": spec[1], "w_prod": spec[2], "youth": spec[3], "repl": REPL},
                                                         "fair_band_log": best[3], "median_abs_log": best[1], "n_completed": len(done), "n_declined_mine": len(mine)}, indent=1), encoding="utf-8")
