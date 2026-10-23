"""Trade finder v2 (2026-10-23): ideas that are good for you by the SAME engine as the Trades-tab analyzer, and that the other owner is likely to accept by the league's own
perception of value (trade_value.LeaguePerception: production + what this league paid in its draft/keepers, youth premium, star premium, keeper attachment).

For every other team:
  1. enumerate 1-for-1, 2-for-1 and 1-for-2 swaps among each side's top players (by perceived value);
  2. keep only those the other owner would see as at least 'possible' (value received / given >= 0.95, his likely keepers count extra);
  3. screen what each does for you with a quick lineup-free estimate (rest-of-season points above your 10th man + next year's keeper strength);
  4. run the full analyzer (lineups for every week, playoff simulation for both teams, forced drops, free-agent fill) on the best few per team.
Writes dashboard/trade_ideas_v2.json, ranked by your value. Runs weekly in the PC refresh (the full analyzer needs a few minutes per team).
  uv run python research/find_trades.py [--per-team 3] [--n 3000]       (from ingest/)
"""
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import trade_value as TV

ARGS = sys.argv[1:]
PER_TEAM = int(ARGS[ARGS.index("--per-team") + 1]) if "--per-team" in ARGS else 3
NSIM = int(ARGS[ARGS.index("--n") + 1]) if "--n" in ARGS else 3000
POOL = 9                       # players per side considered (by perceived value)


def main():
    t0 = time.time()
    W = TV.World()
    inp = TV.sim_inputs(W)
    LP = TV.LeaguePerception(W)
    me_ab = W.wp["my_abbrev"]
    me_id = W.by_abbrev[me_ab]
    mine = [p for p in W.roster(me_id) if not p["ir"]]
    games_left = {p["id"]: W.cum_games(p["team"], W.last + 1) for p in W.roster(me_id)}
    tenth = sorted((p["level"] for p in mine), reverse=True)[9] if len(mine) >= 10 else 0.0
    k_me = TV.keeper_strength(W.roster(me_id))
    ctx = W.ctx or {}
    slope = ctx.get("slope", {}).get(str(me_id), TV.DEFAULT_SLOPE)
    s_next = ctx.get("slope_mean", TV.DEFAULT_SLOPE)

    def quick_me(give, get):
        """rest-of-season points you gain (only above your 10th man counts fully; extra bodies push out your worst) plus next year's keeper strength, in title points (rough)"""
        def contrib(p, g):
            return max(0.0, p["level"] - tenth) * g + 0.25 * min(p["level"], tenth) * g
        pts = sum(contrib(p, W.cum_games(p["team"], W.last + 1)) for p in get) - sum(contrib(p, games_left.get(p["id"], 0)) for p in give)
        extra = len(get) - len(give)
        if extra > 0:                                       # you must drop your worst non-IR players
            worst = sorted(mine, key=lambda p: p["level"])[:extra]
            pts -= sum(contrib(p, games_left.get(p["id"], 0)) for p in worst)
        new = [p for p in W.roster(me_id) if p["id"] not in {g["id"] for g in give}] + list(get)
        dk = TV.keeper_strength(new) - k_me
        weeks = max(1, W.last - W.first + 1)
        return slope * pts / weeks + TV.FUTURE_LAMBDA * TV.GAMES_PER_WEEK_PER_ASSET_PT * s_next * dk

    my_pool = sorted(mine, key=lambda p: -LP.pv(p["id"]))[:POOL + 3]
    my_keepers = {p["name"] for p in sorted(W.roster(me_id), key=lambda p: -p["asset"])[:5]}     # your likely keepers (top 5 by keeper asset value)
    ideas = []
    for ab, tid in W.by_abbrev.items():
        if tid == me_id:
            continue
        theirs = sorted([p for p in W.roster(tid) if not p["ir"]], key=lambda p: -LP.pv(p["id"]))[:POOL]
        cands = []
        shapes = [(1, 1), (2, 1), (1, 2)]
        for ng, nr in shapes:
            for G in itertools.combinations(my_pool, ng):
                for R in itertools.combinations(theirs, nr):
                    acc = LP.accept([p["id"] for p in G], [p["id"] for p in R], tid)
                    if acc["band"] == "unlikely":
                        continue
                    q = quick_me(G, R)
                    if q > 0.3:
                        cands.append((q, G, R, acc))
        cands.sort(key=lambda c: -c[0])
        seen, picked = set(), []
        for q, G, R, acc in cands:                          # do not run near-duplicates (same player you get, same player you give)
            sig = (tuple(sorted(p["id"] for p in R))[:1], tuple(sorted(p["id"] for p in G))[:1])
            if sig in seen:
                continue
            seen.add(sig)
            picked.append((q, G, R, acc))
            if len(picked) >= PER_TEAM:
                break
        for q, G, R, acc in picked:
            try:
                res = TV.evaluate(me_ab, ab, [p["name"] for p in G], [p["name"] for p in R], n=NSIM, world=W, inp=inp)
            except SystemExit as ex:
                print("skip", ab, ex)
                continue
            ideas.append({"partner": ab, "give": res["give"], "get": res["get"], "screen": round(q, 2), "me": res["me"], "them": res["them"], "market": res["market"], "verdict": res["verdict"],
                          "my_keepers_given": [n for n in res["give"] if n in my_keepers]})
            print(f"{ab}: give {res['give']} get {res['get']} -> you {res['me']['value']['total']:+.2f}, them {res['them']['value']['total']:+.2f}, {res['market']['band']} ({res['market']['ratio']})  [{time.time() - t0:.0f}s]", flush=True)
    good = [i for i in ideas if i["me"]["value"]["total"] > 0.3 and i["market"]["band"] != "unlikely"]
    good.sort(key=lambda i: -i["me"]["value"]["total"])
    uses, varied = {}, []                                   # no single player of yours in more than 3 ideas, so the list shows different routes
    for i in good:
        if any(uses.get(n, 0) >= 3 for n in i["give"]):
            continue
        for n in i["give"]:
            uses[n] = uses.get(n, 0) + 1
        varied.append(i)
    good = varied
    out = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "me": me_ab, "n_checked": len(ideas), "sims": NSIM,
           "note": "Every idea was run through the full analyzer (both teams' lineups and playoff odds) and passes the league-perception acceptance check (possible or likely).",
           "ideas": good}
    (TV.HUB / "trade_ideas_v2.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote trade_ideas_v2.json: {len(good)} ideas of {len(ideas)} analyzed in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
