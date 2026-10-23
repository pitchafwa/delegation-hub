"""Context the trade analyzer needs, from the playoff simulation (dashboard/trade_context.json): for every team, the baseline odds and how much its title / playoff odds move per
extra point per week (slope, title percentage points per +1 point/week, estimated with a +25/week shift and the same random numbers), the league-average slope (used to turn next
year's keeper strength into title equity) and the posture the odds suggest (contending / balanced / rebuilding).
  uv run python research/build_trade_context.py      (from ingest/; needs the ESPN cookies for standings; run after the plan refresh)"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trade_value as TV

W = TV.World()
inp = TV.sim_inputs(W)
ids = inp[0]
mu = {(t, w): v for t in W.teams for w, v in W.base_proj[t].items()}
base = TV.run_odds(W, inp, mu, 20000)
slope, slope_made, out = {}, {}, {}
for t in ids:
    up = {(a, w): (v + 25.0 if a == t else v) for (a, w), v in mu.items()}
    r = TV.run_odds(W, inp, up, 20000)
    i = ids.index(t)
    slope[str(t)] = round(100 * (float(r["title"][i]) - float(base["title"][i])) / 25.0, 4)
    slope_made[str(t)] = round(100 * (float(r["made"][i]) - float(base["made"][i])) / 25.0, 4)
    made, title = float(base["made"][i]), float(base["title"][i])
    out[str(t)] = {"abbrev": next(x["abbrev"] for x in W.wp["teams"] if x["id"] == t), "made": round(made, 3), "title": round(title, 3), "posture": TV.posture_auto(made, title)}
mean = sum(slope.values()) / len(slope)
res = {"generated": datetime.now(timezone.utc).isoformat(), "slope": slope, "slope_made": slope_made, "slope_mean": round(mean, 4), "teams": out,
       "note": "slope = title percentage points per +1 point/week of forecast strength (from +25/week); posture from made/title odds (contend: made>=60% and title>=10%; rebuild: made<=25% and title<=4%)"}
DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"
(DASH / "trade_context.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
# per-player numbers for the page's INSTANT estimate (no workflow needed): how owners see him (perceived rank / value), his future-only keeper asset, level, age
ranks, hvals = TV.perceived_ranks(W)
LP = TV.LeaguePerception(W)                      # how THIS league values players (the acceptance check the analyzer and the finder use)
T5 = {tid: LP.top5(tid) for tid in W.teams}
market = {}
for tid, t in W.teams.items():
    for p in W.roster(tid):
        market[str(p["id"])] = {"n": p["name"], "t": t["abbrev"], "pr": round(ranks.get(p["id"], 400), 1), "v": round(TV.value_of_rank(hvals, ranks.get(p["id"], 400))), "af": round(p["asset_f"], 1), "lvl": p["level"], "age": p["age"], "out": p["out"],
                                "pv": round(LP.pv(p["id"]), 1), "t5": p["id"] in T5.get(tid, set())}
(DASH / "trade_market.json").write_text(json.dumps({"generated": res["generated"], "players": market}, separators=(",", ":")), encoding="utf-8")
# next draft's picks (original owner, round) -> current owner, expected slot, value in points-per-game-equivalent and in market value
l2v, own, picks = TV.level_to_market(), TV.pick_owner_map(W), {}
for t in ids:
    ab = out[str(t)]["abbrev"]
    for rnd in range(1, 6):
        probs = TV.pick_slot_probs(base["draft_pos"][ids.index(t)], rnd)
        e, m = TV.pick_value(probs, l2v)
        picks[f"{ab}:{rnd}"] = {"orig": ab, "round": rnd, "owner": own.get((t, rnd), t), "slot": round(sum(s * q for s, q in probs.items()), 1), "ppg_eq": round(e, 1), "mkt": round(m), "pv": round(TV.pick_pv(probs), 1)}
res["picks"] = picks
(DASH / "trade_context.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
print("wrote trade_context.json and trade_market.json (%d players)" % len(market))
print("league-average slope %.3f title pp per +1 pt/week; per team:" % mean, {out[k]["abbrev"]: (slope[k], out[k]["posture"]) for k in slope})
