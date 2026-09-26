"""Playoff odds: standings from ESPN + forecast strength of every team for every remaining week (schedule_plan.json) -> playoff_sim.py -> dashboard/playoff_odds.json.

  uv run python research/build_playoff_odds.py                       live (2026-27 season)
  uv run python research/build_playoff_odds.py --backtest            check the model on last season's real checkpoints (weeks 3, 6, 9, 12, 15) and print calibration
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from espn_api.basketball import League
from playoff_sim import simulate

sys.stdout.reconfigure(encoding="utf-8")
DASH = Path(__file__).resolve().parent.parent.parent / "dashboard"
REG_LAST = 19
PARAMS = json.load(open(Path(__file__).resolve().parent / "playoff_odds_params.json")) if (Path(__file__).resolve().parent / "playoff_odds_params.json").exists() else {"tau": 100.0, "sigma_e": 175.0, "bias_preseason": 25.0, "bias_blend": 0.0}


def pull(year):
    lg = League(league_id=config.LEAGUE_ID, year=year, espn_s2=config.ESPN_S2, swid=config.SWID)
    raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mTeam"]})
    teams = {t["id"]: t.get("abbrev") or str(t["id"]) for t in raw["teams"]}
    games = []
    for m in raw["schedule"]:
        if m["matchupPeriodId"] > REG_LAST or "away" not in m:
            continue
        a, h = m["away"], m["home"]
        done = m.get("winner") in ("HOME", "AWAY", "TIE")
        n = len(h.get("pointsByScoringPeriod") or {}) or None
        games.append(dict(week=m["matchupPeriodId"], a=a["teamId"], b=h["teamId"], done=done, sa=float(a.get("totalPoints") or 0), sb=float(h.get("totalPoints") or 0), n=n))
    return teams, games


def standings(games, upto=None):
    wins, pf = {}, {}
    for g in games:
        if not g["done"] or (upto is not None and g["week"] > upto):
            continue
        for t in (g["a"], g["b"]):
            wins.setdefault(t, 0)
            pf.setdefault(t, 0.0)
        pf[g["a"]] += g["sa"]
        pf[g["b"]] += g["sb"]
        if g["sa"] > g["sb"]:
            wins[g["a"]] += 1
        elif g["sb"] > g["sa"]:
            wins[g["b"]] += 1
    return wins, pf


def live():
    teams, games = pull(2027)
    sp = json.load(open(DASH / "schedule_plan.json", encoding="utf-8"))
    ids = sorted(teams)
    wins, pf = standings(games)
    for t in ids:
        wins.setdefault(t, 0)
        pf.setdefault(t, 0.0)
    lens = {w["id"]: w["days"] for w in sp["calendar"]}
    mu = {}
    for t in sp["teams"]:
        for w, v in t["proj"].items():
            mu[(t["id"], int(w))] = float(v)
    pairs = [(g["week"], g["a"], g["b"]) for g in games if not g["done"]]
    cur = min((w for w, _, _ in pairs), default=REG_LAST + 1)
    my = next(t["id"] for t in sp["teams"] if t["abbrev"] == sp["my_abbrev"])
    basis = sp["teams"][0].get("proj_basis")
    PR = {"tau": PARAMS["tau"], "sigma_e": PARAMS["sigma_e"], "bias": PARAMS["bias_blend"] if basis == "blend" else PARAMS["bias_preseason"]}
    res = simulate(ids, wins, pf, pairs, mu, lens, REG_LAST, focus=my, focus_week=cur, **PR)
    # what-if: +25 points a week for my team from now on
    up = simulate(ids, wins, pf, pairs, mu, lens, REG_LAST, shift={my: 25.0}, **PR)
    mi = ids.index(my)
    out = {"generated": datetime.now(timezone.utc).isoformat(), "current_week": cur, "weeks_left": len({w for w, _, _ in pairs}), "basis": basis, "params": PR,
           "my_id": my, "my_abbrev": sp["my_abbrev"],
           "teams": [{"id": t, "abbrev": teams[t], "wins": wins[t], "pf": round(pf[t]), "exp_wins": round(float(res["wins"][i]), 1), "made": round(float(res["made"][i]), 3),
                      "bye": round(float(res["bye"][i]), 3), "final": round(float(res["final"][i]), 3), "title": round(float(res["title"][i]), 3),
                      "seed": [round(float(x), 3) for x in res["seed"][i][:6]]} for i, t in enumerate(ids)],
           "me": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k.startswith("focus_") and k != "focus_wins_dist"},
           "me_wins_dist": {str(k): {"p": round(v["p"], 3), "made": round(v["made"], 3)} for k, v in (res["focus_wins_dist"] or {}).items()},
           "me_plus25": {"made": round(float(up["made"][mi]), 3), "title": round(float(up["title"][mi]), 3)}}
    out["teams"].sort(key=lambda x: -x["made"])
    (DASH / "playoff_odds.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("wrote playoff_odds.json; me:", out["me"], "+25/wk:", out["me_plus25"])
    for t in out["teams"]:
        print(f"{t['abbrev']:5s} {t['wins']:2d}w  exp {t['exp_wins']:5.1f}  playoffs {t['made']:.0%}  bye {t['bye']:.0%}  title {t['title']:.1%}")


def backtest():
    teams, games = pull(2026)
    ids = sorted(teams)
    real_w, real_pf = standings(games)
    order = sorted(ids, key=lambda t: (-real_w[t], -real_pf[t]))
    made_real = {t: order.index(t) < 6 for t in ids}
    weeks = {g["week"] for g in games}
    lens = {w: next((g["n"] for g in games if g["week"] == w and g["n"]), 7) for w in weeks}
    # week-level score history per team
    hist = {t: {} for t in ids}
    for g in games:
        hist[g["a"]][g["week"]] = g["sa"]
        hist[g["b"]][g["week"]] = g["sb"]
    grid = [(150.0, 175.0), (100.0, 175.0), (200.0, 175.0), (150.0, 220.0), (100.0, 220.0), (60.0, 220.0)]
    for tau, se in grid:
        rows = []
        for c in (3, 6, 9, 12, 15):
            wins, pf = standings(games, upto=c)
            for t in ids:
                wins.setdefault(t, 0)
                pf.setdefault(t, 0.0)
            pairs = [(g["week"], g["a"], g["b"]) for g in games if g["week"] > c]
            # forecast: shrink each team's mean per-7-days score halfway to the league mean (a crude stand-in for the plan forecast)
            per7 = {t: np.mean([hist[t][w] / lens[w] * 7 for w in range(1, c + 1)]) for t in ids}
            lg = np.mean(list(per7.values()))
            mu = {(t, w): (0.5 * per7[t] + 0.5 * lg) * lens[w] / 7 for t in ids for w in weeks}
            r = simulate(ids, wins, pf, pairs, mu, lens, REG_LAST, tau=tau, sigma_e=se, bias=0.0, n=8000)
            for i, t in enumerate(ids):
                rows.append((c, float(r["made"][i]), made_real[t]))
        p = np.array([x[1] for x in rows])
        y = np.array([x[2] for x in rows], float)
        brier = np.mean((p - y) ** 2)
        bins = [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]
        cal = " ".join(f"[{lo:.1f}-{min(hi,1):.1f}: pred {p[(p>=lo)&(p<hi)].mean():.2f} act {y[(p>=lo)&(p<hi)].mean():.2f} n{((p>=lo)&(p<hi)).sum()}]" for lo, hi in bins if ((p >= lo) & (p < hi)).any())
        print(f"tau {tau:5.0f} sigma_e {se:4.0f}  Brier {brier:.4f}  {cal}")


if __name__ == "__main__":
    backtest() if "--backtest" in sys.argv else live()
