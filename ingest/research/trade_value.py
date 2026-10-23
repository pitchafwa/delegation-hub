"""Trade value engine (2026-10-11; design in RESEARCH_trade_value.md).  A trade is judged by what it does to each team's OWN chances, not by a sum of prices:

    V(team) = w_now x (change in TITLE probability, in percentage points)  +  w_future x lambda x c_asset x (change in next year's keeper strength)

  season part   rebuild each team's best lineups for every remaining week with the roster after the trade (same mechanics as the schedule plan: 10 slots with eligibility, the
                40-start games cap, daily NBA schedule, back-to-backs, availability), then run the playoff simulation (playoff_sim.simulate, same random numbers for the baseline
                and the trade so differences are not noise) with the two teams' weekly forecasts changed.  Playoff weeks matter most because the simulation knows they decide the title.
  injuries      a player who is out now contributes nothing until his expected return and then ramps up (injury advisor: games out, chance back for the playoffs, return ramp), so
                a Butler-type player COSTS this season's weeks and PAYS in the playoff weeks and next year.  IR slots (4) are free; beyond that he costs a roster spot.
  roster spots  15 non-IR spots.  A team that ends up over 15 drops whoever hurts least; a team under 15 fills the slot with the best free agent (adds are instant here).
  keepers       next year each team keeps its top 5 assets (3 this year, already chosen).  Asset values are uncertain, so the 'top 5' is soft: E[sum of the top 5] under noise
                plus a small chip value for ranks 6-9 (assets that can still be traded).  Players outside any team's top ~7 contribute nothing here.
  posture       rebuilding / balanced / contending changes the weights on now vs future (auto picks from the team's playoff and title odds).
  acceptance    a separate model of what the OTHER owner will think, from perceived market value (not from the true model): see perceived_value() and acceptance().

Run:  uv run python research/trade_value.py --me DRNK --partner YUNG --give "Player A; Player B" --get "Player C" [--posture-me auto|rebuild|balanced|contend] [--posture-them ...]
"""
import argparse
import itertools
import json
import math
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import season_calendar as SCAL
from playoff_sim import simulate
from season import current_season_id
from team_abbr import canon

ROOT = Path(__file__).resolve().parent
HUB = ROOT.parent.parent / "dashboard"
SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UT", "UT", "UT"]
ROSTER_SPOTS, IR_SLOTS = 15, 4
OUT_STATUS = ("OUT", "INJURY_RESERVE", "SUSPENSION")
PLAYOFF_WT = 1.5                 # weight of a playoff week in the greedy drop/fill decisions (the final verdict uses the simulation)
RETURN_FACTOR, NEAR_WEEKS = 0.6, 2
RAMP_MULT = 0.87                 # average share of normal output after a return (injury advisor ramp: 71% game 1, 85% games 2-3, 88% later)
GAMES_PER_WEEK_PER_ASSET_PT = 3.4  # one asset point = +1 point per game for a season = about 3.4 points a week
FUTURE_LAMBDA = 0.15             # share of the discounted 10-year "points above the keeper cutoff" that counts as real title equity (uncertainty beyond a year or two, injuries, roster churn,
                                 # keeper slots being scarce, title odds saturating). JUDGMENT parameter, shown in every output and adjustable on the page; 0.15 puts a young superstar's future at a
                                 # few times his single-season title swing. (The old trade finder used 0.10 on the same units.)
KEEP_PTS = 3.4 * 19 * FUTURE_LAMBDA   # season points one future-asset point is worth in the greedy drop decisions (units of week_points)
CHIP_WEIGHT = 0.15               # value of assets ranked 6-9 on a roster (tradeable even if not kept)
ASSET_NOISE = 0.25               # relative uncertainty of an asset value when asking 'who will be in the top 5'
POSTURE = {"rebuild": (0.7, 1.4), "balanced": (1.0, 1.0), "contend": (1.25, 0.7)}      # (weight on this season, weight on the future)
DEFAULT_SLOPE = 0.17             # title percentage points per +1 point/week, used only when the context file is missing


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", n, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", n.lower())).strip()


def p_future(level):
    return 0.61 if level < 15 else (0.81 if level < 20 else 0.85)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi_inv(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# =============================================================== world
class World:
    def __init__(self):
        L = lambda n: json.load(open(HUB / n, encoding="utf-8"))
        self.wp, self.sp, self.hub = L("week_plan.json"), L("schedule_plan.json"), L("hub_data.json")
        self.hub_by_id = {p["id"]: p for p in self.hub["players"]}
        self.cal = SCAL.load()
        self.weeks = {p["id"]: p for p in self.cal["periods"]}
        self.first = self.sp["current_week"]
        self.last = self.cal["last_week"]
        self.reg_last = self.cal["regular_weeks"]
        self.playoff_weeks = tuple(self.cal["playoff_weeks"])
        self.heat = self.sp["nba"]
        sched = L("nba_schedule.json")["games"]
        self.games = {}
        for ds, gl in sched.items():
            self.games[date.fromisoformat(ds)] = {canon(t) for g in gl for t in g[:2] if t != "TBD"}
        self.teams = {t["id"]: t for t in self.wp["teams"]}
        self.by_abbrev = {t["abbrev"]: t["id"] for t in self.wp["teams"]}
        self.abbrev_of = {v: k for k, v in self.by_abbrev.items()}
        self.base_proj = {t["id"]: {int(w): float(v) for w, v in t["proj"].items()} for t in self.sp["teams"]}
        self.proj_basis = self.sp["teams"][0].get("proj_basis")
        self.fa_pool = [self.mk(dict(f, ir=False), None) for f in self.wp.get("fa_pool", []) if not f.get("waiver")]
        ctx = HUB / "trade_context.json"
        self.ctx = json.load(open(ctx, encoding="utf-8")) if ctx.exists() else None
        self._wk_cache = {}

    # ---- calendar helpers
    def plays(self, team, d):
        return team in self.games.get(d, ())

    def b2b(self, team, d):
        return self.plays(team, d) and self.plays(team, d - timedelta(days=1))

    def wk_days(self, w):
        p = self.weeks[w]
        s = date.fromisoformat(p["start"])
        return [s + timedelta(days=i) for i in range(p["days"])]

    def cum_games(self, team, w):
        """NBA games the team plays from the current week up to (not including) week w"""
        row = self.heat.get(team) or []
        return sum(row[i - 1]["g"] for i in range(self.first, w) if i - 1 < len(row))

    # ---- players
    def mk(self, p, advice):
        hid = p.get("hub_id")
        hp = self.hub_by_id.get(hid) if hid else None
        asset = (hp["asset_k"][5] if hp and hp.get("asset_k") and len(hp["asset_k"]) > 5 else p.get("asset")) or 0.0
        # asset values sum the discounted points-above-the-keeper-cutoff over 10 seasons, and season 1 is THIS season, which the odds part already measures. Keep only the future
        # (seasons 2-10) so it is not counted twice: subtract this season's term, max(0, year-0 projection - the 5-keeper cutoff).
        oc = self.hub.get("opportunity_cost") or {"floor": 14.84, "amp": 20.37, "tau": 19.96}
        cut5 = oc["floor"] + oc["amp"] * math.exp(-5.0 / oc["tau"])
        asset_f = max(0.0, float(asset or 0.0) - max(0.0, float((hp or {}).get("year0_ppg") or 0.0) - cut5))
        pl = {"id": p["id"], "name": p["name"], "team": canon(p.get("team") or ""), "slots": [s for s in p.get("slots", []) if s in set(SLOTS)] or ["UT"], "status": p.get("status", "ACTIVE"),
              "level": float(p.get("level") or 0.0), "ir": bool(p.get("ir")), "hub_id": hid, "asset": float(asset or 0.0), "asset_f": asset_f, "age": (hp or {}).get("age"), "advice": advice,
              "out": p.get("status") in OUT_STATUS or bool(p.get("ir"))}
        pl["avail"] = self.availability(pl)
        return pl

    def availability(self, pl):
        """{week: share of normal output} for a player who is out now (1.0 for everyone else)"""
        if not pl["out"] or not pl["team"]:
            return {}
        adv = pl["advice"] or {}
        G = adv.get("plan_games")
        weeks = range(self.first, self.last + 1)
        if not G or G <= 0:
            return {w: (0.0 if w < self.first + NEAR_WEEKS else RETURN_FACTOR) for w in weeks}          # no estimate: the schedule plan's old assumption
        n_po = max(self.cum_games(pl["team"], self.playoff_weeks[0]), 1)
        pbp = adv.get("p_back_playoffs")
        z = phi_inv(pbp) if pbp is not None else None
        sig = 0.6
        if z is not None and abs(z) > 0.05:
            s_ = (math.log(n_po) - math.log(G)) / z
            if s_ > 0:
                sig = min(max(s_, 0.15), 1.5)
        out = {}
        for w in weeks:
            row = self.heat.get(pl["team"]) or []
            g_w = row[w - 1]["g"] if w - 1 < len(row) else 3
            cg = self.cum_games(pl["team"], w) + g_w / 2.0
            out[w] = (phi((math.log(max(cg, 1e-6)) - math.log(G)) / sig) if cg > 0 else 0.0) * RAMP_MULT
        return out

    def roster(self, team_id):
        t = self.teams[team_id]
        adv = {a["id"]: a for a in (t.get("injury_advice") or [])}
        return [self.mk(p, adv.get(p["id"])) for p in t["roster"]]

    # ---- weekly points (the schedule plan's solver, with per-week availability)
    def p_play(self, pl, d, w):
        if not pl["team"] or not self.plays(pl["team"], d):
            return 0.0
        p = p_future(pl["level"]) * pl["avail"].get(w, 1.0)
        if self.b2b(pl["team"], d) and not pl["out"]:
            p *= 0.89 / 0.94
        return p

    @staticmethod
    def slot_matcher(cands):
        match = {}

        def try_assign(ci, seen):
            for si, s in enumerate(SLOTS):
                if s in cands[ci][1]["slots"] and si not in seen:
                    seen.add(si)
                    if si not in match or try_assign(match[si], seen):
                        match[si] = ci
                        return True
            return False
        return [ci for ci in range(len(cands)) if try_assign(ci, set())]

    def week_points(self, roster, w):
        key = (w, tuple(sorted((p["id"] for p in roster))))
        if key in self._wk_cache:
            return self._wk_cache[key]
        pr = self.weeks[w]
        days, cap = self.wk_days(w), pr["cap"]
        opts = []
        for d in days:
            cands = sorted(((self.p_play(pl, d, w) * pl["level"], pl, self.p_play(pl, d, w)) for pl in roster if self.p_play(pl, d, w) > 0), key=lambda x: -x[0])
            acc = self.slot_matcher(cands)
            vals, cnt = [0.0], [0.0]
            for ci in acc:
                vals.append(vals[-1] + cands[ci][0])
                cnt.append(cnt[-1] + cands[ci][2])
            opts.append((vals, cnt))
        NS = int(cap) + 12
        memo = {}

        def best(di, c):
            if di == len(days) or c >= cap:
                return 0.0
            k = (di, c)
            if k in memo:
                return memo[k]
            vals, cnt = opts[di]
            top = 0.0
            for m in range(len(vals)):
                v = vals[m] + best(di + 1, min(int(round(c + cnt[m])), NS))
                if v > top:
                    top = v
            memo[k] = top
            return top
        out = best(0, 0)
        self._wk_cache[key] = out
        return out

    def season_value(self, roster):
        """weighted expected points over every remaining week (playoff weeks count more); used for the greedy drop / fill decisions"""
        return sum(self.week_points(roster, w) * (PLAYOFF_WT if w in self.playoff_weeks else 1.0) for w in range(self.first, self.last + 1))

    def weekly(self, roster):
        return {w: self.week_points(roster, w) for w in range(self.first, self.last + 1)}


# =============================================================== roster logic
def spots_used(roster):
    """non-IR roster spots: players who are out can sit on IR (4 slots, the longest absences first); everyone else, and any extra injured player, takes a spot"""
    inj = sorted((p for p in roster if p["out"]), key=lambda p: -((p["advice"] or {}).get("plan_games") or 99))
    on_ir = {p["id"] for p in inj[:IR_SLOTS]}
    return sum(1 for p in roster if p["id"] not in on_ir)


def keeper_strength(roster, k=5, draws=400):
    """E[sum of the top k asset values] when each asset is uncertain, plus a chip value for ranks k+1..k+4"""
    vals = [p["asset_f"] for p in roster if p["asset_f"] > 0]
    if not vals:
        return 0.0
    Z = np.stack([np.random.default_rng(p["id"]).normal(0, 1, draws) for p in roster if p["asset_f"] > 0])
    a = np.array(vals)[:, None]
    noisy = np.maximum(a * (1 + ASSET_NOISE * Z) + 0.8 * Z, 0)
    noisy.sort(axis=0)
    top = noisy[::-1]
    return float(top[:k].sum(axis=0).mean() + CHIP_WEIGHT * top[k:k + 4].sum(axis=0).mean())


def fix_roster(world, roster, label):
    """after a trade: drop whoever hurts least while over 15 spots; fill open spots with the best free agents (instant adds). Returns (roster, drops, adds)."""
    drops, adds, roster = [], [], list(roster)
    while spots_used(roster) > ROSTER_SPOTS:
        base = world.season_value(roster) + KEEP_PTS * keeper_strength(roster)
        best = None
        cur_spots = spots_used(roster)
        for p in roster:
            rest = [x for x in roster if x is not p]
            if spots_used(rest) >= cur_spots:
                continue                                  # dropping a player who sits on IR does not free a roster spot
            loss = base - (world.season_value(rest) + KEEP_PTS * keeper_strength(rest))
            if best is None or loss < best[0]:
                best = (loss, p)
        drops.append(best[1])
        roster = [x for x in roster if x is not best[1]]
    have = {p["id"] for p in roster}
    cands = sorted((f for f in world.fa_pool if f["id"] not in have), key=lambda f: -f["level"] * sum(world.heat.get(f["team"], [{"g": 3}])[i - 1]["g"] for i in range(world.first, min(world.first + 3, world.last + 1)) if i - 1 < len(world.heat.get(f["team"], []))))[:14]
    while spots_used(roster) < ROSTER_SPOTS and cands:
        base = world.season_value(roster)
        gains = [(world.season_value(roster + [f]) - base, f) for f in cands]
        g, f = max(gains, key=lambda x: x[0])
        if g <= 0:
            break
        roster = roster + [f]
        adds.append(f)
        cands = [c for c in cands if c is not f]
    return roster, drops, adds


# =============================================================== odds
def sim_inputs(world):
    """standings and the remaining schedule from ESPN (needs the league cookies), in playoff_sim's input format"""
    import config
    from espn_api.basketball import League
    lg = League(league_id=config.LEAGUE_ID, year=current_season_id(), espn_s2=config.ESPN_S2, swid=config.SWID)
    raw = lg.espn_request.league_get(params={"view": ["mMatchupScore", "mTeam"]})
    teams = {t["id"]: t.get("abbrev") or str(t["id"]) for t in raw["teams"]}
    wins, pf, pairs = {t: 0 for t in teams}, {t: 0.0 for t in teams}, []
    for m in raw["schedule"]:
        if m["matchupPeriodId"] > world.reg_last or "away" not in m:
            continue
        a, h = m["away"], m["home"]
        done = m.get("winner") in ("HOME", "AWAY", "TIE")
        if done:
            sa, sb = float(a.get("totalPoints") or 0), float(h.get("totalPoints") or 0)
            pf[a["teamId"]] += sa
            pf[h["teamId"]] += sb
            wins[a["teamId"]] += sa > sb
            wins[h["teamId"]] += sb > sa
        else:
            pairs.append((m["matchupPeriodId"], a["teamId"], h["teamId"]))
    return sorted(teams), wins, pf, pairs


def params(world):
    pp = ROOT / "playoff_odds_params.json"
    P = json.load(open(pp)) if pp.exists() else {"tau": 100.0, "sigma_e": 175.0, "bias_preseason": 25.0, "bias_blend": 0.0}
    return {"tau": P["tau"], "sigma_e": P["sigma_e"], "bias": P["bias_blend"] if world.proj_basis == "blend" else P["bias_preseason"]}


def run_odds(world, inp, mu, n=20000):
    ids, wins, pf, pairs = inp
    lens = {w["id"]: w["days"] for w in world.cal["periods"]}
    return simulate(ids, wins, pf, pairs, mu, lens, world.reg_last, playoff_weeks=world.playoff_weeks, n=n, **params(world))


def posture_auto(made, title):
    if made >= 0.6 and title >= 0.10:
        return "contend"
    if made <= 0.25 and title <= 0.04:
        return "rebuild"
    return "balanced"


# =============================================================== perceived value / acceptance
def perceived_ranks(world):
    """a blended 'how owners see him' rank for every rostered player: Hashtag dynasty rank, ESPN ADP, our rest-of-season rank, the 2026 draft pick (if drafted)"""
    import pandas as pd
    hk = pd.read_csv(ROOT / "data" / "hashtag_dynasty_latest.csv")
    hrank = {norm(r.player): r.rank for r in hk.itertuples()}
    hvals = hk.sort_values("rank")[["rank", "value"]].values
    rd = json.load(open(HUB / "redraft_data.json", encoding="utf-8"))
    rrank = {norm(p["name"]): p["rank"] for p in rd["players"]}
    pick = {}
    try:
        dd = json.load(open(HUB / "draft_data.json", encoding="utf-8"))
        names = {int(k): v["name"] for k, v in dd["players"].items()}
        for pk in dd["draft"]["picks"]:
            if pk.get("player") and not pk.get("keeper"):
                pick[norm(names.get(pk["player"], ""))] = pk["n"] - 36 if pk["n"] > 36 else pk["n"]
    except Exception:
        pass
    out = {}
    for tid in world.teams:
        for p in world.roster(tid):
            n = norm(p["name"])
            hp = world.hub_by_id.get(p["hub_id"] or "", {})
            comps = [x for x in (hrank.get(n), hp.get("adp"), rrank.get(n), pick.get(n)) if x is not None]
            out[p["id"]] = float(np.mean(comps)) if comps else 400.0
    return out, hvals


def value_of_rank(hvals, rank):
    """Hashtag's own rank -> value curve, above a replaceable player (baseline 1000): a bench player is worth about nothing"""
    r = np.clip(rank, hvals[0][0], hvals[-1][0])
    return max(0.0, float(np.interp(r, hvals[:, 0], hvals[:, 1])) - 1000.0)


def market_lookup(world):
    """player -> perceived value. Uses dashboard/trade_market.json (committed, built by build_trade_context.py from the local Hashtag file) so the cloud run needs no local data;
    falls back to computing it from the local Hashtag file."""
    mp = HUB / "trade_market.json"
    if mp.exists():
        m = json.load(open(mp, encoding="utf-8"))["players"]
        return lambda p: float(m.get(str(p["id"]), {}).get("v", 0.0))
    ranks, hvals = perceived_ranks(world)
    return lambda p: value_of_rank(hvals, ranks.get(p["id"], 400))


def acceptance(give_vals, get_vals):
    """the OTHER owner's view: they receive `give` (our give) and send `get`. Needs to come out ahead by the endowment margin (~10%)."""
    recv, sent = sum(give_vals), sum(get_vals)
    if sent <= 0:
        return {"ratio": None, "band": "likely", "recv": round(recv), "sent": round(sent)}
    r = recv / sent
    band = "likely" if r >= 1.10 else ("possible" if r >= 0.92 else "unlikely")
    return {"ratio": round(r, 2), "band": band, "recv": round(recv), "sent": round(sent)}


# =============================================================== how THIS league's owners see a trade (acceptance)
# Fitted 2026-10-23 (research/trade_acceptance_fit.py) on what the league actually did: only 9 trades were completed in 2023-26 (all league-wide evidence ESPN shows) plus Tommy's
# declined offers. Perceived value of a player = half production as everyone sees it (ESPN's projection, blended with this season's average once he has 5 games) above
# replacement, half what this league PAID for him in this season's draft or keeper rounds; 25% more for players 25 and under (the league's rookie and youth premium).
# A side is valued with a STAR PREMIUM: best player at full value, then 0.5, 0.3, 0.2 for the rest (owners do not trade stars for quantity: Tommy's 5-for-1 offer for Doncic
# looked like a 1.66x win for them on a plain sum and was declined; with the star premium it is about even). The owner's own likely keepers (his top 5 by this value) count
# 25% more when he gives them up. Bands on value received / value given by the other owner: likely >= 1.15, possible >= 1.0 (he must come out ahead by his own valuation). Thin data: these are priors with a sanity check,
# not a fitted probability.
LEAGUE_PV = {"w_prod": 0.5, "repl": 25.0, "draft_scale": 35.0, "youth": 0.25, "youth_age": 25, "star_w": [1.0, 0.5, 0.3, 0.2], "keeper_endow": 0.25, "likely": 1.15, "possible": 1.0}


class LeaguePerception:
    def __init__(self, world):
        rdp = HUB / "redraft_data.json"
        self.rd = {str(x["id"]): x for x in (json.load(open(rdp, encoding="utf-8"))["players"] if rdp.exists() else [])}
        ddp = HUB / "draft_data.json"
        self.pick = {}
        if ddp.exists():
            for x in json.load(open(ddp, encoding="utf-8"))["draft"].get("picks", []):
                if x.get("player"):
                    self.pick[int(x["player"])] = 1 if x.get("keeper") else x["n"]      # a keeper is what his owner valued most: priced like a first-round pick (keeper rounds sit at the end of the board)
        self.rows, self.team_ids = {}, {}
        for t in world.wp["teams"]:
            self.team_ids[t["id"]] = [p["id"] for p in t["roster"]]
            for p in t["roster"]:
                hp = world.hub_by_id.get(p.get("hub_id")) or {}
                self.rows[p["id"]] = {"espn": p.get("espn_level") or p.get("level") or 0.0, "age": hp.get("age")}
        self.cache = {}

    def pv(self, pid):
        pid = int(pid)
        if pid in self.cache:
            return self.cache[pid]
        P = LEAGUE_PV
        row = self.rows.get(pid, {})
        r = self.rd.get(str(pid), {})
        seen = row.get("espn") or r.get("espn_ppg") or r.get("ppg") or 0.0
        if (r.get("n_season") or 0) >= 5 and r.get("s_avg") is not None and r.get("sa_season") == r.get("gs"):
            seen = 0.5 * seen + 0.5 * r["s_avg"]
        prod = max(0.0, seen - P["repl"])
        n = self.pick.get(pid)
        draft = max(0.0, 150.0 - n) / 150.0 * P["draft_scale"] if n else 0.0
        v = P["w_prod"] * prod + (1 - P["w_prod"]) * draft
        age = row.get("age") if row.get("age") is not None else r.get("age")
        if age is not None and age <= P["youth_age"]:
            v *= 1 + P["youth"]
        self.cache[pid] = v
        return v

    def top5(self, team_id):
        return set(sorted(self.team_ids.get(team_id, []), key=lambda i: -self.pv(i))[:5])

    @staticmethod
    def side(vals):
        w = LEAGUE_PV["star_w"]
        return sum(v * (w[i] if i < len(w) else w[-1]) for i, v in enumerate(sorted(vals, reverse=True)))

    def accept(self, give_ids, get_ids, them_id, give_extra=(), get_extra=()):
        """the OTHER owner's view: he receives give_ids (+ picks given) and sends get_ids (+ picks). His likely keepers count extra when he sends them."""
        t5 = self.top5(them_id)
        recv = self.side([self.pv(i) for i in give_ids] + list(give_extra))
        sent = self.side([self.pv(i) * (1 + LEAGUE_PV["keeper_endow"] if int(i) in t5 else 1.0) for i in get_ids] + list(get_extra))
        if sent <= 0:
            return {"ratio": None, "band": "likely", "recv": round(recv, 1), "sent": round(sent, 1)}
        r = recv / sent
        band = "likely" if r >= LEAGUE_PV["likely"] else ("possible" if r >= LEAGUE_PV["possible"] else "unlikely")
        return {"ratio": round(r, 2), "band": band, "recv": round(recv, 1), "sent": round(sent, 1), "keepers_sent": [int(i) for i in get_ids if int(i) in t5]}


def pick_pv(slot_probs):
    """a pick as this league sees it: what the slot costs in its draft plus the player it usually brings, with the youth premium"""
    P = LEAGUE_PV
    v = 0.0
    for slot, pr in slot_probs.items():
        draft = max(0.0, 150.0 - slot) / 150.0 * P["draft_scale"]
        prod = max(0.0, PICK_K * (slot + PICK_C) ** (-PICK_P) - P["repl"])
        v += pr * (1 + P["youth"]) * (P["w_prod"] * prod + (1 - P["w_prod"]) * draft)
    return v


# =============================================================== draft picks (next draft)
# A pick is valued as the player it will turn into. Draft order = reverse of finish (regular-season rank for teams that miss the playoffs, playoff finish for the others), so the slot
# comes from the simulation's draft-position distribution for the pick's ORIGINAL owner. Snake: even rounds run backwards. What a slot produces = this league's own draft history
# (draft_capital_curve.py: 612 non-keeper picks 2021-25, realized fantasy points per game that season; power-law fit below, spread 15 early to 11 late).
PICK_K, PICK_C, PICK_P = 70.2, 1.96, 0.1995
PICK_STARTER_LEVEL = 30.0        # a drafted player only helps if he beats the replacement-level free agent (3rd-best free agent is about 27-30 points per game)
PICK_KEEPER_LEVEL = 42.0         # ... and he is kept (free) the next year if he is about a top-5 asset level
PICK_KEEP_YEARS = 0.95           # extra discounted year of value for a kept player
TEAMS_N = 12


def pick_slot_probs(draft_pos_row, rnd):
    """probability over the overall slot (1-based) for a real-round pick given the owner's draft-position distribution (row of T probabilities, index 0 = first pick)"""
    out = {}
    for k, pr in enumerate(draft_pos_row):
        pos = k + 1 if rnd % 2 == 1 else TEAMS_N - k
        out[(rnd - 1) * TEAMS_N + pos] = out.get((rnd - 1) * TEAMS_N + pos, 0.0) + float(pr)
    return out


def pick_value(slot_probs, lvl_to_v):
    """expected (points-per-game-equivalent for the title model, market value) of a pick"""
    grid = np.arange(0.0, 91.0, 1.0)
    ppg_eq, mkt = 0.0, 0.0
    for slot, pr in slot_probs.items():
        mu = PICK_K * (slot + PICK_C) ** (-PICK_P)
        sd = 15.0 - 3.5 * min(max(slot - 12, 0), 88) / 88
        w = np.exp(-0.5 * ((grid - mu) / sd) ** 2)
        w /= w.sum()
        ppg_eq += pr * float(np.sum(w * (np.maximum(0, grid - PICK_STARTER_LEVEL) + PICK_KEEP_YEARS * np.maximum(0, grid - PICK_KEEPER_LEVEL))))
        mkt += pr * float(np.sum(w * lvl_to_v(grid)))
    return ppg_eq, mkt


def level_to_market():
    """points-per-game level -> perceived market value, from the committed player table (binned medians, forced to rise)"""
    m = json.load(open(HUB / "trade_market.json", encoding="utf-8"))["players"]
    pts = sorted((v["lvl"], v["v"]) for v in m.values() if v.get("lvl"))
    xs = np.array([p[0] for p in pts])
    bins = np.linspace(xs.min(), xs.max(), 14)
    bx, by = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [v for x, v in pts if lo <= x <= hi]
        if sel:
            bx.append((lo + hi) / 2)
            by.append(float(np.median(sel)))
    by = np.maximum.accumulate(by)
    return lambda g: np.where(g < bx[0], 0.0, np.interp(g, bx, by))


def pick_owner_map(world):
    """(original team id, real round) -> current owner id, from the hand ledger of picks ESPN's board does not show"""
    led = ROOT / "draft_trades.json"
    out = {}
    if led.exists():
        for t in json.load(open(led, encoding="utf-8")).get("picks", []):
            out[(t["from"], t["real_round"])] = t["to"]
    return out


def pick_label(world, k):
    return f"{world.abbrev_of[k[0]]} round {k[1]} pick (next draft)"


def parse_picks(names, world):
    """'PICK:YUNG:1' -> (original team id, round); returns (players, picks)"""
    ps, pk = [], []
    for nm in names:
        if nm.upper().startswith("PICK:"):
            _, ab, r = nm.split(":")
            pk.append((world.by_abbrev[ab.strip().upper()], int(r)))
        else:
            ps.append(nm)
    return ps, pk


# =============================================================== the trade
def find(roster, names):
    out = []
    for nm in names:
        q = norm(nm)
        hit = next((p for p in roster if norm(p["name"]) == q), None) or next((p for p in roster if q and q in norm(p["name"])), None)
        if not hit:
            raise SystemExit(f"not on that roster: {nm}")
        out.append(hit)
    return out


def evaluate(me_abbrev, partner_abbrev, give_names, get_names, posture_me="auto", posture_them="auto", n=20000, world=None, inp=None, future_lambda=None, rid=None):
    global FUTURE_LAMBDA, KEEP_PTS
    if future_lambda is not None:
        FUTURE_LAMBDA = float(future_lambda)
        KEEP_PTS = 3.4 * 19 * FUTURE_LAMBDA
    W = world or World()
    inp = inp or sim_inputs(W)
    me_id, th_id = W.by_abbrev[me_abbrev], W.by_abbrev[partner_abbrev]
    give_names, give_picks = parse_picks(give_names, W)
    get_names, get_picks = parse_picks(get_names, W)
    r_me, r_th = W.roster(me_id), W.roster(th_id)
    give, get = find(r_me, give_names), find(r_th, get_names)
    r_me, _, _ = fix_roster(W, r_me, "me")            # baseline: rosters as they would legally stand (ESPN rosters can sit above 15 right after a draft)
    r_th, _, _ = fix_roster(W, r_th, "them")
    give = [p for p in r_me if p["id"] in {g["id"] for g in give}] or give
    get = [p for p in r_th if p["id"] in {g["id"] for g in get}] or get
    new_me = [p for p in r_me if p not in give] + get
    new_th = [p for p in r_th if p not in get] + give
    new_me, drops_me, adds_me = fix_roster(W, new_me, "me")
    new_th, drops_th, adds_th = fix_roster(W, new_th, "them")
    # weekly strengths before and after (the same engine both times), scaled onto the playoff-odds forecasts
    mu0, mu1 = {}, {}
    wk = {}
    for tid, before, after in ((me_id, r_me, new_me), (th_id, r_th, new_th)):
        b, a = W.weekly(before), W.weekly(after)
        wk[tid] = (b, a)
    for tid in W.teams:
        for w, v in W.base_proj[tid].items():
            mu0[(tid, w)] = v
            if tid in wk and w in wk[tid][0] and wk[tid][0][w] > 1:
                mu1[(tid, w)] = v * wk[tid][1][w] / wk[tid][0][w]
            else:
                mu1[(tid, w)] = v
    base = run_odds(W, inp, mu0, n)
    after = run_odds(W, inp, mu1, n)
    ids = inp[0]
    res = {}
    l2v = level_to_market()
    pv = {}
    for (oid, rnd) in set(give_picks) | set(get_picks):
        pv[(oid, rnd)] = pick_value(pick_slot_probs(after['draft_pos'][ids.index(oid)], rnd), l2v)
    for who, tid, post_in, rb, ra, drops, adds in (("me", me_id, posture_me, r_me, new_me, drops_me, adds_me), ("them", th_id, posture_them, r_th, new_th, drops_th, adds_th)):
        i = ids.index(tid)
        b = {k: float(base[k][i]) for k in ("made", "bye", "final", "title")}
        a = {k: float(after[k][i]) for k in ("made", "bye", "final", "title")}
        post = posture_auto(b["made"], b["title"]) if post_in == "auto" else post_in
        w_now, w_fut = POSTURE[post]
        slope = (W.ctx or {}).get("slope", {}).get(str(tid), DEFAULT_SLOPE) if W.ctx else DEFAULT_SLOPE
        s_next = (W.ctx or {}).get("slope_mean", DEFAULT_SLOPE) if W.ctx else DEFAULT_SLOPE
        c_asset = GAMES_PER_WEEK_PER_ASSET_PT * s_next
        k0, k1 = keeper_strength(rb), keeper_strength(ra)
        d_title_pp = 100 * (a["title"] - b["title"])
        v_now = w_now * d_title_pp
        pk_in, pk_out = (get_picks, give_picks) if who == "me" else (give_picks, get_picks)
        d_pick = sum(pv[k][0] for k in pk_in) - sum(pv[k][0] for k in pk_out)
        v_fut = w_fut * FUTURE_LAMBDA * c_asset * ((k1 - k0) + d_pick)
        wb, wa = wk[tid]
        res[who] = {"abbrev": me_abbrev if who == "me" else partner_abbrev, "posture": post, "posture_auto": post_in == "auto",
                    "odds_before": {k: round(v, 4) for k, v in b.items()}, "odds_after": {k: round(v, 4) for k, v in a.items()},
                    "d_title_pp": round(d_title_pp, 2), "d_made_pp": round(100 * (a["made"] - b["made"]), 2),
                    "season_points": {"regular": round(sum(wa[w] - wb[w] for w in wa if w not in W.playoff_weeks)), "playoffs": round(sum(wa[w] - wb[w] for w in wa if w in W.playoff_weeks))},
                    "keeper_strength": {"before": round(k0, 1), "after": round(k1, 1)},
                    "value": {"now": round(v_now, 2), "future": round(v_fut, 2), "total": round(v_now + v_fut, 2), "w_now": w_now, "w_future": w_fut},
                    "drops": [p["name"] for p in drops], "adds": [p["name"] for p in adds],
                    "spots_after": spots_used(ra), "ir_used": sum(1 for p in ra if p["out"])}
    LP = LeaguePerception(W)
    ppv = {k: pick_pv(pick_slot_probs(after["draft_pos"][ids.index(k[0])], k[1])) for k in set(give_picks) | set(get_picks)}
    acc = LP.accept([p["id"] for p in give], [p["id"] for p in get], th_id, [ppv[k] for k in give_picks], [ppv[k] for k in get_picks])
    gv = [LP.pv(p["id"]) for p in give] + [ppv[k] for k in give_picks]
    tv = [LP.pv(p["id"]) for p in get] + [ppv[k] for k in get_picks]
    acc["keepers_sent"] = [next(p["name"] for p in get if p["id"] == i) for i in acc.get("keepers_sent", [])]
    me_v, th_v = res["me"]["value"]["total"], res["them"]["value"]["total"]
    if me_v > 0.15 and acc["band"] != "unlikely":
        verdict = "good for you and the other owner is likely to see it as fair" if th_v > 0 else "good for you; they may accept on market value, but it is a worse deal for them than it looks"
    elif me_v > 0.15:
        verdict = "good for you but the other owner will probably see it as a loss in market value"
    elif me_v < -0.15:
        verdict = "bad for you"
    else:
        verdict = "close to neutral for you"
    return {"generated": datetime.now(timezone.utc).isoformat(), "rid": rid, "give": [p["name"] for p in give] + [pick_label(W, k) for k in give_picks], "get": [p["name"] for p in get] + [pick_label(W, k) for k in get_picks],
            "picks": {pick_label(W, k): {"owner_finish_slot": round(sum(s * pr for s, pr in pick_slot_probs(after["draft_pos"][ids.index(k[0])], k[1]).items()), 1), "ppg_equiv": round(v[0], 1), "market": round(v[1])} for k, v in pv.items()}, "me": res["me"], "them": res["them"],
            "market": {"you_send_value": [round(x, 1) for x in gv], "you_get_value": [round(x, 1) for x in tv], "model": "league perception (production + this league's draft/keeper price, youth premium, star premium, keeper attachment)", **acc}, "verdict": verdict,
            "assumptions": {"future_lambda": FUTURE_LAMBDA, "asset_noise": ASSET_NOISE, "chip_weight": CHIP_WEIGHT, "ramp_mult": RAMP_MULT, "sims": n, "proj_basis": W.proj_basis}}


def report(r):
    f = lambda x: f"{100 * x:.1f}%"
    print(f"\nGIVE {', '.join(r['give'])}   GET {', '.join(r['get'])}\n")
    for who in ("me", "them"):
        x = r[who]
        print(f"{x['abbrev']} ({x['posture']}{' (auto)' if x['posture_auto'] else ''}): title {f(x['odds_before']['title'])} -> {f(x['odds_after']['title'])} ({x['d_title_pp']:+.2f} pts), playoffs {f(x['odds_before']['made'])} -> {f(x['odds_after']['made'])}, "
              f"season points {x['season_points']['regular']:+d} regular / {x['season_points']['playoffs']:+d} playoffs, keeper strength {x['keeper_strength']['before']} -> {x['keeper_strength']['after']}")
        print(f"    value now {x['value']['now']:+.2f}  future {x['value']['future']:+.2f}  TOTAL {x['value']['total']:+.2f}   drops {x['drops'] or '-'}  adds {x['adds'] or '-'}  spots {x['spots_after']}/15, injured {x['ir_used']}")
    m = r["market"]
    print(f"\nMarket view for THEM: they get {m['recv']} value, send {m['sent']} (ratio {m['ratio']}): acceptance {m['band'].upper()}")
    print("VERDICT:", r["verdict"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default=None)
    ap.add_argument("--partner", required=True)
    ap.add_argument("--give", required=True)
    ap.add_argument("--get", required=True)
    ap.add_argument("--posture-me", default="auto")
    ap.add_argument("--posture-them", default="auto")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--rid", default=None)
    a = ap.parse_args()
    W = World()
    me = a.me or W.wp["my_abbrev"]
    r = evaluate(me, a.partner, [x.strip() for x in a.give.split(";") if x.strip()], [x.strip() for x in a.get.split(";") if x.strip()], a.posture_me, a.posture_them, a.n, world=W, future_lambda=a.lam, rid=a.rid)
    report(r)
    if a.out:
        Path(a.out).write_text(json.dumps(r, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
