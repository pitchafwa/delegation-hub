"""Waiver-claim competition intel from three seasons of this league's transactions (2024, 2025, 2026 files from waiver_history_check.py + league_days for player quality).
For every player a team DROPPED (he sat on waivers for 24h+): was he claimed, by whom, how many teams claimed him (executed claim + claims that failed because he was already awarded).
Output dashboard/waiver_intel.json:
  tiers          per quality tier (by the player's points per game over the next 3 weeks): drops, share claimed, claims per drop
  teams          per ESPN team id: claims by tier, share of the league's claims (smoothed), wins/losses
  The alert uses it: P(you win) = product over teams AHEAD of you in waiver order of (1 - P(that team claims him)), P(team claims) = claims-per-drop(tier) x team share.
Run:  uv run python research/waiver_intel.py        (from ingest/)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent.parent.parent / "dashboard" / "waiver_intel.json"
TIERS = [("star", 38.0), ("good", 30.0), ("filler", 0.0)]          # points per game cut-offs
WINDOW_MS = 4 * 24 * 3600 * 1000


def tier_of(ppg):
    for name, lo in TIERS:
        if ppg >= lo:
            return name
    return "filler"


def player_ppg(year):
    """pid -> list of (scoring period, points) for games played, from the daily lineups of every team"""
    days = json.load(open(D / f"league_days_{year}.json"))
    res = collections.defaultdict(dict)
    for d, recs in days.items():
        for r in recs:
            for e in r["e"]:
                if e[3] and e[4] is not None:
                    res[e[0]][int(d)] = e[4]
    return res


def main():
    drops, claims = [], []
    for year in (2024, 2025, 2026):
        tx = json.load(open(D / f"league_transactions_{year}.json"))
        pts = player_ppg(year)
        for t in tx:
            ts = t.get("processDate") or t.get("proposedDate")
            if not ts:
                continue
            for it in t.get("items", []):
                pid = it.get("playerId")
                if it.get("type") == "DROP" and t.get("status") == "EXECUTED" and t.get("type") in ("FREEAGENT", "WAIVER", "ROSTER"):
                    drops.append({"year": year, "pid": pid, "ts": ts, "team": it.get("fromTeamId"), "day": t.get("scoringPeriodId")})
                if it.get("type") == "ADD" and t.get("type") == "WAIVER" and t.get("status") in ("EXECUTED", "FAILED_INVALIDPLAYERSOURCE"):
                    claims.append({"year": year, "pid": pid, "ts": ts, "team": t.get("teamId"), "won": t.get("status") == "EXECUTED"})
        for dr in [d for d in drops if d["year"] == year]:
            g = [v for p, v in pts.get(dr["pid"], {}).items() if dr["day"] and dr["day"] <= p <= dr["day"] + 21]
            dr["ppg"] = float(np.mean(g)) if len(g) >= 2 else 0.0
    # attach claims to the drop they answer: the latest drop of that player before the claim, within the window; the dropper cannot claim him back inside the window anyway
    by = collections.defaultdict(list)
    for d in drops:
        by[(d["year"], d["pid"])].append(d)
    for c in claims:
        cand = [d for d in by.get((c["year"], c["pid"]), []) if 0 <= c["ts"] - d["ts"] <= WINDOW_MS]
        c["drop"] = max(cand, key=lambda d: d["ts"]) if cand else None
    matched = [c for c in claims if c["drop"]]
    print(f"drops {len(drops)}, claims {len(claims)} ({len(matched)} matched to a drop)")
    # one waiver episode = one drop; claims per episode
    tiers = {}
    for name, _ in TIERS:
        ds = [d for d in drops if tier_of(d["ppg"]) == name]
        cs = [c for c in matched if tier_of(c["drop"]["ppg"]) == name]
        cl = collections.Counter(id(c["drop"]) for c in cs)
        tiers[name] = {"drops": len(ds), "claimed_share": round(len(cl) / max(1, len(ds)), 3), "claims_per_drop": round(len(cs) / max(1, len(ds)), 3),
                       "contested_share": round(sum(1 for v in cl.values() if v >= 2) / max(1, len(ds)), 3)}
        print(f"  {name:<7} drops {len(ds):<5} claimed {tiers[name]['claimed_share']:.1%}  claims/drop {tiers[name]['claims_per_drop']:.2f}  contested {tiers[name]['contested_share']:.1%}")
    # teams: claim share (smoothed toward uniform), wins / losses. ESPN team ids can change owner across years; ids are stable within the league
    cnt = collections.Counter(c["team"] for c in claims)
    won = collections.Counter(c["team"] for c in claims if c["won"])
    tot = sum(cnt.values())
    teams = sorted(cnt)
    prior = 2.0                                                             # pseudo-claims per team so a team with no history still has a small share
    share = {t: (cnt[t] + prior) / (tot + prior * 12) for t in teams}
    out_teams = {str(t): {"claims": cnt[t], "wins": won[t], "losses": cnt[t] - won[t], "share": round(share[t], 4)} for t in teams}
    # per-team recent behaviour: last season only (owners change)
    last = collections.Counter(c["team"] for c in claims if c["year"] == 2026)
    for t in out_teams:
        out_teams[t]["claims_2026"] = last[int(t)]
    print("team claims (3 seasons) / wins / last season:", {t: (v["claims"], v["wins"], v["claims_2026"]) for t, v in sorted(out_teams.items(), key=lambda x: -x[1]["claims"])})
    # how often does the player end up claimed at all, and by how many, when a good player is available: this is what matters for 'who competes'
    # validation: predict, for each matched claim episode in 2026, the winner among claimants by waiver order? ESPN order isn't in the log, so skip; report calibration of 'claimed or not' by tier
    res = {"generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"), "tiers": tiers, "teams": out_teams,
           "tier_cuts": {n: lo for n, lo in TIERS}, "note": "claims per drop and team shares from 2024-2026; P(team claims him) = claims_per_drop(tier) x team share, capped at 0.9 (shares of unseen teams default to 1/12)"}
    OUT.write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT)


main()
