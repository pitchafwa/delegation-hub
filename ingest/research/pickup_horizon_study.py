"""Pickup horizon study (2026-11-12, Tommy: "anyone I pick up will almost certainly become a dropped player in the next 10-15 days"; and DELCO is slow on risers like Traore).

The plan credits a pickup it KEEPS with the season-long level gap to the player dropped (fut_hold, x 0.6 when the add is better), and ranks drops by DELCO alone.
Two questions:

A. HOW LONG DO PICKUPS STAY? Every executed free-agent / waiver ADD in this league, 2023-24 .. 2026-27 (data/league_transactions_<year>.json), followed day by day through
   that team's real roster (data/league_days_<year>.json) until he leaves. Days kept and games played for that team, all teams and Tommy's (DRNK, id 12) alone, by how well he
   scored while there. This season's adds still on a roster are censored (shown separately).

B. DOES THE TREND ADD ANYTHING TO DELCO FOR THE NEXT 2-3 WEEKS? Every regular-season game of 2011-12 .. 2025-26 where a player had 10+ games so far and DELCO (after that game)
   between 18 and 38 (the waiver / bench tier), with 7+ games after it. trend = his last-10 average minus DELCO. Target = his average over the next 7 and next 15 games.
   Fit fut - DELCO = k x trend (one k per horizon), leave-one-season-out; report the error vs DELCO alone, and the bins (risers and fallers) so k is not hiding a shape.

Run from ingest/:  .venv/Scripts/python research/pickup_horizon_study.py      (writes data/pickup_horizon_study.log)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import form_common as FC  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
OUT = open(D / "pickup_horizon_study.log", "w", encoding="utf-8")
ME = 12


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")


# ---------------- A. tenure ----------------
say("A. HOW LONG PICKUPS STAY (free-agent and waiver adds)")
rows = []
for y in (2024, 2025, 2026, 2027):
    tf, lf = D / f"league_transactions_{y}.json", D / f"league_days_{y}.json"
    if not (tf.exists() and lf.exists()):
        continue
    days = json.load(open(lf))
    played = [int(k) for k, v in days.items() if v and any(e[3] for r in v for e in r["e"])]
    if not played:
        continue
    last = max(played)
    ros = {}                                                     # (sp, team) -> {pid: (played, pts)}
    for k, v in days.items():
        for r in v:
            ros[(int(k), r["team"])] = {e[0]: (e[3], e[4]) for e in r["e"]}
    for t in json.load(open(tf)):
        if t.get("status") != "EXECUTED" or t.get("type") not in ("FREEAGENT", "WAIVER"):
            continue
        sp0 = t.get("scoringPeriodId") or 0
        for it in t.get("items", []):
            if it.get("type") != "ADD" or not sp0 or sp0 > last:
                continue
            team, pid = it.get("toTeamId"), it.get("playerId")
            sp, g, pts = sp0, 0, []
            while sp <= last and pid in ros.get((sp, team), {}):
                pl, p_ = ros[(sp, team)][pid]
                if pl:
                    g += 1
                    if p_ is not None:
                        pts.append(p_)
                sp += 1
            if sp == sp0 and pid not in ros.get((sp0 + 1, team), {}):
                continue                                         # never showed up (same-day add/drop or a period mismatch)
            rows.append({"year": y, "team": team, "pid": pid, "days": sp - sp0, "games": g, "ppg": np.mean(pts) if pts else np.nan, "censored": sp > last})
A = pd.DataFrame(rows)
done = A[~A.censored]


def tenure(x, label):
    if len(x) < 5:
        return
    q = x.days.quantile([0.25, 0.5, 0.75]).to_numpy()
    say(f"   {label:34s} n={len(x):4d}  days kept: median {q[1]:4.0f} (middle half {q[0]:.0f}-{q[2]:.0f})   games for the team: median {x.games.median():3.0f}"
        f"   kept 15+ days {np.mean(x.days >= 15):4.0%}   30+ {np.mean(x.days >= 30):4.0%}   60+ {np.mean(x.days >= 60):4.0%}")


tenure(done, "all teams, all seasons")
tenure(done[done.team == ME], "Tommy (DRNK), all seasons")
for y in sorted(done.year.unique()):
    tenure(done[done.year == y], f"all teams, season {y - 1}-{str(y)[2:]}")
tenure(done[(done.year == 2027) & (done.team == ME)], "Tommy, this season")
say("   by how he scored while on the team (all teams, all seasons):")
for lo, hi in [(0, 20), (20, 25), (25, 30), (30, 35), (35, 99)]:
    tenure(done[(done.ppg >= lo) & (done.ppg < hi)], f"   {lo}-{hi} ppg")
c = A[A.censored & (A.year == 2027)]
say(f"   this season, still on a roster now: {len(c)} adds (not counted above), {int((c.team == ME).sum())} of them Tommy's")
say("")

# ---------------- B. trend vs DELCO ----------------
say("B. DOES THE TREND ADD TO DELCO FOR THE NEXT 7 / 15 GAMES? (DELCO 18-38, 10+ games so far, 2011-12 .. 2025-26 regular seasons)")
g = FC.load_all()[["pid", "date", "fp"]].copy()
g["date"] = pd.to_datetime(g["date"]).dt.normalize()
g = g.groupby(["pid", "date"], as_index=False).fp.max()
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "GAME_DATE", "SEASON", "MIN", "after"], low_memory=False)
k["date"] = pd.to_datetime(k.GAME_DATE).dt.normalize()
k = k[k.MIN > 0]
m = k.merge(g, left_on=["PLAYER_ID", "date"], right_on=["pid", "date"]).sort_values(["PLAYER_ID", "date"])
P = []
for (pid, season), x in m.groupby(["PLAYER_ID", "SEASON"], sort=False):
    if season < "2011-12" or season > "2025-26" or len(x) < 17:
        continue
    fp_, cur = x.fp.to_numpy(float), x["after"].to_numpy(float)
    cs = np.concatenate([[0], np.cumsum(fp_)])
    for i in range(9, len(x) - 7, 3):                            # every 3rd game, so neighbouring checkpoints overlap less
        c_ = cur[i]
        if np.isnan(c_) or not (18 <= c_ <= 38):
            continue
        n15 = min(15, len(x) - 1 - i)
        P.append({"season": season, "cur": c_, "l10": (cs[i + 1] - cs[i - 9]) / 10, "l5": (cs[i + 1] - cs[i - 4]) / 5,
                  "f7": (cs[i + 8] - cs[i + 1]) / 7, "f15": (cs[i + 1 + n15] - cs[i + 1]) / n15, "n15": n15})
P = pd.DataFrame(P)
P["tr"], P["tr5"] = P.l10 - P.cur, P.l5 - P.cur
say(f"   checkpoints {len(P):,}")
for tgt, lab in (("f7", "next 7 games"), ("f15", "next 15 games")):
    y_ = P[tgt] - P.cur
    for feat in ("tr", "tr5"):
        pred = np.zeros(len(P))
        for s in P.season.unique():                              # leave one season out
            tr_, te_ = P.season != s, P.season == s
            kk = (P.loc[tr_, feat] * y_[tr_]).sum() / (P.loc[tr_, feat] ** 2).sum()
            pred[te_.to_numpy()] = kk * P.loc[te_, feat]
        kk_all = (P[feat] * y_).sum() / (P[feat] ** 2).sum()
        e0, e1 = np.abs(y_).mean(), np.abs(y_ - pred).mean()
        say(f"   {lab:14s} {'last-10' if feat == 'tr' else 'last-5 '} trend: k = {kk_all:.2f}   error DELCO alone {e0:.2f} -> with trend {e1:.2f} ppg  ({(e0 - e1) / e0:+.1%})")
say("   by trend (last-10 average minus DELCO):")
say("      bin          n     next 7 minus DELCO   next 15 minus DELCO   share of the trend kept (15)")
for lo, hi in [(-99, -8), (-8, -5), (-5, -2), (-2, 2), (2, 5), (5, 8), (8, 99)]:
    x = P[(P.tr >= lo) & (P.tr < hi)]
    if len(x) < 30:
        continue
    say(f"      {lo:+4.0f} to {hi:+4.0f} {len(x):6d}        {(x.f7 - x.cur).mean():+6.2f}               {(x.f15 - x.cur).mean():+6.2f}               {(x.f15 - x.cur).mean() / x.tr.mean():5.0%}")
say("")
say("   what that means for a Traore-vs-Wiggins call: two players DELCO has 4 apart, one trending +5 over DELCO, the other -3")
kk = (P.tr * (P.f15 - P.cur)).sum() / (P.tr ** 2).sum()
say(f"      next 15 games: gap {4 - kk * 5 - kk * 3:+.1f} instead of +4.0 (k = {kk:.2f})")
OUT.close()
