"""Trade tag study (2026-11-09, Trades Command Center): which player tags hold up on past seasons, and where to put the cutoffs.

At 5, 8, 12 and 20 games into each season 2011-12..2025-26 (players DELCO opened at 20+, 15+ games that season, 10+ games after the checkpoint):
  sofar = his average so far, cur = DELCO after that game, open = DELCO's opening number, fut = his average over the next 20 games.
A. BUY LOW / SELL HIGH: by gap = sofar - cur. Does the rest of the season land nearer DELCO or nearer what he has averaged? Tag only where the average misleads by 3+.
B. UNDER / OVERPERFORMING (vs his opening number): same, by sofar - open. Descriptive, but checked so the tag does not imply more than it means.
C. RISING ROLE: minutes over his last 3 games vs his minutes so far. Does a jump of 4+ minutes hold over the next 20?
Run from ingest/:  .venv/Scripts/python research/trade_tag_study.py        (writes data/trade_tag_study.log)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.as_posix())
import form_common as FC  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).resolve().parent / "data"
OUT = open(D / "trade_tag_study.log", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")


g = FC.load_all()[["pid", "date", "fp"]].copy()
g["date"] = pd.to_datetime(g["date"]).dt.normalize()
g = g.groupby(["pid", "date"], as_index=False).fp.max()
k = pd.read_csv(D / "delco_pregame_all.csv.gz", usecols=["PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "SEASON", "MIN", "pre", "after"], low_memory=False)
k["date"] = pd.to_datetime(k.GAME_DATE).dt.normalize()
k = k[k.MIN > 0]
m = k.merge(g, left_on=["PLAYER_ID", "date"], right_on=["pid", "date"]).sort_values(["PLAYER_ID", "date"])

rows = []
for (pid, season), x in m.groupby(["PLAYER_ID", "SEASON"], sort=False):
    if season < "2011-12" or season > "2025-26" or len(x) < 15:
        continue
    fp_, after, pre, mins = x.fp.to_numpy(float), x["after"].to_numpy(float), x["pre"].to_numpy(float), x.MIN.to_numpy(float)
    if np.isnan(pre[0]) or pre[0] < 20:
        continue
    for n in (5, 8, 12, 20):
        fut = fp_[n:n + 20]
        if len(fut) < 10 or np.isnan(after[n - 1]):
            continue
        rows.append({"pid": pid, "season": season, "n": n, "open": pre[0], "cur": after[n - 1], "sofar": fp_[:n].mean(), "fut": fut.mean(),
                     "min_sofar": mins[:n].mean(), "min_l3": mins[n - 3:n].mean(), "min_fut": mins[n:n + 20].mean()})
P = pd.DataFrame(rows)
say(f"checkpoints {len(P):,}, player-seasons {P.groupby(['pid', 'season']).ngroups:,}\n")


def table(col, bins, title):
    say(title)
    say("   bin            n    avg gap   next 20 minus avg so far   next 20 minus DELCO   share landing nearer DELCO")
    for lo, hi in bins:
        x = P[(P[col] >= lo) & (P[col] < hi)]
        if len(x) < 20:
            continue
        near = (np.abs(x.fut - x.cur) < np.abs(x.fut - x.sofar)).mean()
        say(f"   {lo:+4.0f} to {hi:+4.0f}  {len(x):5d}   {x[col].mean():+6.1f}   {(x.fut - x.sofar).mean():+12.2f}            {(x.fut - x.cur).mean():+8.2f}             {near:6.0%}")
    say("")


P["gap"] = P.sofar - P.cur
P["vs_open"] = P.sofar - P.open
BINS = [(-99, -10), (-10, -7), (-7, -5), (-5, -3), (-3, 3), (3, 5), (5, 7), (7, 10), (10, 99)]
table("gap", BINS, "A. by average so far minus DELCO (negative = averaging under DELCO)")
for n in (5, 8, 12, 20):
    x = P[P.n == n]
    lo, hi = x[x.gap <= -5], x[x.gap >= 5]
    say(f"   at {n:2d} games: 5+ under DELCO n={len(lo):4d}, next 20 minus avg so far {(lo.fut - lo.sofar).mean():+.2f}  |  5+ over n={len(hi):4d}, next 20 minus avg so far {(hi.fut - hi.sofar).mean():+.2f}, minus DELCO {(hi.fut - hi.cur).mean():+.2f}")
say("")
table("vs_open", BINS, "B. by average so far minus DELCO's opening number")

say("C. minutes over the last 3 games minus minutes so far")
say("   bin            n   jump   next 20 minutes minus minutes so far   share of the jump kept")
P["mjump"] = P.min_l3 - P.min_sofar
for lo, hi in [(-99, -4), (-4, 2), (2, 4), (4, 6), (6, 99)]:
    x = P[(P.mjump >= lo) & (P.mjump < hi)]
    if len(x) < 20:
        continue
    kept = ((x.min_fut - x.min_sofar) / x.mjump.where(x.mjump.abs() > 0.5)).median()
    say(f"   {lo:+4.0f} to {hi:+4.0f}  {len(x):5d}  {x.mjump.mean():+5.1f}   {(x.min_fut - x.min_sofar).mean():+8.2f}                          {kept:6.0%}")
OUT.close()
