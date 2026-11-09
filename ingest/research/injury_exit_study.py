"""Injury-exit study (2026-11-09, Tommy, after Kon Knueppel: back from injury, played 6 minutes in his first game, aggravated it; season-mode DELCO read the 6 minutes as
a role cut and fell 40.1 -> 33.1, expected minutes 31 -> 26).

Question: when a player leaves a game hurt, should that game's MINUTES count as evidence about his role?
Injury exit = minutes under half of what DELCO expected going in (season mode) AND his next game is 6+ days later (he missed games after it) or never comes that season.
Two versions of season-mode DELCO's minutes filter (same settings as kalman_vor.py: kalman_fit_MIN.json + delco_season_mode.json):
  as is   every game updates expected minutes
  masked  an injury exit leaves expected minutes unchanged (per-minute rates still update as usual: a 6-minute game barely moves them anyway)
Scored on what happened when he came back: his average fantasy points (and minutes) over his next 10 games after the exit, vs DELCO right after the exit.
Fantasy points: masked DELCO = DELCO as is x (masked minutes / as-is minutes). Seasons 2011-12..2025-26. Also checked: every other game is untouched (no change elsewhere).
Run from ingest/:  .venv/Scripts/python research/injury_exit_study.py        (writes data/injury_exit_study.log)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalman_engine import run_filter_all_players  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
R_ = Path(__file__).resolve().parent
D = R_ / "data"
OUT = open(D / "injury_exit_study.log", "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")


df = pd.read_csv(D / "kalman_input.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
k = pd.read_csv(D / "delco_pregame_all.csv.gz")
assert len(k) == len(df) and (k.PLAYER_ID.to_numpy() == df.PLAYER_ID.to_numpy()).all(), "delco_pregame_all is out of step with kalman_input: rerun kalman_vor.py"
pid = df.PLAYER_ID.to_numpy()
days = df.DAYS_SINCE_LAST.to_numpy(float)
age = df.AGE_AT_GAME.to_numpy(float)
mins = df.MIN.to_numpy(float)
seas = df.SEASON.to_numpy()
newp = np.r_[True, pid[1:] != pid[:-1]]
season_start = np.r_[False, (seas[1:] != seas[:-1]) & ~newp[1:]]
fit = json.load(open(D / "kalman_fit_MIN.json"))["params"]
Q, R, peak, su, sd = fit
sm = json.load(open(D / "delco_season_mode.json"))["params"]["MIN"]
fp = (df.PTS + 1.5 * df.REB + 2 * df.AST + 3 * df.STL + 3 * df.BLK + df.FG3M + 2 * df.FTM - df.FTA - df.TOV + 3 * df.TD3).to_numpy(float)

pre_min = k.preMIN.to_numpy(float)
nxt_same = np.r_[pid[1:] == pid[:-1], False] & np.r_[seas[1:] == seas[:-1], False]
gap_next = np.r_[days[1:], np.nan]
flag = (mins < 0.5 * pre_min) & (pre_min >= 15) & ((~nxt_same) | (gap_next >= 6)) & (seas >= "2011-12")
say(f"games {len(df):,}; injury exits flagged {flag.sum():,} ({flag.mean():.2%})")

x0 = float(mins.sum()) / len(mins)
gain = np.ones_like(mins)
a_post = run_filter_all_players(pid, days, age, gain, mins, Q, R, peak, su, sd, x0, q_mult=float(sm["QM"]), season_start=season_start, jump=float(sm["J"]) * R)
g2 = gain.copy()
g2[flag] = 0.0
m_post = run_filter_all_players(pid, days, age, g2, mins, Q, R, peak, su, sd, x0, q_mult=float(sm["QM"]), season_start=season_start, jump=float(sm["J"]) * R)
say(f"check: the as-is filter matches kalman_vor's season-mode minutes within 0.5 on {np.mean(np.abs(a_post - k.afterMIN.to_numpy()) < 0.5):.1%} of games")

rows = []
idx = np.where(flag)[0]
for i in idx:
    j = i + 1
    fut = []
    while j < len(df) and pid[j] == pid[i] and seas[j] == seas[i] and len(fut) < 10:
        fut.append(j)
        j += 1
    if len(fut) < 5:
        continue
    fut = np.array(fut)
    late = []
    while j < len(df) and pid[j] == pid[i] and seas[j] == seas[i] and len(late) < 30:
        late.append(j)
        j += 1
    late = np.array(late, dtype=int)
    after = k["after"].iloc[i]
    rows.append({"pid": pid[i], "name": df.PLAYER_NAME.iloc[i], "season": seas[i], "min": mins[i], "pre_min": pre_min[i], "first_game": bool(season_start[i]), "age": age[i],
                 "pre_fp": k["pre"].iloc[i], "asis_fp": after, "mask_fp": after * (m_post[i] / a_post[i] if a_post[i] > 0 else 1.0),
                 "asis_min": a_post[i], "mask_min": m_post[i], "fut_fp": fp[fut].mean(), "fut_min": mins[fut].mean(), "n": len(fut),
                 "late_fp": fp[late].mean() if len(late) >= 10 else np.nan, "late_min": mins[late].mean() if len(late) >= 10 else np.nan, "all_fp": fp[np.r_[fut, late]].mean()})
S = pd.DataFrame(rows)
say(f"exits with 5+ games after the return: {len(S):,}  (first game of a season: {S.first_game.sum()})\n")


def score(x, lab):
    e1, e2 = np.abs(x.asis_fp - x.fut_fp), np.abs(x.mask_fp - x.fut_fp)
    m1, m2 = np.abs(x.asis_min - x.fut_min), np.abs(x.mask_min - x.fut_min)
    say(f"{lab:38s} n={len(x):5d}  fantasy pts error: as is {e1.mean():5.2f}  masked {e2.mean():5.2f}  | minutes error: as is {m1.mean():4.2f}  masked {m2.mean():4.2f}"
        f"  | masked better in {np.mean(e2 < e1):.0%}  | avg DELCO as is {x.asis_fp.mean():.1f}, masked {x.mask_fp.mean():.1f}, actual next 10 {x.fut_fp.mean():.1f}")


score(S, "all injury exits")
L = S.dropna(subset=["late_fp"]).copy()
for lab, x in (("games 11-40 after the return (all)", L), ("games 11-40, expected 25+ minutes", L[L.pre_min >= 25]), ("games 11-40, first game of season", L[L.first_game])):
    e1, e2 = np.abs(x.asis_fp - x.late_fp), np.abs(x.mask_fp - x.late_fp)
    say(f"{lab:38s} n={len(x):5d}  fantasy pts error: as is {e1.mean():5.2f}  masked {e2.mean():5.2f}  | masked better in {np.mean(e2 < e1):.0%}  | avg DELCO as is {x.asis_fp.mean():.1f}, masked {x.mask_fp.mean():.1f}, before the exit {x.pre_fp.mean():.1f}, actual {x.late_fp.mean():.1f} (minutes {x.late_min.mean():.1f} vs expected before {x.pre_min.mean():.1f})")
Y = L[(L.age < 24) & (L.pre_min >= 25)]
e1, e2 = np.abs(Y.asis_fp - Y.late_fp), np.abs(Y.mask_fp - Y.late_fp)
say(f"{'games 11-40, under 24 and 25+ minutes':38s} n={len(Y):5d}  as is {e1.mean():5.2f}  masked {e2.mean():5.2f}  | masked better in {np.mean(e2 < e1):.0%}  | DELCO as is {Y.asis_fp.mean():.1f}, masked {Y.mask_fp.mean():.1f}, before {Y.pre_fp.mean():.1f}, actual {Y.late_fp.mean():.1f}")
x = S[S.pre_min >= 25]
e1, e2 = np.abs(x.asis_fp - x.all_fp), np.abs(x.mask_fp - x.all_fp)
say(f"{'next 40 games, expected 25+ minutes':38s} n={len(x):5d}  as is {e1.mean():5.2f}  masked {e2.mean():5.2f}  | masked better in {np.mean(e2 < e1):.0%}")
score(S[S.first_game], "first game of the season")
score(S[S["min"] < 0.25 * S.pre_min], "under a quarter of expected minutes")
score(S[S.season >= "2021-22"], "2021-22 onward")
for lo, hi in ((15, 25), (25, 32), (32, 48)):
    score(S[(S.pre_min >= lo) & (S.pre_min < hi)], f"expected {lo}-{hi} minutes")
# placebo: big minute drops where he did NOT miss games after (blowouts, foul trouble, real role cuts): masking those would be wrong
pl = (mins < 0.5 * pre_min) & (pre_min >= 15) & nxt_same & (gap_next <= 3) & (seas >= "2011-12")
say(f"\nfor contrast, big minute drops WITHOUT missed games after: {pl.sum():,} (not masked); their next-10 minutes vs expected before: "
    f"{np.mean([mins[i + 1:i + 11].mean() / pre_min[i] for i in np.where(pl)[0] if pid[min(i + 10, len(pid) - 1)] == pid[i]]):.0%}")
say(f"injury exits: next-10 minutes vs expected before: {(S.fut_min / S.pre_min).mean():.0%}")
OUT.close()
