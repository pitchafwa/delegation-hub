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

# ---------------- PART 2 (2026-11-09, Tommy: "if he had simply missed the game his DELCO would have stayed higher"): the like-for-like comparison.
# Where the site values an injured player (trade analyzer, plan), his output after a return is ALREADY cut by the measured return ramp (injury_advisor.ramp_ratio,
# ~1,000 real returns). So the fair question: with the ramp applied on top, which is right after an injury EXIT: DELCO as is, or DELCO as if he had simply missed that game?
# And for contrast: plain absences (no exit game), where DELCO is unchanged by the absence and only the ramp applies.
import injury_advisor as IA  # noqa: E402

OUT = open(D / "injury_exit_study.log", "a", encoding="utf-8")


def ramped(i, base, total_out, n=40):
    j, g, pr, ac = i + 1, 0, [], []
    while j < len(df) and pid[j] == pid[i] and seas[j] == seas[i] and g < n:
        g += 1
        pr.append(base * IA.ramp_ratio(total_out, g))
        ac.append(fp[j])
        j += 1
    return np.array(pr), np.array(ac)


res = {"exit": [], "plain": []}
for i in idx:
    if not nxt_same[i] or pre_min[i] < 25:
        continue
    tot = int(round(gap_next[i] / 2.1))                   # team games missed after the exit (about one every 2.1 days)
    if tot < 5:
        continue
    a = k["after"].iloc[i]
    m = a * (m_post[i] / a_post[i] if a_post[i] > 0 else 1.0)
    pa, ac = ramped(i, a, tot)
    pm, _ = ramped(i, m, tot)
    if len(ac) >= 10:
        res["exit"].append((np.abs(pa[:10] - ac[:10]).mean(), np.abs(pm[:10] - ac[:10]).mean(), np.abs(pa - ac).mean(), np.abs(pm - ac).mean(), pa.mean(), pm.mean(), ac.mean(), age[i] < 24))
# plain absences: a game, then 10+ days off (5+ games), no short exit; DELCO going into the return game = the belief after the last game played
pa_idx = np.where(nxt_same & (gap_next >= 10.5) & (mins >= 0.7 * pre_min) & (pre_min >= 25) & (seas >= "2011-12"))[0]
for i in pa_idx:
    tot = int(round(gap_next[i] / 2.1))
    pa, ac = ramped(i, k["after"].iloc[i], tot)
    if len(ac) >= 10:
        res["plain"].append((np.abs(pa[:10] - ac[:10]).mean(), np.abs(pa - ac).mean(), pa.mean(), ac.mean()))
E = np.array(res["exit"])
P_ = np.array(res["plain"])
say("\nPART 2. with the return ramp applied (regulars, 25+ expected minutes, 5+ games missed)")
say(f"  plain absences   n={len(P_):5d}  next 10 error {P_[:, 0].mean():.2f} | next 40 error {P_[:, 1].mean():.2f} | predicted {P_[:, 2].mean():.1f} vs actual {P_[:, 3].mean():.1f}")
say(f"  injury exits     n={len(E):5d}  next 10 error: as is {E[:, 0].mean():.2f}, as if missed {E[:, 1].mean():.2f} | next 40: as is {E[:, 2].mean():.2f}, as if missed {E[:, 3].mean():.2f}"
    f" | predicted as is {E[:, 4].mean():.1f}, as if missed {E[:, 5].mean():.1f}, actual {E[:, 6].mean():.1f}; 'as if missed' better in {np.mean(E[:, 3] < E[:, 2]):.0%}")
Y = E[E[:, 7] == 1]
say(f"  ... under 24     n={len(Y):5d}  next 40: as is {Y[:, 2].mean():.2f}, as if missed {Y[:, 3].mean():.2f} | predicted as is {Y[:, 4].mean():.1f}, as if missed {Y[:, 5].mean():.1f}, actual {Y[:, 6].mean():.1f}")
OUT.close()
