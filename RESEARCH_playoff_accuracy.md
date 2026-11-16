# Rest-of-season projection and playoff-odds accuracy study

Started 2026-11-16 (Tommy): tune the future-weeks projection behind playoff odds and projected record "as accurately as humanly possible".
Recommend only; Tommy approves each change; after any change to the season forecast or playoff sim, rerun `rebuild_odds_history.py`.

## How accuracy is judged
Replay 2025-26 (real rosters and box scores, `backtest_phase2.py`) plus 2026-27 weeks so far. From checkpoints (weeks 3, 6, 9, 12, 15),
using only what was known then, forecast every remaining week, rest-of-season totals, final standings, playoff spots and title odds; score
against what happened. A change is kept only if it wins on checkpoints it was not tuned on. Effects too small to separate from noise with
one past season are reported as "can't tell yet".

## Items (order: 1, 4, 7 first)
1. Roster churn / manager moves: held roster vs how teams really change; per-manager move value; how fast the held roster stops predicting.
2. Player levels weeks out: season DELCO, next-few-days DELCO, ESPN, DARKO by horizon (the Dec 14 study covers tonight / next 3 games only).
3. Availability: injury return dates vs ESPN / our curves, star rest, missed-game rates through the season, late-season shutdowns.
4. 15-spot crunch: part-week returners pushing out full-week players (bug found 2026-11-16: ranking by level, not by points that week);
   which 15 count.
5. Start usage per manager (some teams leave starts unused).
6. Schedule: the 2 unscheduled NBA Cup games per team, back-to-backs, playoff-week schedules.
7. Uncertainty: refit weekly noise and season-long drift so stated odds are calibrated across checkpoints.

## Tommy's caveats (apply to every item)
- DRNK was only actively managed from 2026-27: DRNK's 2025-26 manager tendencies are not a guide to DRNK now (use this season / a league-active prior).
- New NBA draft-lottery rules should reduce late-season tank resting; past seasons' late shutdown rates are an upper bound; check this season Feb-Mar.

## Log
- 2026-11-16 stage 1 (`research/ros_accuracy_study.py` -> `data/ros_accuracy_stage1.txt`): error by horizon, persistence of manager factors,
  shrink choice by horizon, rest-of-season totals.
  Results (2025-26 replay, 12 teams, 17 full weeks):
  - Held-roster forecasts miss by 16.5% a week at 0 weeks ahead, rising slowly to ~21% at 10+ weeks; they run 4% high at 0 weeks and
    6-7% high at 7+ weeks (the held roster decays faster than churn replaces it; the league factor fixes most of it).
  - Manager factors persist strongly: correlation of a team's factor before the checkpoint with its later real/forecast is +0.6 to +0.9 from
    week 7 on (+0.35 at week 4: early factors are noisy).
  - Shrink: K=2 is best at every horizon (per-week miss 168/185/223 for 0-2/3-6/7+ weeks ahead) but K=5 (live) is within 1-5 points: keep.
    No horizon-specific shrink needed.
  - Rest-of-season totals: the live method cuts the miss from 13-19% to 8-12% and lifts rank correlation from 0.2-0.5 to 0.7-0.9 (from week 7 on).
  - DRNK caveat already holds in the live system: the factor uses 2026-27 weeks only.
  Open from stage 1: a small extra horizon decay (~0.25%/week) to test with item 7; the manager factor's within-season trend (does a team's
  edge grow or fade?) once 2026-27 has ~8 weeks.
- 2026-11-16 item 4 (roster spots): `build_schedule_plan.capped_days` (SPOT_DAY=1; off until approved). Day-aware: returners stay on IR until
  their first game back, then replace the weakest kept player. Cannot be replayed on 2025-26 (the replay drops IR players and has no return
  dates), so it rests on logic: it never projects a team lower than the weekly rule. Today: DRNK week 5 1,524 -> 1,648; rest-of-season
  totals +0 to +430 by team; playoff odds move 0-3 points.
- 2026-11-16 item 7 (uncertainty): `playoff_sim.simulate(walk={step, decay})` (off unless passed) and title_odds_study.py part D.
  * Misses grow with distance at the same point of the season (~0.6% further short per week ahead; not a late-season effect, so Tommy's
    lottery caveat does not touch it), and the part of a team's misses shared across weeks grows from ~160 to ~240 per 7 days (near -> far).
  * But scored on 2025-26's real outcomes (who made the playoffs, bye, champion), build-up drift and the distance shortfall both scored
    WORSE, and smaller drift (tau ~105) scored better than the live 165 early / 105 late. One season, 12 teams, one champion who was the
    favourite: outcome scores reward over-confidence in a chalky season, which is why the 11-07 study fit tau from the misses, not outcomes.
  * Verdict: can't tell yet. Keep the live settings; refit at the end of 2026-27 with two seasons of outcomes. Today's effect if adopted:
    DRNK title 23% -> 19%, favourites -1 to -4 points, long shots +1.
