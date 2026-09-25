# Start/sit Phase 2 — plan (prepared 2026-09-24; features 1-4 BUILT 2026-09-25)

Phase 1 ("This week": today's lineup, adds, opponent) is live. Phase 2 answers the questions Phase 1 can't: **who should I hold, add or
target so the next several weeks (and the playoffs) score more?** This is the plan and what is already in place; nothing below is built
except the schedule data.

## What the backtests say Phase 2 should focus on (2025-26, this league's real weeks)
* **Lineup management is worth a lot, but mostly to weaker managers.** Replaying every team with the planner's lineups (same rosters,
  day-of availability, recency-weighted levels, cap logic): +102 pts/team-week on average (median +50); teams already managing well gained
  only +7 to +34, teams that left games on the bench gained +150 to +350. Cap-aware planning beat "start everyone" by +45 pts/week and
  changed the outcome in about a third of weeks. For DRNK: +80 pts/week, about one extra win over 18 matchups.
* **Adds help, but about half of what the page predicted.** With the older assumptions, suggested add/drop moves were predicted at
  about +93 pts/week and realized about +49 (positive in 70% of weeks with suggestions, +41 averaged over all weeks). The gap is
  free agents' recent form regressing toward average (we pick the ones who look best) and marginal players missing games.
  Calibration fixes are in section "Calibration" below.
* **Implication:** Phase 2 value is in *fewer, better-timed moves* and *schedule planning*, not more churn.

## Phase 2 features (in priority order)
1. **Schedule view** (data pulled, see below): games per NBA team for every fantasy week, light weeks, back-to-backs, and the three
   playoff weeks (weeks 20–22, Mar 8–28 under ESPN's assumed calendar; ESPN's 160-scoring-period season ends Mar 28). On the current
   schedule teams get 9–12 games across those three weeks (DAL, MEM, PHX 12; CLE 9). Page: a heat-map grid by team and week.
2. **Roster schedule value**: for each of your players, his games over the next 4 weeks and through the playoffs, and an
   "extra games vs an average team" number. Flags: players about to have a light week (stash or bench), players with a heavy
   playoff schedule (hold), and free agents to add *before* a light week for your stars.
3. **Playoff planner** (from ~week 17): projected points for weeks 20–22 for your roster and your likely opponents given games and
   the 40-game cap; ranks add/trade targets by playoff-week games (the ESPN strategy article calls game count in the finals
   more important than opponent). Uses the same daily lineup solver as Phase 1.
4. **Add budgeting across the week**: adds don't roll over (about 8 a week). Given the schedule, when in the week each add is worth the
   most (an add on Monday covers more days), and a hold-back rule when a better free agent tends to appear mid-week.
5. **Trade helper** (optional): value of a player to my team = points added across the remaining schedule + playoff weeks (not just
   dynasty asset), so a trade can be judged for both this season and the keeper decision.
6. **Reminders** (low priority; you open the page in the morning): tip-off times per slot so late swaps are obvious. Already
   partly available (the page has each game's tip-off in the refresh data).

## Data status
* `dashboard/nba_schedule.json` (new, refreshed daily by `pull_nba_schedule.py` once wired into the workflow): 1,206 of ~1,230 games
  for 2026-27. **Provisional:** ESPN's feed still shows NBA Cup knockout games as "TBD" and looks incomplete around Dec 7–13
  and late Feb, so weekly counts there are not trustworthy until the schedule is final. Re-pull weekly.
* ESPN's fantasy-week calendar for 2026-27 is inferred (6-day opener, Monday–Sunday weeks, one 14-day All-Star matchup, playoffs Mar 8–28);
  the real one appears in season and must be read from ESPN (Phase 1 already flags this).
* Free-agent projections and positions come from ESPN (already in Phase 1).

## Calibration fixes before Phase 2 builds on Phase 1
* Availability is not 94% for every future game: measured play rate for currently-healthy players in a *later* scheduled game is 85%
  for rotation players, 81% at 15–20 pts/g and 61% below 15 (`availability_by_level.py`). 94% stays right for today's ACTIVE-listed players.
* Free-agent levels are now shrunk 40% toward replacement before ranking adds (done; see the backtest results in
  RESEARCH_start_sit_waivers.md when finished).
* Injury statuses: done. Measured rates by the NBA official designation now drive today's availability (see RESEARCH file).

## Validation plan
Every Phase 2 feature gets a replay on 2025-26 like the Phase 1 backtests (`backtest_weekly_planner.py`) before it is trusted, and a
live shadow log (`projection_log.csv`) once the season starts.


## Build status (2026-09-25)
* Built: (1) Schedule tab heat map, (2) roster schedule value (games over the next 4 weeks and the playoffs, +/- vs average, flags),
  (3) playoff planner (projected points weeks 20-22 for every team, best free-agent adds and trade targets by playoff points, streamers),
  (4) add timing on the This week tab (net gain of each top add if made on each remaining day, with a budget rule). Code:
  `ingest/research/build_schedule_plan.py` (schedule_plan.json) and the add-timing block in `build_week_plan.py` (`by_day`, `fa_pool`).
  Not yet validated by a 2025-26 replay (the plan's rule); the shadow log will score it in season.
* Assumptions to verify in season: the fantasy-week calendar, "OUT now" players absent 2 weeks then 60% availability, whether adds of
  recently dropped players are instant (waivers process Sundays), the reserve-one-add heuristic (not measured).
* Injury-timing fix: `.github/workflows/refresh-gameday.yml` re-runs rosters + weekly plan + schedule plan at 19:07 and 22:07 UTC
  (about 3pm and 6pm ET in summer time, 2pm and 5pm ET in winter), after the NBA's official injury report (due ~5pm ET). Known gap: it cannot
  help 12-3pm tips, and the plan for "today" still lists players whose game has already started.
* DFS projections (`probe_dfs_projections.py`): DraftKings is the closest format (r = 0.993 per game vs league points; league = 1.206 x DK).
  Cannot be verified until slates exist (Oct 20). On that day run the probe, read the saved HTML, write the parser, and add a `dk_proj` column to
  `projection_log.csv` so it can be scored against ESPN + recent form before it drives any decision.

## Backtest results (2025-26, this league; `ingest/research/backtest_phase2.py`, run 2026-09-26)
* **A. Schedule-aware forecasts** (roster fixed at the start of an anchor week, forecast a later week; 168/156/144/120 team-weeks for 1/2/4/6 weeks ahead):
  MAE in points per team-week (1 wk / 4 wk ahead): schedule-aware solver 207 / 212; blind (no schedule) 334 / 318; "same as the anchor week's real points" 246 / 316;
  hybrid (anchor week's real points x solver ratio target/anchor) 178 / 191. Explaining which weeks are big or small for a team (within-team correlation,
  1 wk ahead): solver 0.71, hybrid 0.65, blind 0.02, persistence 0.21. The solver over-forecasts by about 77-98 points a week (5-7%).
  => the schedule is what explains a team's week-to-week swings; in season, use the hybrid for totals.
* **B. Playoff-week ranking** (12 teams, one season): from week 16 / 19 the rank correlation with real playoff points was solver 0.46 / 0.22,
  persistence 0.70 / 0.72, hybrid 0.73 / 0.75. The fixed-roster forecast is worse than "how the team has actually been scoring" for ranking teams
  (it misses manager quality and roster changes). Use the playoff projected-points table as a schedule guide, not a power ranking.
* **C. Add timing** (152 team-weeks with a suggested add, day-by-day replay with real box scores): predicted vs realized gain by day of the week is well
  calibrated (day 1: 73 vs 73; day 4: 35 vs 37; day 7: 14 vs 17); across (add, day) pairs r = 0.54. Waiting one day costs about 12 points realized
  (13 predicted). The model's timing verdict never beat "add on day 1" (+72.6 both), the oracle best day was +76.4. => "add now" is nearly always right;
  the table's value is showing the cost of waiting. The reserve-one-add rule cannot be tested with this data.
* **D. Games in the week do not change availability**: rotation players' play rate was 82.7% / 83.7% / 80.8% in 2 / 3 / 4-game weeks, so games count is
  linear, which is what the extra-games (+/-) columns assume.
* Not tested: streamers before a light week, the trade-target ranking by playoff games, week-1 calendar assumptions.
