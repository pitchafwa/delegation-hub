# Start/sit Phase 2 — plan (prepared 2026-09-24)

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
* Free-agent levels need shrinking toward average before ranking adds (backtest in progress; result recorded in the results section of
  RESEARCH_start_sit_waivers.md when finished).
* Injury statuses: replace the guessed 55% for day-to-day with measured rates by the NBA's official designations (see RESEARCH file).

## Validation plan
Every Phase 2 feature gets a replay on 2025-26 like the Phase 1 backtests (`backtest_weekly_planner.py`) before it is trusted, and a
live shadow log (`projection_log.csv`) once the season starts.
