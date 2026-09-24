# Fantasy Hub — Backlog

Pending features/improvements, not yet built. Approve one at a time.

## Automated daily refresh (DARKO-style)

Right now the whole pipeline is manual — nothing updates on its own as new
real games/data come in. `kalman_input.csv` (the real game-log data
everything is built on) is already stale as of 2026-09-22 (last refreshed
2026-09-18).

What a real daily refresh would need:
1. Re-pull fresh real game logs (`ingest/research/prep_kalman_input.py`
   already exists for this, would need to run against updated `nba_api`
   data).
2. Re-run `ingest/research/kalman_vor.py` (updates current players' real-
   time posteriors against new games) and
   `ingest/research/project_prospect_trajectory.py`.
3. Re-pull ESPN data (`ingest/research/pull_espn_positions.py`) — especially
   time-sensitive since injury status changes daily.
4. Rebuild the dashboard data (`ingest/research/build_hub_data.py`).

Would need real scheduling infra (a cron-style scheduled task) wired up to
run this sequence automatically, plus real error handling for missed/failed
pulls (e.g. a game postponed, an API hiccup) so a bad run doesn't silently
corrupt the live dashboard.

## Daily waiver/start-sit optimizer

A morning-open tool: tells Tommy what waiver moves and starting-lineup
changes to make that day to optimize points, specifically exploiting the
daily-lineup + limited-weekly-moves structure of ESPN fantasy basketball
(different from fantasy football's season-long "set and forget" lineups).
The core lever: some teams play more games than others in a given fantasy
week (schedule quirks), and there's a limited number of waiver
claims/adds ("games played" cap or similar -- exact mechanic TBD) that
constrain how aggressively this can be exploited.

Needs before building:
- **Research this league's actual real settings** (waiver claim limits,
  weekly acquisition cap, whatever the real "games played" constraint is)
  via the ESPN API config already used elsewhere in this project
  (`ingest/config.py`, league ID 600271905).
- **Research general streaming/waiver strategy** for daily-lineup fantasy
  basketball -- this is an established, well-covered strategy area, not
  something to invent from scratch.
- **Research Vegas-implied projections** as a methodology -- converting
  real sportsbook lines (game totals, spreads) into implied team
  pace/possessions and then into implied player-level stat projections.
  Tommy's read (plausible, worth verifying): more accurate than ESPN's own
  daily projections. Also an established technique, not novel -- real
  write-ups exist to draw from rather than deriving it blind.
- Would need a **real odds/lines data source** (pulling from sportsbooks)
  as a new real-time data dependency, on top of the existing NBA/ESPN
  pipelines.

Output: open it any morning, see recommended adds/drops and starting lineup
for that day, ranked by expected point gain.

## Breakout-candidate model -- v3 BUILT (2026-09-24)

Pipeline: `build_breakout_panel.py` -> `build_breakout_context.py` (situation features) ->
`pull_espn_adp.py` + `pull_coaches.py` (data pulls) -> `breakout_model.py` (writes
`data/breakout_candidates.csv` + `breakout_validation.json`) -> `build_hub_data.py` -> "Breakouts" tab.

v2 changes (from Tommy's feedback):
1. **All calibers.** Two ADP-independent models: Emerging (below the 32.4 line -> relevant;
   AUC 0.82) and Elevating (already good -> +6 jump, the "Jalen Johnson" case; AUC 0.67,
   much harder). Backtest note: Jalen Johnson's 2024 jump was only ranked #34 of 109 (16%).
2. **Situation.** Tested changed teams, usage/minutes vacated at the NEW team (first-game
   team of next season, so in-season trades can't leak), and head-coach changes. Team move +
   vacated usage help (small: ~+0.02-0.03 AUC); coach change did NOT (wrong-signed, no lift,
   coach data ~70% covered) -> excluded. Playstyle/scheme change is NOT captured.
3. **ADP benchmark.** Real ESPN ADP 2019-2025 (2025-26 has none) + live 2026-27. ADP and last
   year's production are near-independent signals; blending cuts error ~12% vs ADP alone.
   "Edge vs ADP" backtest (ADP-listed, 2020-23): top-10% edge picks beat ADP-implied by ~+7
   pts/g, 69% beat it by 5+ vs 27% base. Breakout-specific signals (minutes trend, late-season)
   add ~nothing beyond ADP + production + age -- the crowd already prices those.
   Caveat: edge is per-game; ADP also prices injury/games missed.

v3 additions (Tommy's follow-up list):
- Soft hit threshold: landing within 1 pt of the 32.4 line counts (Filipowski 32.39 = hit).
- Projected pts/g with 10-90% range (error 6.2 vs 6.8 for repeating last year).
- Availability: projected GP (off ~19 games vs ~21 naive) and a season-total edge column. Result:
  per-game edge picks did NOT beat ADP in season totals (-5 pts), total-edge picks +97 pts vs ~645 noise.
  Shown as additional info, not folded into topline.
- Bust model (AUC 0.70, top-10% dropped 6+ 55% vs 22% base): foundation for the buy-low/sell-high tab.
- "VOR + breakout" separate column. Accuracy test: folding breakout odds into projections cut error
  only 1.5% (1yr) / 0.2% (3yr) vs a simple proxy base -> NOT folded into core trajectories.
- Rookie model (separate, pre-NBA profile): draft slot alone gets AUC 0.92; the college profile adds
  nothing (0.87-0.89). ESPN ADP beat the model on ADP-listed rookies (error 6.7 vs 12.5), so no
  rookie edge vs ADP is shown -- P(relevant year 1) only.
- Preseason ledger: freeze_breakout_ledger.py (opener 2026-10-20), grade_breakout_ledger.py.
- Head-coach change re-tested: no lift, excluded.

Open ideas:
- Playstyle/scheme change (needs coach-level tendency data, pace/3PA history by coach).
- Availability-adjusted edge (price games missed) -- ties into injury-risk tiers.
- Score incoming rookies (needs Output B priors).
- Wire into the daily-refresh job (ADP and rosters change daily preseason).
- ESPN ADP is early (only ~157 listed players as of 9/24); re-pull closer to drafts.
- Sensitivity check of the +6 / 32.4 label thresholds.

## Buy-low / sell-high candidate tracker

A dedicated, persistently-updated dashboard tab listing current buy-low and
sell-high candidates for easy reference.

Exact calculation method still to be determined, but the natural approach
given what's already built: compare a player's REAL recent trailing
performance (the noisy, hot-or-cold read most other owners react to)
against this model's more stable underlying estimate of their true level
(the Kalman posterior / aging-curve trajectory already built for the main
dashboard). A player performing notably BELOW their model-implied true
level recently = buy-low (bought cheap while cold, real bounce-back
expected). Notably ABOVE = sell-high (riding an unsustainable hot streak,
sell before real regression hits). Needs a real, validated definition of
"notably" (a real statistical threshold, not a gut-feel cutoff) before
this is trustworthy enough to act on.

## Live draft assistant mode

A mode on the dashboard for use during a live draft: click a player to mark
them as drafted/taken, and the board updates to make it easy to see the
highest-VOR player still available among the undrafted pool -- so Tommy can
always see "best player left" at a glance without manually cross-checking
who's gone.

Likely needs:
- A toggle per player (drafted / not drafted), persisted in the browser
  (localStorage, same pattern as the other standalone apps in this
  workspace -- e.g. job-tracker.html) so state survives through a real
  multi-hour draft.
- Drafted players either hidden or visibly grayed out (not removed
  entirely -- still useful as reference for who's gone).
- Should cover the full player pool relevant to whatever draft it's used
  for (not just prospects -- likely wants both current players and rookies
  together, since this is a dynasty/keeper league context).
- A "reset draft" control to clear all marks and start fresh for the next
  draft.

## Breakout tab -- deferred follow-ups (agreed 2026-09-24)

- **League context (build once the season starts):** show who is rostered in Tommy's
  league and by which team, who's a free agent, and keeper cost / draft slot. Also
  consider this league's own draft history as an alternative "market price" next to
  ESPN ADP.
- **Competition at position / role opened up:** sharper version of the team-level
  "vacated usage" signal (depth chart / same-position minutes available). Likely a long
  testing process -- expect it to mainly help the weaker Elevating model.
- **In-season updating:** breakouts show up as role/minutes jumps in the first 10-20 games.
  Not needed yet; revisit alongside the daily-refresh + waiver optimizer items.
- **Qualitative / news signals** (camp reports, starter injuries, role announcements): Tommy
  wants these but no consistent, reliable source exists yet. Needs a source decision first.

## Keeper-count-aware "Asset value" -- v1 BUILT 2026-09-24 (separate column next to VOR)

League facts confirmed by Tommy: 12 teams; keepers cost nothing (only a roster slot).
Why: VOR rank order is identical at K=3/5/19 (Spearman 1.000) -- K only moves the bar. Asset value makes K change
who is valuable: next season always counts; future seasons only count where a player clears the KEEP CUTOFF (pts/g of
the (K x 12)-th best player; K=0 -> future worth 0), weighted by career-survival odds and the real historical outcome
spread (ceilings). Scripts: research/asset_value_v2.py (model), value_horizon_analysis.py + value_horizon_prototype.py
(diagnostics), ceiling_calibration_backtest.py, young_vet_bias_check.py, data/hashtag_dynasty_2026-09-24.csv (anchor).
Out-of-sample (2015-19): beats "current output persists" on error at every horizon; ranking at very high cutoffs is
worse than persistence (few stars, noisy).
KEY FINDING (bigger than asset value): our Kalman/Output-B TRAJECTORIES are too flat. Past top-3 picks produced +8 pts/g
more than projected by year 3 (mean 41.7 vs 33.4; 27% of top-10 picks hit 50+ pts/g, our trajectories give 0%).
Empirical forecasts sit 3-4 pts/g above the Kalman for players <=21 and 2-5 pts/g BELOW it for players 31+.
=> the aging-curve shape (steeper early growth, steeper late decline) needs a fix; VOR trajectories NOT changed yet.
Remaining gap vs dynasty crowd: young stars (Boozer ~#59 at K=5 vs crowd #15) -- unknown whether crowd is hyping.
Assumptions to revisit: delta=0.92/yr, replacement=22 pts/g, 6-year horizon, only players with >=20 GP / >=10 mpg last
season (460 of 590) get a value.

### Aging-shape work: RESULT and CORRECTION (2026-09-24)
Correction to the "trajectories are too flat" claim above: the "young veterans projected 3-4 pts/g too low" figure
came from grading only players who stuck around (survivorship / peeking at outcomes). Graded honestly (players the model
already expected to matter at the forecast date; fit on targets <=2019, tested 2020-25), a fully re-fit steeper curve was
WORSE for young players (bias -4 pts/g, RMSE +9%) and the ORIGINAL curve was roughly unbiased through age ~30.
What held up out-of-sample:
 1. LATE-CAREER DECLINE (age 31+): original curve forecast ~2.6 pts/g too high. Fix: keep original slope below 28, blend to
    the fitted age-dependent slope by 31 at 60% strength (aging_shape.py; best held-out error; small overall gain, 31+ error -2%).
    Wired into kalman_vor.py (year-0 partial-year drift + build_trajectory). LeBron/Durant/Curry VOR@K=3 roughly -40%.
 2. PROSPECT CEILINGS (no NBA data yet, top-15 picks): the Output-B-based trajectories undershoot real outcomes, growing
    over the first 3 years (pick 1: +4.5 rookie yr, +8.8 yr 2, ~+7.8 by yr 3-5; pick 8: +0.6/+4.4/+4.7). Leave-one-class-out:
    bias removed, error -13% at year 2 to -8% at year 5. Fixed in build_hub_data.py (prospect_calibration.py); full strength
    through pick 10, tapering to zero at pick 16. Existing NBA players' trajectories are unaffected.
Not fixed: VOR still stops counting at the first year below the opportunity-cost line, so a prospect who starts just under
it and climbs later (Peterson) can show VOR 0 -- Asset value exists for exactly that case.
Scripts: fit_aging_shape.py, aging_shape.py, prospect_calibration.py, ceiling_calibration_backtest.py.
