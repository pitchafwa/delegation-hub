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

**Research done 2026-09-24: see RESEARCH_start_sit_waivers.md** (league rules verified, what decides weeks, strategy ranking, proposed build). **Phase 1 BUILT 2026-09-24:** "This week" tab (dashboard) fed by ingest/research/build_week_plan.py -> dashboard/week_plan.json, refreshed daily by .github/workflows/refresh-league.yml. Cap-aware day-by-day lineup plan (DP over starts-so-far; goal is points not games), suggested add/drop sequence (top-120 dynasty assets never dropped), opponent projection/win chance. NOT yet backtested against 2025-26 weeks; assumptions to verify in week 1: matchup dates, add limit scaling, waiver behavior, DAY_TO_DAY=55% play chance, model+ESPN 50/50 level blend. Phase 2 (schedule-aware roster planning, lock-time reminders, playoff planner) not started.

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

### Asset value prospect model now uses pre-NBA talent (2026-09-24)
Prompted by Tommy questioning Boozer at ~#59: the prospect model used only draft slot + age, so an elite (100th-pct) profile was
treated like any #3 pick. Among past top-5 picks, talent 95+ averaged 44 pts/g in years 3-5 vs 35 for <80 (n=8 vs 32; weak once
pick is controlled, t=0.7), but adding talent_pctile lowers held-out (leave-one-class-out) error 2-3% at every horizon -> adopted.
Boozer K=5 rank #59 -> #48 (dynasty crowd #15). Remaining gap = model still treats him as a top-3 pick with a good profile, not a
consensus generational talent; the crowd is pricing scouting/hype we can't measure. Any further move toward the crowd would be a
judgment call, not a validated fix.

### UNIFIED projection engine (2026-09-24) -- Tommy's Boozer / Rollins / Fears inconsistency
Problem: Asset value ran on its own models while the trajectory/VOR columns used the Kalman path, so the two disagreed
(Fears and Rollins ranked above Boozer while showing worse projections). Fix: ONE path per player feeds both.
Held-out tests (leak-free; grading set = players rated relevant at the forecast date):
  Veterans (engine_blend_test.py): blend beats either alone at every horizon (RMSE h1 6.66 Kalman / 6.39 ridge / 6.32 blend; h4 10.41/10.17/10.01).
    Kalman runs ~4-5 pts/g LOW for breakout-jump players (+10 pts/g last year; the Rollins case) and for age<=21 (+3.9); ridge is ~unbiased.
    Rule: 75% empirical / 25% Kalman next season, 50/50 after. (Extra rules for jump/young players added nothing on RMSE.)
  Prospects (prospect_engine_test.py, leave-one-class-out incl. refitting Output B): picks 1-15 -- calibrated Output B path and the
    empirical pick/age/talent model are about equally accurate, blend slightly best; picks 16-30 -- the Output-B-based path is 4-9 pts/g
    too optimistic by yr 3-5, the empirical model is unbiased. Rule: 40% Output B path for picks <=10 tapering to 0 at 16+, else empirical.
Result @ K=3/5: Boozer #41/#39, Fears #58/#53, Rollins #67/#64 (at 0 keepers Rollins/Fears are ahead of Boozer -- honest: they project better NEXT season).
Agreement with anchors improved (dynasty 0.827 @K=5; ESPN ADP 0.865 @K=0).
SCOUTING LEVER: ingest/research/prospect_overrides.json {"Name": {"equivalent_pick": n, "note": "..."}} values a prospect as a different
draft slot in every model (transparent, off by default). Test: Boozer as a #1-pick-equivalent -> #23 at K=5 (path 36 -> 50). The data cannot
justify more; the crowd's #15 requires believing he is a tier above any historical #1 pick.
Known VOR quirk: VOR stops at the first year below the opportunity-cost line, so Boozer's VOR@K=3 is 0 (yr-1 31.8 vs 32.4 line) despite
a rising path -- Asset value handles that case; a "stash-adjusted VOR" (net early sub-replacement years against later surplus) is an option.

### Upside tools (ceiling scenarios / star odds / risk-appetite tilt): BUILT THEN REVERTED at Tommy's request (2026-09-24)
Judged as adding confusion without a clear answer. (Findings kept: asset model's outcome spread is calibrated to history; a 48 pts/g rookie year is
~97th pct for a top-3 pick; only 3 of 36 top-3 picks did it.)
### What DID change instead: the #1 overall pick is its own tier
Held-out (leave-one-class-out): #1 picks were under-projected by ~4-8 pts/g in both the empirical model and the calibrated Output-B path
(actual avg 39/47/51 at rookie yr/yr 3/yr 5 vs model 34/41/43; median best season of the 13 #1 picks 2010-2025 = 51), while #2-3 were slightly
over-projected. Adding a #1-pick term cuts #1-pick bias to ~0, improves top-3 RMSE (yr 5: 12.0 -> 11.5), leaves picks 4+ untouched.
Dybantsa (#1, 2026): projected path 38 -> 53 by 2033 (was ~30 -> 45.5), asset rank #32 -> #21 at K=5. Boozer/Peterson (#3/#2) unchanged.

### Ceiling VOR / Ceiling asset value (2026-09-24, Tommy's request)
"If he hits a 90th-percentile career": the expected career shape given TOTAL career value (discounted pts/g above replacement, independent of keeper
count) lands at the 90th percentile of outcomes for players like him. Method: deviations from the projected path are jointly normal across years,
using each age-group / draft-tier's own spread and the real pooled year-to-year correlation of deviations (a hit persists); E[deviation | career score
at its 90th pct] = z * Cov*w / sqrt(w'Cov w). Assumes he stays in the league (retirement/flop risk is priced in the base asset value, not here).
First attempt (sampling actual analog careers) was too jagged with ~30 comparable top-3 picks (injury years, flameouts) and was replaced.
Shown as columns + a dashed green line on the player chart. Ceiling asset value is the path treated as realized (no extra spread). Examples (K=5):
Dybantsa VOR 118 / asset 58 -> ceiling 214 / 115; Boozer path 40->60; Peterson 39->58. Scripts: asset_value_v2.py (ceiling_paths), data/ceiling_paths.csv.

### Rookie team situation ("available usage") -- Tommy's idea (2026-09-24)
Measure (team_context.py): production that LEFT the team minus veterans who ARRIVED, every rotation player (>=10 mpg) counted at his HEALTHY per-game fantasy level.
The availability-weighted version (fpg x GP share) was worse: injured stars look small (Morant 46 fpg x 20 gp counted as 11; A. Davis as 12). Same-position
competition was tested and did NOT beat the team-wide measure; crowd (returning+arriving) and star-returning measures were weaker than "open".
Held-out (leave-one-draft-class-out): lottery picks' rookie-year RMSE 8.85 -> 7.93 (-10%); all picks 7.46 -> 7.02. Effect per +1 sd (sd = 52 pts/g of production):
+3.8 pts/g rookie yr, +2.5 yr 2, +1.9 yr 3, +1.5 yr 4, +1.0 yr 5 for top-15 picks (fades; out-of-sample gain mainly yrs 1-2).
Model: added to the empirical prospect model for years 1-3 (open + open x top-15), capped at +/-2 sd; the effect flows to both engines in the blend.
Live 2026-27 (z): MEM +1.0 (+79), CHI +0.3, BKN +0.3, WAS -1.1 (-25), UTA +3.4, SAC +2.9. Boozer +4.8/+3.5/+1.3 pts/g (yr 1/2/3), Dybantsa -2.9/-2.2/-1.3.
Result K=5: Boozer #44, Dybantsa #26; at 0 keepers Boozer #79 vs Dybantsa #95 (redraft favours Boozer; #1-pick tier favours Dybantsa long-run).
Payroll: Basketball-Reference committed 2026-27 payroll (prospect... payroll_2026_27.csv) puts MEM $161M (29th), BKN/CHI $161.5M, DET lowest $153M; WAS $192.8M (23rd,
NOT top 5). Uncorrelated with the production measure across the 30 teams (rho -0.08). Cannot test as a predictor: no historical payroll (Spotrac 403, HoopsHype 402).
Open: a historical payroll/cap-space series by team-season would let us test Tommy's "invested in others" idea properly. Incoming-class teams come from
ingest/research/prospect_teams_2026.csv (transcribed from hashtagbasketball.com/keeper, 2026-09-24); 23 of 61 prospects have one.

### Variant: team-situation boost as a year-1 baseline (carry-through) -- live at ?variant=carry (2026-09-24, Tommy's request to compare)
CTX_MODE=carry (asset_value_v2.py / build_hub_data.py): opportunity applied to year 1 only; the boosted year-1 level shifts every later season equally (no fade).
Default site keeps the measured fade (+3.8/+2.5/+1.9/+1.5/+1.0 pts/g per sd in yrs 1-5). Data says carry-through overstates later years (the effect decays historically).
K=5 ranks fading -> carry: Boozer #44 -> #30, Dybantsa #26 -> #35, Peterson #39 -> #20, Acuff #95 -> #61. Year-1-to-peak growth: Dybantsa +20 -> +17, Boozer +11 -> +16, Peterson +6 -> +14.

### 10-season horizon, elite-young scouting prior, market rank (2026-09-24, Tommy's go-ahead)
- HORIZON: Kalman/prospect trajectories, unified paths, asset value and ceiling paths all run 10 seasons (was 6-7); yearly discount 0.95 (was 0.92);
  seasons 7-10 come from per-season ridge/logistic models trained on real outcomes (fewer rows), blended like earlier seasons, lightly smoothed (seasons 6+).
  VOR now sums up to 10 seasons. K=5: Boozer #44 -> #24, Dybantsa #26 -> #14, Wilson #83 -> #42, Harper #85 -> #50.
- Corrected research target (Tommy): dynasty ranks are judged by whether value HOLDS. Young players' market ranks held/gained (2-yr median -9 vs +16 for established);
  expert_rank_drift_check.py. But: uniform youth premium (fails cross-source), per-season convex tilt (tiny), convex CAREER value with persistent variance
  (year-to-year deviation correlation 0.74; tiny effect, no agreement gain), and a multiplier premium on top-5 picks all failed or behaved inconsistently across K.
- SCOUTING PRIOR (calibration, not a validated production effect): +3 pts/g every season for players with draft pick <= 5 and age <= 21 (ELITE_UPLIFT in asset_value_v2.py).
  After the horizon fix this group averaged rank 44 (K=19) vs Hashtag 32 / RotoWire 40; +3 puts it at 39 (K=5) / 35 (K=19), between the two outlets, raising agreement with
  Hashtag (0.839 -> 0.845) at some cost vs RotoWire (0.701 -> 0.635). elite_talent_shift_test.py. Disclosed in the site legend. Override lever unchanged.
- MARKET RANK column (Hashtag Basketball crowdsourced dynasty rankings as of 2026-09-24, data/hashtag_dynasty_2026-09-24.csv) with our-minus-market gap; refresh by re-pasting the list.
- Carry-through variant page (?variant=carry) removed.

### Scouting prior moved to pricing only; market-gap sign flipped (2026-09-24, Tommy's request)
The +3 pts/g prior for young top-5 picks now lives ONLY in asset value and ceiling asset value (EA_v/EA_p and C_price in asset_value_v2.py). Projected pts/g,
peak year, trajectory chart, VOR and ceiling VOR/path are the unadjusted model output (Dybantsa year 1 = 33, Boozer 34). Asset ranks unchanged.
Market gap shown as market rank minus our asset rank: + = we rank him HIGHER than the market (Dybantsa +11, Peterson +9), - = lower (Boozer -9, Wilson -21, Harper -37).
