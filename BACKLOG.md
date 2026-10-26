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


## Added 2026-09-24 (end of session)
- **Week-1 checks for the This week tab:** ESPN per-day projections (lineup view / box_scores projected_points), real matchup dates, add-limit scaling, Sunday-only waiver behavior (Add vs Claim), day-to-day play chance (55% is a guess).
- **Backtest the weekly planner** on 2025-26 weeks (what would its lineups/adds have scored vs what teams scored).
- **Per-game projection source:** none found free + permitted (FantasyPros needs an account/terms bar republishing; DFF is DK points only; DARKO daily page not live; The Odds API props cost $30/mo, declined). Shadow-log (projection_log.csv) compares sources; revisit DARKO when live. Interim: ESPN blended with last 15 games.
- **Phase 2 of start/sit:** schedule-aware roster planning, lock-time/late-swap reminders, playoff-week planner.
- **Daily refresh of the valuation model itself** (needs local game logs; not workable in GitHub Actions as-is). Named **DELCO** (2026-09-27).
- **Draft history as a second "market price"** next to ESPN ADP; real keeper flags once ESPN records them.

## Roadmap from the strategy review (added 2026-09-26; full tactic list in STRATEGY_PLAYBOOK.md)

All approved by Tommy for the backlog. Priority order agreed 2026-09-26. Also do in week 1 (not a feature): verify ESPN's real matchup calendar (a second 14-day week during the NBA Cup? playoff dates?), waiver/add behavior, per-day projections.

### Tier 1: build first
1. **Keeper decision tool** (deadline Sun 9/27 5pm ET). Compare candidate SETS of keepers side by side for Tommy's roster: this season's lineup points with position/UTIL fit and replacement fill, next year's top-5 keeper assets, injury tier and age, and the draft consequence (chance each player he does NOT keep is still there at his picks, from the availability simulation, and what he would take instead). The Board ranks players one at a time; this ranks decisions.
2. **Live matchup tracker.** Live score vs the opponent (ESPN box scores), remaining games for both sides, win probability, swing games, games-cap status, and aggression advice (underdog = variance, favourite = floor). Also feeds the alerts.
3. **Injured-player advisor.** For each injured player on Tommy's roster or a target list: stash / drop / trade-for by expected return date (absence-length curve), value tier, IR availability, and the minutes ramp after return (needs a small study on post-return minutes from box scores).

### Tier 2: next
4. **Streaming-slot manager.** Multi-day chain of streamers within the add budget (Mon-Tue guy, then Wed-Thu, then weekend), keeping one open bench spot; extends the add-timing table.
5. **Playoff-odds and seed simulator.** Simulate the rest of the season from projected weekly points (games cap included) for all 12 teams; playoff odds, seed, points-for tiebreak, and "what does this add/trade do to my odds".
6. **"Why is he hot" split + points-per-minute and minutes-trend columns** on free-agent cards: separate minutes change, teammate-out boost (usage flow) and shooting luck from a real role change.

### Tier 3: backburner
7. **Team-motivation flags.** Late-season tanking and seed-locked teams (from standings) feeding star rest risk and youth-streamer boosts. Needed by February, not before.
8. **Trade extras.** Draft-pick valuation in the trade finder (Tommy trades picks), injured-star buy list, trade-deadline calendar.
9. **Small items bundle.** Late-swap planner (leave slots open for late games), two-way/call-up flag on free agents, just-dropped watch with waiver clear time, position-eligibility gain tracker, opponent scouting.

### Standing item: accountability ledger
Log every plan suggestion daily from opening night (cheap, do in week 1) so a ledger that grades suggestions and our availability/usage-flow numbers can be built later. The UI can wait; the logging cannot.

### Also still open from earlier
Usage-flow add/drop replay on real 2026-27 weeks; injury-recurrence study extended to 14 seasons using the box-score DNP reasons; DFS projection parser (Oct 20); Tatum/age-28+ aging check; unvalued players; Phase 2 replays on real weeks.

### Added 2026-09-26 (live tracker follow-up)
* **Optional add-on: matchup alerts** - send an ntfy alert when the live win probability crosses 25% or 75% (deferred by Tommy; the tracker is on-page only for now).

### Refinement watch-list (added 2026-10-04, after the draft; revisit as real games arrive)
Things noticed while reviewing the Week 1 plan and the draft tools. None are bugs; each is a place the model may be off and should be checked against real results before changing anything. Do NOT change the forecast engine on a hunch: its calibration (the "forecasts run ~5% high" note, FA shrink backtest) was measured on the current accounting.

1. **Bench cover is ignored by the points forecast and the add/drop search.** `plan_team()`/`day_options()` in `build_week_plan.py` counts each starter at p(plays) and gives bench players zero credit, so on days with spare players (Week 1: Tue/Thu have 13 players for 10 slots) a missing starter is really replaced. Display fix shipped 2026-10-04 (`starts_with_cover()`, `starts_cover` per day, label now "about 33 of 34.3"); the engine itself is unchanged. Consequences to check: (a) expected points are probably a bit low for deep-bench teams (rough size: +~90 pts/week for Tommy's Week 1, maybe offset by the calibration); (b) dropping a bench player looks cheaper than it is, since his cover value counts as 0. TO DO after 2-3 weeks of ledger data: compare forecast vs actual points split by "spare players that day"; if forecast error correlates with spare players, put replacement value into `day_options` (expected points = best lineup after random absences) and recalibrate.
2. **`future_cost` assumes you are stuck with the add for 6 weeks.** In `search_moves()`, an add that is worse than the dropped player is charged (gap x 3.3 games x 6 weeks), but a pure light-day streamer would be swapped out next week, so the real cost is closer to (dropped player minus the best free agent you could re-stream). Week 1 example: Mara/Caruso fill the empty Tue/Thu slots for about +37 pts but are charged -100 to -135; with a re-stream cost it is about break-even (~38). Not worth a change unless streaming-heavy weeks keep showing "no adds" while real streamers win. Related: only 4 of Tommy's 16 players are droppable (12 protected), which limits every search.
3. **Julian Reese (NOP) rates 35.9 raw, 30.3 shrunk** and is the one free agent above all four droppable players. Re-check after Week 1: is the role real (starter minutes, teammate injuries) or a projection artifact? He has no dynasty valuation yet.
4. **Draft availability model, post-draft grade.** The sim (`leagueKeys`/`simulateAvail` in index.html, tested by `draft_avail_backtest.py`) was rebuilt on 2026-10-04 and fit to 2021-26 history. Rookie pricing came from only ~35 rookies (log error ~0.8, i.e. off by about 2x). After the 2026 draft: (a) add this year's real draft to `draft_history_2021_2026.csv` (rerun `build_team_profiles.py`), (b) reconstruct our pre-draft expected pick numbers from git history (`dashboard/draft_data.json` + `hub_data.json` as committed Oct 4) and compare with actual picks, especially rookies and the 4 rookies priced in the top 12, (c) refit the rookie curve (`1.215 + 1.157*ln(NBA pick)` in `leagueKeys`) and the noise width (4 + 0.25*seen), (d) decide whether owner tendencies (still hand-set weights 3.0/2.0/2.0, never validated) help or should go. Note this year had only 3 keeper rounds vs 6-8 before, which may shift rookie prices later than the curve says.
5. **Held by Tommy (2026-09-30): live start-rate tracker + `start_bench_model.json` healthy-roster adjustment** into `level_on()`. Wait for real season lineup data. Related: `pull_lineups.py` (RotoWire lineup scrape) is an unvalidated parser; spot-check on a real game day, then add to the game-day refresh. Free tier is today-only, so start news is only known same day.
6. **FIXED 2026-10-10** (was: hub team vs ESPN team mismatch, 190 of 360 players). Root cause: a stale `SEASON=2026` line in the local `.env` made `pull_espn_positions.py` read LAST season's ESPN pool (teams, injury statuses, position eligibility), and the hub took current players' team from their last box-score game. Fixed: `.env` override removed, `config.SEASON` now defaults to the current season, hub `team` = ESPN's current team (old value kept as `team_last`, unsigned free agents flagged `unsigned`). Remaining differences are only unsigned free agents. (Original note: LeBron: hub LAL, ESPN PHL). The Draft tab's G1 column follows ESPN's team; the Team column follows the hub. Decide whether the hub team should follow ESPN's current team (matters for schedule/matchup features once games start).
7. **DONE 2026-10-10 (yearly dates are derived):** `research/season_calendar.py` derives the opener (first game in `nba_schedule.json`), the matchup lengths (opener through the first Sunday, Mon-Sun weeks, the week at the start of the All-Star break = longest no-game run merged with the next into the 14-day matchup), playoff weeks (ESPN playoffTeamCount) and last week; it checks the total against ESPN's finalScoringPeriod and VERIFIES every regular-season matchup boundary against ESPN's own scoreboard, then writes `dashboard/season_calendar.json` (`verified: true`, which also turns off the 'dates are assumed' banner). Used by build_week_plan, build_schedule_plan, build_redraft, build_draft_data, build_hub_data (also season id and ledger file names), alerts (active-from), refresh_all (opener, opening tip, ledger file, ADP season), freeze_breakout_ledger, kalman_vor, asset_value_v2, build_team_profiles; the page's year text, G1 header and keeper-deadline text are derived too. Next season nothing needs typing; if ESPN's matchup structure changes, the verification step reports the mismatch and the banner returns.
8. **Player news (shipped 2026-10-04):** RotoWire items only via ESPN's fantasy news feed (`pull_player_news.py`, game-day workflow + local refresh). If it stops updating or returns empty, check the feed shape first; other sources considered and not built: beat-writer X accounts (no free API), Reddit r/nba, Bluesky lists.
9. **ESPN roster snapshot guard.** A transient bad snapshot (5-19 players/team, 2026-10-01) was blocked by a draft-status check in `pull_league_rosters.py`. Watch automated refresh runs for any recurrence once the season starts (the guard only applies pre-draft).
10. **cron-job.org alerts timeout (2026-10-03):** one `Timeout` on the alerts.yml dispatch, all later runs fine. If it repeats, raise the job's timeout setting and check GitHub API status before debugging the workflow.

### Implemented (marked 2026-10-08)
* **Late-scratch watch** (built 2026-10-04, timer live 2026-10-08): per-game alert when a starter is ruled out, names the swap (bench / slide-over chain / free agent), 35-minute reminder. `alerts.py lineup` + cron-job.org `lineup` job (every 5 min, 10:00-23:55 ET). Still to judge on the first real scratch (see OPENING_NIGHT_CHECKLIST 9b).
* **Live draft assistant mode**, **player news on cards**, **G1 column**, **draft availability model v2** (shipped earlier; the sections above are historical). The **Automated daily refresh** exists (GitHub workflows + local `FantasyHubDailyRefresh`).

### Set-lineup-for-the-matchup tool (added 2026-10-08, Tommy's request; SKELETON BUILT)
One action that sets the lineup for every remaining day of the current matchup (Tommy wants this, NOT a daily auto-updater; he keeps control). `ingest/research/set_lineup.py` builds `dashboard/lineup_preview.json` (read-only: plan starters per day vs ESPN's current lineup for that scoring period; scoring period 1 = 2026-10-20) and the This week tab shows it in a "Set lineup in ESPN" panel with a disabled button and an example view (`--demo` -> `lineup_preview_example.json`). Refreshed by the game-day workflow. First real finding: ESPN already carries the lineup forward so that every day matches the plan this week (0 moves); the tool matters in weeks where the games cap makes the plan bench players.
TO DO: (1) Tommy captures one real ESPN lineup move (browser network request: endpoint, body, slot ids) so `apply()` can be written; fall back to browser automation if the direct request fails. (2) Decide where Apply lives: local server (button on localhost, cookies never leave the PC; like `draft_live.py`) vs GitHub-workflow dispatch from the site with a GitHub token Tommy pastes once into his own browser (works from phone; token is the 'password') vs hosted worker. (3) Apply rules: moves only (never add/drop), never touch locked players, snapshot first for undo, apply exactly the previewed hash, ledger line per change, dry-run default. (4) Preview currently uses the plan with NO adds; refresh the plan after making suggested adds.

**Update 2026-10-09 (Tommy chose option 2: GitHub workflow button, works from his phone).** Built: `.github/workflows/set-lineup.yml` (workflow_dispatch `mode` preview|apply + `hash`; holds ESPN_S2/SWID as secrets; publishes `lineup_preview.json`, `lineup_result.json` and snapshots to the throwaway `lineup-data` branch), `set_lineup.py --mode apply --hash H --out DIR` (refuses unless the fresh preview hash equals the reviewed one; snapshot before writing; stops at the first failed day; `APPLY_READY=False` so it never writes yet), and the site panel (token saved only in the browser's localStorage, Refresh preview / Set my lineup buttons, status polling of `lineup_result.json`). Note: Tommy already set this week's lineups by hand to match the plan, which is why the preview shows 0 moves (my earlier guess that ESPN carries lineups forward was wrong). STILL TO DO: capture one real ESPN lineup move, implement `write_lineup()`, test on a real move, then flip `APPLY_READY`; write `--mode undo` from a snapshot.

**Capture DONE 2026-10-09.** ESPN's lineup write is `POST https://lm-api-writes.fantasy.espn.com/apis/v3/games/fba/seasons/2027/segments/0/leagues/600271905/transactions/` with body `{"isLeagueManager":false,"teamId":12,"type":"ROSTER","memberId":"<SWID>","scoringPeriodId":N,"executionType":"EXECUTE","items":[{"playerId":..,"type":"LINEUP","fromLineupSlotId":..,"toLineupSlotId":..}, ...]}` (slot ids: PG0 SG1 SF2 PF3 C4 G5 F6 UT11 BE12 IR13; memberId equals the SWID cookie; auth is the espn_s2 + SWID cookies; scoring period 1 = 2026-10-20). `write_lineup()` in `set_lineup.py` is implemented from it and re-reads ESPN afterwards to verify every moved player. `APPLY_READY` is still False until the live self-test passes: `python research/set_lineup.py --selftest --day 2026-10-20 --write` (swaps two NON-playing players and swaps them back; run only with Tommy's OK). Open questions the self-test answers: do cookies alone authenticate (no extra headers), and does ESPN accept a multi-player item list in one transaction. Then: flip APPLY_READY, write `--mode undo`.

**APPLY SWITCHED ON 2026-10-09.** Live self-test passed (swap + revert verified in ESPN with cookies alone), and a full `do_apply` -> `do_undo` round trip on the same harmless pair restored the lineup exactly. `APPLY_READY=True`; `--mode undo` reverses the last apply (kept in `lineup_last_apply.json`, carried between workflow runs via the lineup-data branch). Remaining to watch: (1) first real apply with several players per day (the multi-item transaction was only tested with 2 items; failures are verified and reported, never silent), (2) bench-to-empty-slot moves (unbalanced) untested, (3) the very first end-to-end run through the site button + token (Tommy to press Refresh preview).

**SET-LINEUP TOOL: DONE AND VERIFIED 2026-10-09.** Stress test by Tommy: scrambled the week across several days; the site button applied 14 moves (up to 8 players in one day's transaction) and the follow-up preview showed 0 moves. Lessons recorded: ESPN lineup writes use `type: ROSTER` (with memberId = SWID) only for the CURRENT scoring period and `type: FUTURE_ROSTER` (no memberId) for every later day (the first attempt got HTTP 409 'can only be executed in the current scoring period'); cookies alone authenticate; multi-player transactions are accepted; every write is re-read and verified. Workflow `set-lineup.yml` (preview | apply | undo) + panel on This week + token held only in the browser. Untested still: moving a player into an EMPTY slot with nothing going the other way; in-season behavior once today's period is no longer period 1 (code compares against `lg.scoringPeriodId`); token expiry (re-create the fine-grained token and paste it again). Possible later: an optional 'plan after my suggested adds' preview, and running the preview after every plan refresh with a phone notification when moves are pending.

### STREAM MANAGER: BUILT 2026-10-10 (replaces the greedy add sequence for Tommy's team; other teams keep `search_moves`)
`stream_search()` in `build_week_plan.py` plans the week's adds day by day as rentals: every day a free agent plays is a possible add day, he is held through his last game that week, a new add can replace an EARLIER streamer (so one slot rotates inside the add budget), the 15-man limit is checked every day, waiver players only count from their clear date, and dropping a real contributor is charged only the gap to the 3rd-best true free agent (x games x weeks left), not six weeks of the add's own level. Output keeps the old `sequence` schema plus `on`, `hold_through`, `release`. The page feeds it into the existing 'The plan across the week' box and Suggested moves table (no new section); phone digests/alerts understand `on`. Test hooks: `STREAM_DEBUG=1` prints each round's best candidate; `STREAM_TEST_WEAKEN="Allen,Bey"` makes those rostered players 14 pts worse to exercise chains. Week 1 result for Tommy's real roster: NO move clears the bar (best was Barlow for Allen: +48 this week, -105 future cost). With four junk droppables it chains 5 adds (Wolf replaces Wade midweek). To grade after real weeks: do suggested chains beat the old one-shot adds? Not yet built: Sunday lookahead into next week's first days, multi-streamer-per-slot beyond greedy, using the live `adds_used` mid-week.

### Implemented (marked 2026-10-10)
* **Streaming-slot manager** (Tier 2 #4) - built; see the STREAM MANAGER entry above. Follow-ups still open: Sunday lookahead into next matchup's first days; multi-streamer search beyond greedy; grade chains vs one-shot adds after real weeks.
* **Hub team vs ESPN team mismatch** - fixed (stale `.env` SEASON override; ESPN positions/teams/injuries now current).
* **Waiver rules** - confirmed; planner/alerts/cards distinguish waiver players from free agents; pickup alert.

**SUNDAY LOOKAHEAD: BUILT 2026-10-10 (stream manager follow-up).** On the last two days of a matchup (`LOOK_DAYS` in `build_week_plan.py`, 3 days of the next matchup, scored with a fresh start count and the next matchup's cap via `plan_team(days_=, cap_=)`), `stream_search` may add players whose value is next matchup's Mon-Wed: unspent adds EXPIRE when the matchup ends, so those adds are free. Steps carry `games_next` and `next_gain`; the page says 'lines up next week: Mon Wed (+N there)' and adds a footer note; the digest says 'also lines up next week's first days'. The FA pool and the scoreboard fetch include the lookahead days. Verified only with a simulated Sunday (`PLAN_TODAY=2026-10-25` test hook plus `STREAM_TEST_WEAKEN`): Nesmith +25 this week / +28 next week, etc.; a normal day is unchanged (expected 1341.8, same as before). FIRST REAL TEST: Sun 2026-10-25 (check the Sunday refresh shows steps with 'lines up next week', and that Monday's plan then treats those adds as already made). Limits: lookahead is a fixed 3 days; weighting of next-week points is 1.0 (not discounted for injury uncertainty beyond the usual p(play)); only adds, no lookahead for IR moves.

### Trade analyzer + trade values model (Tommy's request 2026-10-11; two parts: analyzer first, then draft-pick values)
Research and the proposed model are in `RESEARCH_trade_value.md` (market calculators, roster-aware methods, endowment effect, this league's constraints, validation plan, build phases, decisions needed). No code yet. Existing: Trades tab = finder v1 (`build_trades.py`, unvalidated), no analyzer. Part 2 (draft-pick values) plugs into the same engine. Also done 2026-10-11: live win probability now zeroes a team that has reached the games cap (opponent lockout); `live_matchup.py` year/opener derived.


## Trade analyzer (built 2026-10-11, part 1 of trade values)
- Trades tab top panel: instant check (market balance, acceptance band, top-5 keeper change) + "Analyze exactly" button (workflow trade-analyze.yml, token sl_token, result on branch trade-data).
- Engine: ingest/research/trade_value.py. Judgment parameters: future weight lambda (slider, default 0.15), posture weights (rebuild 0.7/1.4, balanced 1/1, contend 1.25/0.7), keeper noise 25%.
- trade_context.json / trade_market.json are built LOCALLY (build_trade_context.py needs the gitignored Hashtag CSV); re-run weekly and commit.
- TODO: validation replay (RESEARCH_trade_value.md section 4); part 2 draft-pick values; finder v2 on this engine.
- Draft-pick values built 2026-10-11 (RESEARCH_trade_value.md section 8). Open: re-run build_trade_context.py weekly; add new traded picks to ingest/research/draft_trades.json; Tommy-first draft-order fix is not modeled; finder v2 on this engine.
- Waiver claim odds built 2026-10-11: `ingest/research/waiver_intel.py` -> `dashboard/waiver_intel.json` (claims per dropped player by quality tier + each team's share of claims, 2024-26); pickup alert now says "Odds you win the claim: about X%" using every team's live waiver rank. Rebuild the intel file rarely (needs the local transaction files). Finding: even a 38+ pts/g player on waivers is claimed only ~21% of the time and contested ~5%, so claims are rarely competitive. TODO when real data exists: check the first few real claims against the stated odds.
- Late-swap slot planner built 2026-10-11: `ingest/research/late_swap.py` (simulates the night in tip-off order; ESPN locks a slot when that player's game starts; picks the slot assignment with the most expected points, swapping slots only when it beats the plan's own by 0.25 pts) wired into `plan_team(lateswap=True)` for MY team only; `late_swap` field on each plan day; This week tab shows the note and who can replace a doubtful (<80% to play) starter. set_lineup.py uses the plan's slots, so the lineup button now writes the late-swap arrangement. Check on real nights: does the 'no bench player can replace him' warning match what you see in ESPN? Not yet modeled: choosing a weaker early starter to keep a late flexible cover; swapping live at each tip-off.
- DFS "tonight" projections wired in 2026-10-17: `dfs_proj.py` fetches DailyFantasyFuel's table in every week-plan run (daily 5:37am + gameday 3:07pm/6:07pm ET + manual) and overrides the per-day level (`lvl_by_date`) for every date it lists; zero projection + injury tag O = out for that day. Chips show "day-specific projection". Raw numbers are not committed. TODO after opening week: (1) grade DK-implied points vs ESPN/our level vs actual (local pull log: run `pull_dfs_projections.py` at ~4pm ET on game days, data/dfs/log.csv); (2) DONE 2026-10-25: DFF numbers for Questionable players ARE conditional on playing (same player, Q days vs healthy days: median 0.977, 19 players), so no double discount; (3) make the late-scratch swap ranking and pickup alerts use the same number; (4) decide blend weight (env DFS_BLEND, default 1.0 = full replacement on listed days).
- DFS grading rig built 2026-10-17: Windows scheduled task "FantasyHub DFS monitor" (every 10 min, 11am-1am ET, runs ingest/run_dfs_monitor.cmd -> research/dfs_monitor.py) logs DFF changes (data/dfs/log.csv) and ESPN injury changes (data/dfs/inj_log.csv), local only. After a week of games: `uv run python research/pull_recent_gamelogs.py` then `uv run python research/grade_dfs.py --since 2026-10-20` -> accuracy vs ESPN/our model, bias by injury tag (is the number "if he plays"?), and reaction lag to Out/Doubtful news + "stale at tip" count. PC must be on for the log to be complete.
- Start-sit ordering changed 2026-10-17 (Tommy's point): who gets a starting slot is decided by level x p + (1-p) x (cover from the best player left out of the 10 slots), not level x p, because scratches are visible before tip and a bench player can be swapped in. Accounting (ef, totals) unchanged. Late-scratch swaps and free-agent options now use `tonight` (day-specific level).
- Stream chains + reclaim (2026-10-22, Tommy's point that many small rentals add up): `stream_search` now also tries CHAINS: turn the cheapest roster player's slot (3 candidates) into a rotating stream slot; the slot's future cost is charged once for the whole chain and later swaps (streamer for streamer) cost nothing and only need +5 (`CHAIN_STEP_MIN`). Future cost for a rostered drop is now scaled by `RECLAIM_LOSE`=0.25 (you can re-add him once he clears waivers, ~40h; ~18% claimed + free-agent exposure); the page says so. Watch: whether re-adds really succeed (log each reclaim), the 25% constant, and that the first chain step is not judged alone. Not modeled: the reclaim as its own add/roster move, a mid-week re-plan when a streamer is hurt.
- Streamers use the day-specific projection (2026-10-22, Tommy): (1) a free agent on a team whose slate is posted in the DFF table but who is NOT in it (Caleb Love, Cam Whitmore...) is treated as 10% to play that day (`dfs_proj.UNLISTED_P`; free agents only); (2) free agents the table prices for a posted day carry 60% of that number to later days (`DFS_FA_CARRY`); (3) new "Single-day pickups for tonight" table on This week (week_plan teams[me].tonight_picks, ranked by what each adds to tonight's lineup); (4) stream manager models the drop as a reclaim: the dropped player is out until his waiver clears (drop day + 2), back with probability 1-RECLAIM_LOSE, and rentals can end after their first game. Result today: no stream clears the bar once the reclaim risk is counted (+9 this week for the best, about -16 after the season-long risk of not getting the dropped player back). If Tommy's real re-add success is higher than 75%, lower RECLAIM_LOSE (log real reclaims). Also fixed the live-matchup wording that said the opponent HAD reached the cap when it was only projected to.
- Plan and pickup alert agree (2026-10-23): the stream search now also considers KEEPING an add (not just renting him), records per free agent the best "add and keep" option (`pickup_checks`, incl. the 'add today' version), and the pickup alert reads the plan: if the plan has him it says the day, the drop and whether to wait (waiting is worth 84% of the gain because a player like him is taken within 48h ~16% of the time); if not, it says what adding him would do to the week and the season. Injured players section moved below Suggested moves, collapsed by default.
- Add-now-vs-wait and agility (2026-10-23): long-run player levels (what he is worth after this week) are now ESPN and DELCO averaged, free agents shrunk only 10% when a veteran's two sources agree (was a flat 40%, which wrongly made Keegan Murray 29 against a 34-35 consensus); this week's points still use the day-specific numbers. Every plan step lists backups; the pickup alert weighs adding now (loses the dropped player's games) against waiting (risk of being taken: 25% for 34+ ppg, 16% for 27-34, 12% below); a new alert says when a player the plan is counting on is taken (with the backups). The plan re-optimizes at each refresh (5:37am, 3:07pm, 6:07pm ET), alerts check every ~30 min.
- Long-term adds fixed (2026-10-23): the rest-of-season gain was measured against an EMPTY slot, so injured scrubs on a free IR slot (Nae'Qwan Tomlin 17 ppg, +342) outranked real players. It is now measured against the roster plus the six best free agents standing in as the streamers you would use otherwise, and a player below that streaming level gets no rest-of-season gain (he can still appear for dynasty value). Wiggins (33.5) is now #2.
- Audit 2026-10-23 (Opus pass): PC daily refresh had crashed every day since Oct 11 (opener tip 'HH:MMZ'), fixed; breakout ledger frozen from the Oct 10 preseason file; heartbeat file dashboard/local_refresh_status.json + morning-brief warning if stale >36h. Early DFF grade (3 nights, n=314): DFF MAE 8.8 vs ESPN 10.5 vs DELCO 9.9; Questionable-tagged played 10/27; DFF shows outs a median 21 min BEFORE ESPN news; zero projection = out. Approved A-D built: (A) one engine for every free-agent surface (Other free agents = each FA's best rental/keep from the stream search; Long-term adds on the same long-run levels and fillers; Schedule streamers table removed); (B) availability risk ranks/times moves but never removes them (inclusion on full value; page shows both numbers); (C) Monday brief: last matchup expected vs actual + moves + drops not back; (D) extra 9pm ET plan refresh. Process: Claude commits code only; plan JSON comes from the scheduled runs.
- Trade finder v2 + league acceptance model (2026-10-23): ESPN shows only proposals involving your team, so acceptance was fitted on the 9 trades completed league-wide in 2023-26 plus Tommy's declined offers (research/trade_acceptance_fit.py). trade_value.LeaguePerception: half production as ESPN shows it (blended with season average after 5 games) above 25, half what this league paid (draft slot; keepers priced as first-rounders), +25% at age 25 or under, star premium (1, .5, .3, .2 by player), +25% on the other owner's likely keepers (his top 5); bands likely >= 1.15, possible >= 1.0. Used by the analyzer (exact and instant) and by research/find_trades.py (screens 1-1/2-1/1-2 swaps, runs the full analyzer on the best 5 per team, max 3 ideas per player given; weekly in the PC refresh -> dashboard/trade_ideas_v2.json; Trades tab cards with 'Open in the analyzer'). Old finder v1 cards removed (its sell-high/buy-low signals stay). Thin data: watch real responses to offers and refit.

## Standings tab follow-ups (low priority, agreed 2026-10-23)
- **Road ahead:** my next 4 opponents with a projected win chance each, plus remaining strength of schedule for every team (average projected opponent points over the remaining regular-season matchups; pairs and per-week projections are already inside build_playoff_odds.py).
- **Power rankings over time:** a chart of each team's weekly all-play rank. Tommy felt the title-odds sparkline covers most of this; revisit mid-season.

## Leverage-weighted moves (built 2026-10-24)
This week's gains are valued by their effect on this matchup's win chance (build_week_plan.py `this_week_value`, same win model as the live card); next week is scored in full on the last two days; a dropped player's chance of being lost grows with how long a rental keeps him off the roster. Check on 2025-26 final days (research/leverage_check.py): about a third fewer late-week moves than the flat +6 rule, about 40% more wins from them. Watch: real late-week decisions this season, and whether the 1-day sigma (model 134 vs 152 measured on final days) needs refitting.

## Strategy studies (agreed 2026-10-24; running a handful at a time)
- DONE #DELCO season mode: built (see delco_inseason_study.py, data/delco_season_mode.json). Not built: a team-changer-specific reset (tested, no gain). Possible later: teammate-based role signals.
- DONE #1 variance-aware lineups: not worth it (variance_lineup_study.py). Same-average players differ only ~10 vs 14 pts in game spread; a day's matchup spread is ~134, so the best case is +0.25 pts of win chance.
- DONE #2 week-importance weighting: win value varies a lot by week (title odds per win 0 to 1.8 pts, 10th-90th pct) but weighting moves by it adds only ~+0.04 pts of title odds per team-season per candidate move (win_value_study.py). Not built; revisit for the late-season clinched/eliminated case.
- DONE #3 opponent habits: built (opponent_habits_study.py; team_usage.json eff).
- SCHEDULED #4 projection blend: research/blend_study.py; the projection log gained career DELCO, last-15, level columns 2026-10-24; a scheduled Claude task runs it Mon 2026-12-14 9am (fantasy-hub-blend-study).
- DONE #5 depth and streaming (depth_value_study.py, 204 team-weeks of 2025-26): a held player is worth about 2.15 x his points per game per week; the cap trims depth value only ~17% at ranks 10-13 (stars-and-scrubs hypothesis NOT supported). The FIRST streaming spot is worth ~90 pts/wk (like holding a 41-ppg player) because it turns the 7 weekly adds into games; a 2nd/3rd spot only ~30-45 (they share the adds). Consolidation pays when it opens your first streaming spot, about break-even otherwise. Proposed (awaiting Tommy): value open roster spots in the trade analyzer/finder; designated streaming spot in the plan.
- DONE #1-from-#5: trade analyzer/finder credit a team's streaming spot (trade_value.stream_bonus); finder screen counts depth at 85%.
- BUILT #6 (2026-10-24): PO games column (Redraft/Teams/Season list); title_weights + title_per_pt in playoff_odds.json; playoff-prep mode in the plan's long-run costs (last 6 weeks); Season value shown as title odds. Study details:
  DONE #6 study (playoff_schedule_study.py): a point in playoff week 20/21/22 is worth ~6x/18x/21x a regular-season point in title odds (for DRNK at 83% playoff odds); NBA teams play 9-12 games across weeks 20-22. Proposed leverage awaiting Tommy.
- DONE #7 study (ir_stash_study.py): 2025-26 had only 5 injured free agents worth stashing (of 70 long absences by 28+ ppg players), all fringe and weaker after returning. BUILT 2026-10-24: light rule (free IR slot + an add that would go unused + Season value 60+) -> Do now item and alert.
- SKIPPED #8 (Tommy will decide on tanking himself).
- SET UP #9: research/calibration_check.py; monthly scheduled task fantasy-hub-calibration (1st of Dec-Mar, 9am); ledger now logs every team's projections each run. Early hint: dropped players taken within 2 days 2/29 (plan assumes 12-25%).
- DONE #10: playoff-week fixes in build_week_plan.py (no-matchup weeks count nothing; next week weighted by chance of playing; 35+ protection lifted for non-keepers in the final).
- BUILT rookie in-season projections (2026-10-25, Tommy noticed Peterson 37.2 -> 22.9): debuted rookies had dropped off the prospect engine onto a weak Kalman start plus 2-3 games. rookie_inseason_study.py (542 rookies, 2010-2025, leave-one-class-out): rest of season = a + b x prospect projection + c x pts/g so far; games get ~50% weight after 2 games, ~75% after 12 (held-out error 6.2 vs 7.7 projection-only); rookies improve ~+5 pts/g after their first games (in the intercept); minutes added nothing beyond points; one rule for all draft ranges. Later seasons move by 0.28 (1 game) to 0.70 (41 games) of the early surprise. asset_value_v2.py keeps this season's class on the prospect engine all season.
- BUILT opponent streaming (2026-10-25, Tommy's request after ROB's last-day adds swung the week): opponent_adds_study.py (3 seasons, 694 team-weeks: gross streaming is an owner trait, split-half r 0.82, season to season 0.76; assuming no streaming underestimates opponents ~50 pts/wk) and opponent_adds_net.py (2025-26 net of the player each add replaces: league +41/wk, owners -5 to +109, split-half r 0.76, shrink weight 10.5 weeks). Plan, leverage and live card now add the opponent's usual streaming for the days left (capped at 30 per add he has left); Tommy's side counts the plan's queued moves (proj_final). Refresh: rerun opponent_adds_net.py on this season's data once ~10 matchups are in (needs this season's league_days file).

## QoL backlog (2026-10-24)
- Season review page: each week's result vs the plan's projection, adds used, streamer points, moves suggested vs made, projection accuracy, calibration results.
- Trade analyzer polish: player search instead of checkbox lists, side-by-side lineup before/after, shareable link to a trade.
- One-tap moves (QoL #4, planned 2026-10-24, not built): extend the Set lineup workflow to adds, drops and waiver claims; the site's Do now items get a button that checks the move is still in the plan and the roster unchanged, then makes it in ESPN and logs it.
- DONE study (2026-10-25) reclaim risk (reclaim_study.py, 772 healthy drops 2023-26): share of dropped players taken by another team within 2 days = 32% (33+ ppg), 19% (27-33), 11% (20-27), 7% (<20); within 4 days 43/36/22/16%. Plan assumes max(15%, p_taken): about right for 33+, a little low for 27-33, too high for players under 27. Proposed (awaiting Tommy): replace lose_p with the measured table.
- BUILT one-tap moves (2026-10-25): roster_move.py + roster-move.yml + Do now button (adds/claims due today). Formats from the league transaction log (no capture). FIRST REAL USE IS THE TEST: watch it; a waiver claim is the less certain format.
- BUILT late-season motivation (2026-10-25): late_season_study.py + late_season.py (daily in refresh-league) -> late_season.json; plan playoff-week games + site season value. 2027 '3-2-1' lottery: multipliers small (0.95-0.96 for likely bottom-10 cores, 1.04 young). Revisit in February: set OVERRIDES for teams visibly tanking (e.g. teams without their own pick).
- BUILT lineup alert, other direction (2026-10-25): the pre-tip check also says 'Bench X' when a player the plan rests today is in the ESPN lineup.
- SCHEDULED bench-cover check: fantasy-hub-bench-cover-check, Mon 2026-11-16 9am.
- DONE study (2026-10-25) games-max + stream-rules replay (games_max_study.py, stream_rules_replay.py, 204 team-weeks of 2025-26): one more counted start ~39 pts; streaming under the current rules = +4.4 starts and +67 pts/wk net vs no moves (47+ starts in 53% of weeks vs 12%). Protection level makes no difference; looser rules are worse (min gain 0: -20/wk); saving adds for the last 3 days is worse; MIN_NET_GAIN 12 was +6/wk (se 2.5) on the simplified planner. Tommy kept 6; the monthly calibration task now checks whether DRNK's real 6-12 moves pay off.
- DONE study (2026-10-26) cap-risk replay (cap_risk_replay.py, 216 team-weeks of 2025-26 with real rosters and missed games): a planner that averages over every number of no-shows = current planner +0.1 pts/wk (se 0.5); perfect foresight of who plays is only +5.6/wk. The daily re-plan already captures nearly all the value; not built.
- DONE study (2026-10-26) last-day package (stream_rules_replay.py variant now+cram, 204 team-weeks): weighing "sit some players on the second-to-last day, then drop k / add k last-day streamers" as one move changed only 6% of weeks and lost 17 pts in those (overall -1.1/wk, se 0.8). Not built.
- BUILT short-term DELCO (2026-10-26) from short_term_study.py (2015-26, 77k held-out cases): next 3 games = season DELCO + 1.25 + 0.36 x (last-3 minutes - DELCO minutes) x his rate + 0.11 x (last-3 points - DELCO); fixes the +5.5 underrating after minutes jumps; teammates-out adds nothing beyond minutes. Replaces ESPN projection + last-15 for days after tonight (build_week_plan level_on, st_base/st_delta, fade 1.0/0.6/0.3 by days ahead); built in build_redraft.py (st_delta). Logged as st vs level_espn; the Dec 14 blend study judges it. Test #3 (real schedule for drop costs, stream_rules_replay now+sched): -0.2/wk, not built.
