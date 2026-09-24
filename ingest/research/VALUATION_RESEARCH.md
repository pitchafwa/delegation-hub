# Existing-Player Valuation Model — Research Notes

Built to answer one real question: of Tommy's current roster, which 3 players
are most valuable to keep for the 2026-27 season? Scope is existing-player
value change (not rookie/prospect evaluation — that's a separate model, for
the post-keeper draft).

## v2.0.0 — "Delegation League Dynasty Valuation Model" (current, frozen)

Everything below this section describes v1 (a single blended additive model),
superseded by a full DARKO-style rebuild: 9 separately-fit Kalman filters
(one per box-score stat, real game-level data, aging curve built into the
state transition) + a flat-rate fallback for steals + a season-carried-forward
placeholder for triple-doubles, recombined via the league's real scoring
formula. Full detail, every fitted parameter, every rejected variant with its
real numbers, and the final validation is in
`research/data/delegation_dynasty_model.frozen.json` (v2.0.0) — that JSON is
the actual source of truth; this is a pointer to it.

**Why it replaced v1**: v2 beats v1 on a fair, single-season, real held-out
comparison (2023-24 season, 369 real players) — Spearman 0.872 vs 0.866,
Pearson r 0.908 vs 0.890, MAE 5.14 vs 7.04. It also fixed a real, decision-relevant
bug in v1 that only surfaced once applied to a real 19-year-old (Cooper Flagg):
v1's single blended age curve had a peak age pinned at 22 (a boundary
artifact), producing a nonsensical multi-year decline for a teenager. v2's
per-stat curves, after their own bounds audit, don't have this problem.

**Two real, tested rejections worth remembering** (logged in the frozen JSON,
not just this note): steals and triple-doubles both got their own Kalman
filter built and rigorously tested, and both came out *worse* than simply
using a flat/carried-forward rate. They were pulled from the Kalman pipeline
specifically because the fancier approach lost the test, not because they
weren't tried.

**Also tested and rejected**: matching DARKO's own validation approach
(comparing against real next-game/DFS-style outcomes) — the composite
actually did *worse* than a simple rolling-10-game average at single-game
prediction (0.723 vs 0.745 Spearman). This model is validated for
season-ahead and multi-year dynasty value, not day-to-day lineup decisions —
a different model, tuned differently, would be needed for that.

## Literature reviewed

- **Aging curves**: Vaci et al., *"Large data and Bayesian modeling — aging
  curves of NBA players"* (Behavior Research Methods, 2019,
  [PMC6690864](https://pmc.ncbi.nlm.nih.gov/articles/PMC6690864/)). Peer-reviewed
  finding: players peak ~27-29, decline follows a power-law curve (fast then
  slowing), and — notably — decline *slope* doesn't differ meaningfully by
  position for Win Shares.
- **DARKO** ([darko.app/about](https://www.darko.app/about)): the real,
  production projection system this model's approach is modeled on. Exponential
  recency-weighting (each game weighted by β^t, β fit per-stat), a Kalman
  filter to separate real talent change from single-season noise, daily
  re-calibration, explicit team-change adjustments.
- **Shooting efficiency regression**: 3P% year-over-year explains only ~14.5%
  of next-season variance — highly noisy, regresses hard toward the mean
  ([source discussion, thepowerrank.com](https://thepowerrank.com/2020/07/28/predictability-vs-skill-in-sports-analytics-3-point-shooting/)).
- **Trades/role change**: ~59% of players see diminished per-game production
  in the season immediately after a trade; adjustment often takes a year+
  ([indihoops.com](https://www.indihoops.com/aau-boys-basketball-blogs/player-stats-trends-influence-nba-projections/)).
- **Load management ≠ lower injury risk**: a 9-season league study found no
  statistically significant link between resting for load management and
  reduced future injury risk
  ([PMC13499730](https://pmc.ncbi.nlm.nih.gov/articles/PMC13499730/)).

## Data

Real, free, no-auth NBA.com data via `nba_api` (`LeagueDashPlayerStats` +
`PlayerIndex`). 16 seasons, 2010-11 through 2025-26, 8,341 player-seasons,
cached in `research/data/raw/`.

Fantasy points per player-season computed using Tommy's **actual validated
league scoring formula** (see `validate_scoring2.py` — 94/94 real player-days
matched ESPN's own totals exactly):

```
FANTASY_PTS = PTS + 1.5*REB + 2*AST + 3*STL + 3*BLK + 3PM + 2*FTM - FTA - TOV + 3*TD3
```

## Known bug caught and fixed

The first dataset build wrongly flagged every 2025-26 player as "left the
league" because there's no 2026-27 data yet to check against (it hasn't been
played). Fixed by splitting current-season rows out as prediction *inputs*
rather than trainable rows with a missing target. Left in the codebase as a
reminder — this class of bug (using an incomplete "most recent" period as if
it were a resolved outcome) is worth double-checking on any refresh cycle.

## Models tested (real backtest on the model's terms, not adjusted after seeing results)

Train: transitions through 2021 (target seasons through 2022).
Test: transitions targeting **2023, 2024, 2025 seasons — real, held-out, the
model never saw these during training.**

| Model | MAE | Spearman rho | R² |
|---|---|---|---|
| Naive persistence (last season's rate) | 6.74 | 0.796 | 0.679 |
| 65/35 blend, no aging adjustment | 6.97 | 0.777 | 0.658 (worse than naive!) |
| Ridge regression + explicit age term | 6.13 | 0.832 | 0.741 |
| Gradient boosted trees (full feature set) | 6.09 | 0.840 | 0.744 |

**Honest takeaways:**
- A naive "just use last season's rate" baseline is already fairly strong —
  this domain has a real predictability ceiling.
- A blind blend of current + prior season *without* an age adjustment is
  actually worse than doing nothing — recency-weighting has to be
  age-aware or it hurts.
- An explicit aging curve is most of the real gain. Fancier ML (gradient
  boosting) barely beats simple Ridge regression once age is in the model —
  complexity isn't buying much here.
- Per-year backtest is not uniform: R² was 0.79 and 0.81 for 2023/2024 targets
  but dropped to 0.63 for the 2025 target season. That's a real limitation,
  not smoothed over.

## A mistake I made, caught only because Tommy pushed back

My first partial-dependence sanity check held every feature *except* age at
the population median and varied only age. It showed predicted value
declining monotonically from age 19 — which I then reported as a real
(if reinterpreted) finding. That was wrong, and it was a methodology error,
not a real property of the model: there is no real 19-year-old with a
median veteran's usage rate, efficiency, and minutes, so that chart was
extrapolating into a feature combination that doesn't exist and calling the
result a finding.

Corrected check (`realistic_age_check.py`): predict on real players' real
feature combinations at each age, instead of a hypothetical median player.
Result is a plausible shape — roughly flat through the 20s (~22-25 predicted
PPG), with real decline setting in around age 30-31 onward, down into the
teens by the late 30s. Noisier than a textbook peak-at-28 curve (small
samples at the extremes: n=41 at age 19, n=14 at age 40), but nothing like
"peaks at 19."

**This did not affect the actual roster projections already generated** —
`apply_to_roster.py` predicts directly on each real player's own real feature
row, never on a hypothetical median-clamped one. But it's a reminder to
validate a diagnostic itself before trusting what it shows, not just the
model.

One real, honest limitation surfaced by the corrected check: the model
noticeably *underestimates* some of the biggest age-19-21 breakouts in the
data (Giannis Antetokounmpo's real Year 2 jump: actual 33.7, predicted 24.2;
LaMelo Ball: actual 53.6, predicted 45.7). It's conservative on generational
young talent. Worth keeping in mind for Cooper Flagg's projection (49.6) —
if he's in that same tier, the real number could run higher than the model says.

## Known limitations (researched, not solved)

- **Coaching changes**: real, researched effect (coaches can meaningfully
  shift pace/usage), but no clean historical coach-by-team-by-season dataset
  was incorporated — team change (trade/free agency) is captured, coaching
  change is not. Flagged as a future enhancement, not faked with a proxy.
  Same open item the original football project noted as a known gap.
- **Injury history is single-season, not multi-year.** The model uses this
  season's GP as an availability signal but doesn't yet model a player's
  multi-year injury pattern (see Chet Holmgren case note below).

## Applied to Tommy's real 2026-27 roster

Model retrained on the FULL 2010-2024 historical set (all real transitions,
not just the pre-2022 training split used for backtesting) and applied to
his actual current roster, matched player-by-player via ESPN's live API.

Top of the ranking: Jayson Tatum, Amen Thompson, Cooper Flagg, Evan Mobley,
Chet Holmgren. See chat for the full table and the case-by-case real-world
context checks (Tatum's Achilles recovery + Jaylen Brown trade increasing his
role; Holmgren's multi-season injury pattern; Flagg/Mobley/Amen Thompson
reading as stable) that the model itself can't see.
