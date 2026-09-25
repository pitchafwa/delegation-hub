# Usage flow: who picks up a missing player's production (2026-09-26)

**Question.** When a rotation player is out, how much of his production lands on each teammate, and who gets it? Built into the weekly plan so lineups, adds, drops, timing and alerts all see it.

**Data.** `regular_season_box_scores_2010_2024` (found in the workspace): every roster player for every game INCLUDING did-not-play rows with a comment, and a position for starters (14 seasons, 424k player-games). The
official-report data (2021+) was used for absences in the out-of-time test on 2024-25 and 2025-26. Scripts: `usage_flow_panel.py` (panel), `usage_flow_study.py`, `usage_flow_study2.py` (fit + parameters),
`usage_flow.py` (the model), `usage_flow_backtest.py` (out-of-time test), `usage_flow_model.json` (parameters).
A note for later: the same box-score file has an injury/illness reason on many did-not-play rows back to 2010, so the injury-recurrence study could be extended from 5 to 14 seasons.

**Definition of an absence.** Any player averaging 12+ minutes over his previous 15 games who did not play. The box-score comment is unreliable for WHY (coach's decision is far more common than injury for these players and
labels drift), and for teammates the reason does not matter. In the plan, absences are the official report's Out (and Doubtful/Questionable, weighted by their measured play rates) and ESPN's OUT/IR.

## Findings
* About 70% of a missing player's fantasy production reaches his teammates (capture rate 0.70 for equal shares; the rest leaves the team's total). Star absences (33+ mpg) add about +3 to +4 fantasy points per game to a typical teammate; 28-33 mpg about +2 to +2.7.
* **Who gets it:** low-minute players gain the most relative to their share (under 15 mpg: capture 1.2 to 1.5 of their share; 30+ mpg starters: 0.5). Teammates who play the SAME position (from a style-based guard/forward/center model, 79% accurate on starters)
  capture about 3x more per share than the average teammate. Vacated production spreads fairly evenly (best share exponent 0.5) rather than going to the top scorers.
* Out of sample the fitted structure holds (R-squared of within-player variation 0.0227 on 2020-24 vs 0.0279 in sample), but single games are very noisy (MAE about 10 fantasy points), so this corrects the AVERAGE, not the box score.
* **Out-of-time test (2024-25 and 2025-26, the model was fit on 2010-11..2023-24), correct baseline** (recent-games baseline with its own absence uplift removed, then today's added back): forecast bias for games with a rotation absence goes from -1.0 to +0.5; MAE improves 4-7% for
  teammates with baselines under 20 fantasy points a game (the free-agent kind: 12-20 fp/g -4.2%, 6-12 -6.6%), about 1% for 20-30, none for 30+. Star (33+) absent: actual gain for 20+ fp teammates +5.8 vs model +6.95; 28-33: +5.1 vs +5.6; 20-28: +2.3 vs +3.0.
* The raw model overshoots, most for star teammates, so the plan multiplies uplifts by calibration slopes measured out of time and stable across both seasons: 0.91 (baseline under 12 fp/g), 0.89 (12-20), 0.87 (20-30), 0.74 (30+).
* A first version forgot that a player's recent-form baseline already contains whatever uplift the recent games had (an absent star for weeks); adding the full uplift on top overshot by +4.3. The plan therefore discounts long absences (up to 45%, the weight of last-15-game form in the level blend).
* **How long absences last** (official reports, 5 seasons): P(still Out at the next listing) = 76.5% after one listing, 89% after two, 93.5% after three, 96%+ after five, 98.6% after 15+. Multi-day plans use the product of these.

## Wiring (build_week_plan.py)
For every plan day and NBA team: absent players get a chance of being out (1 if listed Out that day; else the survival curve for a player who is Out now; Questionable 0.5 today; ESPN IR = long-term), then `usage_flow.plan_boosts` gives each teammate a per-day boost added to his level.
That flows into lineups, the day-by-day cap plan, add/drop suggestions, add timing, the opponent projection, streamers, and the plan's roster output (`boost`, `boost_why`). New "Injuries creating opportunity" box on This week; the suggested-moves table shows "+X pts/g while Y is out";
alerts send an "Injury opportunity: add X" message only when the plan itself recommends the add, names the drop, the absence is news (under 8 listings), and the boost is at least a quarter of the add's weekly gain.
Team rosters come from ESPN's NBA roster endpoint (the hub still lists retired players), matched by name to the hub's projections.

## Not done / caveats
* No replay of the full add/drop decision on the 2025-26 league weeks with the boost switched on (only the forecast-level out-of-time test above); that comes after real weeks exist.
* Untested live (no absences in the offseason): use `USAGE_TEST_OUT="Player A,Player B" uv run python research/build_week_plan.py` to force absences for a test.
* Positions come from a style model, not lineup data; usage of the missing player at the position level (who exactly takes his shots) is approximate.
* Assumes the projection for teammates is a season-mixture level (ESPN season projection blended with last-15); a player projected from a healthy-roster baseline would need a bigger uplift.
