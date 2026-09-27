
## Follow-up (2026-09-27): does DELCO (v2) account for a player changing teams? Tested.
**No.** DELCO's trajectory is a pure per-player, per-stat time series (`build_trajectory` in `kalman_vor.py`): each stat's Kalman posterior advances
only by age. No team, role or situation feature enters it anywhere -- a trade only shows up lagging, after real games are observed with the new team.
The only team-situation feature in this project is for INCOMING ROOKIES (`team_context.py`'s "available usage" measure) -- a different problem (a
rookie's first NBA team) from a veteran traded or signed elsewhere mid-career, which nothing currently touches. (The "team change is captured" line
in this doc's older "known limitations" section is about the retired v1 regression model, superseded by DELCO -- that capability did not carry over.)

**Does it cost real accuracy? Yes, meaningfully.** Real player-seasons 2010-2024, rotation players (20+ mpg, 40+ GP) in both a season and the next,
persistence-based (age aside, what DELCO reduces to) prediction of next season's rate (fantasy points per minute), split by whether he changed teams:
| | n | RMSE | mean signed error |
|---|---|---|---|
| stayed | 1,800 | 0.1164 | +0.017 |
| changed teams | 460 (20%) | 0.1477 | -0.011 |
Team-changers are ~27% less predictable (RMSE), and persistence is biased slightly HIGH for them (predicts more than they deliver) rather than the
mild low-bias seen for players who stayed. Their minutes also shift more (3.29 vs 2.61 mpg average change) -- a real role/usage effect, not noise.
Controlling for the player's own prior rate (so this isn't just "he was traded because he was declining anyway"), the gap holds: RMSE 0.144 vs 0.114,
a real residual bias of -0.027 (rotation players average roughly 30-33 min/g, so about -0.8 to -0.9 fantasy points a game of unexplained overshoot).

**Not implemented.** This is a real, quantified gap, not yet acted on -- Tommy asked whether it exists and whether it's been tested, not for a fix.
A plausible next step, if wanted: extend the rookie-style team-context measure (open production at the destination) to veteran offseason moves, and
test whether it closes this specific gap the way it helped rookies. Script: `ingest/research/team_change_test.py`.

## Follow-up 2 (2026-09-27): does "open production at the destination" close the team-change gap? Does the injury usage-flow model generalize to departures?
Two tests requested after the team-change gap above. `team_change_context_test.py`.

**1. Open production at the destination (the rookie measure, applied to veterans).** Reused `team_context.py`'s exact "open_fp" (production that left
a team minus production that arrived) for the team a veteran JOINS, added to the same rate0-only regression from the first test.
| | RMSE | 
|---|---|
| team-changers, rate0 only | 0.1443 |
| team-changers, + open_fp at destination | 0.1424 |
| stayed-put players, for reference | 0.1138 |
Real but small: the correlation between the leftover error and open_fp is +0.156 (right direction: joining a team with more open production predicts
a real rate gain), and it closes only about 7% of the gap between team-changers and players who stayed. This measure works far better for ROOKIES
(a blank slate absorbing whatever's open) than for established veterans, where fit, scheme, and role depend on much more than team-level open production.
**Verdict: real, small, does not close the gap.**

**2. Does the injury usage-flow model (usage_flow.py) generalize to normal roster churn?** Treated each offseason departure of a rotation player
(12+ mpg) as a full-season "absence" and ran the exact fitted injury uplift model (`UF.uplifts`) to predict how much of his production his STAYING
teammates would pick up, then checked that against their real next-season change (3,684 returning-teammate observations, seasons with a rotation
departure).
* Correlation between predicted uplift and actual change: **+0.093** (usage_flow's own in-season injury validation was far stronger).
* Bucketed by predicted uplift, the pattern isn't even monotonic: 0.5-1.5 predicted -> actual **-0.82**; 1.5-3.0 predicted -> actual -0.39; 3.0+
  predicted -> actual +0.33. A real signal would rise steadily left to right; this doesn't.
* Regression slope of actual on predicted: 0.12 (the in-season injury model itself is calibrated to about 0.70 of its raw estimate -- this is far
  below even that).
**Verdict: the in-season injury model does not transfer to offseason departures as-is.** Most likely reason: this test only let STAYING teammates
absorb the vacated production, when in reality a departure is usually paired with a NEW arrival competing for the same minutes (the in-season case has
no such complication -- a game missed just redistributes among the same roster that night). Offseason departures also often come bundled with other
simultaneous changes (coaching, scheme, further roster moves) that a single-player in-season absence never has to contend with. Open_fp above already
nets out arrivals at the TEAM level and still only found a small effect, which suggests the real story is that offseason role changes are just much
noisier than in-season ones, not simply a fixable omission.

**Recommendation: do not build a "veteran open production" or "departure usage-flow" feature.** Both real questions, both tested honestly, both come
back small-to-null. The team-change accuracy gap (Follow-up 1) is real but not explained by either mechanism tested here.

## Exhaustive deep-dive (2026-09-27, at Tommy's explicit request): every variable tried for predicting a team-change effect
Tried every candidate that could plausibly matter, including several not used ANYWHERE else in this project, on the same 460 real veteran
team-change transitions (2010-2024, rotation players 20+ mpg/40+ GP before and after). `team_change_exhaustive.py`.

**Candidates tried, alone (correlation with the rate change, after controlling for the player's own prior rate so this isn't just "he was already
declining/improving"):**
| Candidate | Source (new to this project?) | r (controlled) |
|---|---|---|
| Strongest same-position incumbent at the new team | usage_flow.py's position classifier, applied here for the first time | **-0.100** |
| Team-wide positional crowd at the new team | same | -0.099 |
| Team quality change (origin win% -> destination win%) | new: built from the tanking study's team records | **-0.106** |
| His rank on his old team (was he "the guy") | new | +0.035 (redundant with his own rate -- not new info) |
| Team pace change | new: derived from box scores already on hand, no extra pull | +0.063 |
| 3-point-rate (style) change | new: same source | +0.004 |
| Traded (a reciprocal move found) vs likely free agency/waiver | new: inferred from the data, no transaction feed exists | +0.060 |
| Coach change at the destination / at his old team | new pull: `team_coaches.csv` | -0.044 / +0.059 |
| Salary raise or cut on the move | new: `nba_salaries.csv`, 2000-2019 coverage only | +0.081 |
| Real injury heading into the move (health-confirmed) | 2022+ coverage only, n=54 -- too small to trust | +0.097 |

**Combined, held out honestly (leave-one-season-out):** rate0 alone: RMSE 0.1378. Every candidate above, together, via ridge regression: RMSE 0.1383
(worse). Via gradient-boosted trees (to catch interactions a linear model would miss): RMSE 0.1473 (worse still -- overfits on ~400 real transitions
with a dozen weak features). Permutation importance confirms the model leans almost entirely on the player's own prior rate; every team-change
candidate contributes next to nothing once combined.

**Age interaction (the one pattern that looked real) -- checked against a control group and mostly explained away.** Team-changers 24 and under
outperform persistence (+0.032), 25-29 are flat-to-negative (-0.024), 30+ underperform (-0.057) -- looked like a real "young players handle a move
better" story. But players who did NOT change teams show almost the identical age pattern (+0.037 / 0.000 / -0.034) -- this is mostly DELCO's own
per-stat aging curve reappearing, which already exists and already accounts for it. The leftover, team-change-SPECIFIC piece once age is modeled
properly (an explicit age x changed interaction term) is small: roughly -0.005 to -0.03 fp/min extra for players over ~25 who change teams, in the
same ballpark as (not meaningfully bigger than) the -0.027 residual bias already reported in the first team-change test.

**Bottom line, after trying every variable that could plausibly matter:** two real, directionally-sensible, individually-weak signals survive
scrutiny -- a strong same-position incumbent at the new team, and a step up in team quality -- both predict a real but small production dip (r about
-0.10 each, after controlling for his own level). Neither is strong enough to trust alone, and together with everything else tried (style, pace,
coaching, salary, trade-vs-FA, health, age) they do NOT beat simply persisting his own rate in honest held-out testing. **Recommendation: do not
build a team-change adjustment.** The gap is real (confirmed three times now, from three different angles) but this project could not find a
combination of available data that predicts it reliably enough to act on. If revisited, the two candidates worth another look with a bigger sample
(pooling more seasons, or a cleaner "positional overlap" measure using real play-by-play lineup data instead of box-score proxies) are the
same-position incumbent measure and the team-quality-change measure -- everything else tested (pace, style, coaching, salary, trade type) was noise.
