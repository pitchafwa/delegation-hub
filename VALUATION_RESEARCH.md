
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
