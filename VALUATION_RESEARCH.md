
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
