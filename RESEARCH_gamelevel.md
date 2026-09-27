# Game-level projection research (2026-09-27)

**Goal.** Everything in this hub projects a player's SEASON-level rate (DELCO, recent form, ramp). Nothing adjusts for the specific game he's about
to play: who he's facing, how good their defense is, whether they're banged up, back-to-backs. This tests whether that context is worth adding, and
whether switching from a per-minute to a per-possession production basis matters given how much team pace varies.

**Data.** Full team box scores 2010-11 to 2024-25 (`pull_team_gamelogs_full.py`, ~19,000 games) joined to opponents, giving real per-game possessions
(standard NBA pace formula) and each team's defense allowed, both overall and by the position of the player scoring on them (using the existing
position classifier from `usage_flow.py`, applied to a new use here: defense, not usage redistribution). Everything used to predict a game is
**trailing/to-date only** (computed from games before it), so this reflects what could actually be forecast in advance, not hindsight. Rotation
players only (15+ trailing mpg, 10+ games into the season for a real baseline): 241,840 player-games, 1,130 players.

## Part 1: per-minute vs per-100-possessions

| Basis | Oracle (real minutes & pace) RMSE | Real-forecast (trailing minutes & expected pace) RMSE |
|---|---|---|
| Per-minute | 9.998 | 12.792 |
| Per-100-possessions | 9.929 | 12.739 |

Per-possession is consistently a little better, in both the oracle test (given his true minutes and the game's true pace, isolating the rate-basis
question) and the realistic forecast test (using trailing minutes and each matchup's expected pace, both known in advance). The gain is small
(~0.4-0.7%) because pace genuinely doesn't vary that much game to game (sd of 3.3 possessions around a mean of ~97) -- there just isn't a lot of room
for pace-adjustment to matter. Real, and the right thing to do if the game-level system below gets built, but not a fix on its own.

## Part 2: opponent context

Each candidate's effect on the deviation from the (pace-adjusted) forecast, bucketed into quintiles of a 240k-game sample:

| Context | Weakest quintile/state | Strongest quintile/state | Spread |
|---|---|---|---|
| Opponent's overall defense allowed | +0.11 pts (elite defense) | +1.56 pts (worst defense) | **1.4 pts** |
| Opponent's defense allowed to his position specifically | +0.65 pts (toughest matchup) | +1.50 pts (softest) | **0.85 pts** |
| Opponent missing rotation production that night | +0.29 pts (healthy) | +1.78 pts (banged up) | **1.5 pts** |
| Opponent on a back-to-back | +0.82 pts (rested) | +1.27 pts (tired) | **0.46 pts** |
| His OWN team on a back-to-back | +0.91 pts (rested) | +0.89 pts (tired) | ~0 (no real effect) |

**His own team's back-to-back barely matters** for how well he plays IF he's on the floor -- rest mostly affects whether/how many minutes he plays,
which the plan already handles elsewhere (b2b already discounts his chance of playing). It doesn't separately dock his per-minute production.

**Everything else is real, and the three biggest (opponent D, positional D, opponent health) are the most useful together.** Combined, held out
honestly (leave-one-season-out): baseline (pace-adjusted forecast alone) RMSE 12.744 -> with all five context features, ridge regression RMSE
12.695, gradient-boosted 12.697. A real, positive, repeatable ~0.4% improvement -- small in relative terms (single-game fantasy output is extremely
noisy no matter what), but unlike the team-change study, this one is a genuine, validated, positive result, not a null. Permutation importance ranks
opponent missing production and opponent overall defense as the two real drivers; positional defense adds a little on top; both back-to-back flags
contribute almost nothing once the others are in the model.

## Verdict

**Worth building**, unlike the team-change discount. A "matchup adjustment" for the daily plan -- opponent defense (overall + positional) and
opponent missing production, applied as a per-day multiplier alongside the existing usage-flow boost and post-return ramp -- would give a small but
real, honestly-tested improvement to daily start/sit and streaming decisions specifically, which is exactly where this matters (a close start/sit
call or a streaming target on a given night, not the season-long trajectories). Own-team back-to-back is not worth adding to the per-game rate (it's
already captured via play probability); opponent back-to-back is worth including but is the smallest of the four real signals.

**Not done yet:** this is research, not a build. If Tommy wants it, the natural next step is wiring live versions of these four features (opponent
defense to date, positional defense to date, opponent missing production for that day, opponent b2b) into `build_week_plan.py`'s daily level
computation, the same way usage-flow and the post-return ramp already work. Scripts: `pull_team_gamelogs_full.py`, `gamelevel_study.py`; intermediate
data in `ingest/research/data/gamelevel/`.
