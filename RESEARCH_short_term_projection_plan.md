# Short-term projection program ("next few days" DELCO) — study plan

Drafted 2026-10-26 at Tommy's request. Goal: the best possible estimate of what a player will score over the **next 1–14 days**, for weekly decisions
(adds, drops, streams, who to save starts for, opponent projection). Not season or career value (season/career DELCO), not tonight (DailyFantasyFuel).
Starting point: v1 (short_term_study.py, live 2026-10-26) = season DELCO + 1.25 + 0.36 × recent minutes change × rate + 0.11 × recent scoring gap.

## 0. Principles (same as DELCO and the rookie model)
- Every signal must be computable **live** from feeds we have every morning (game logs, DELCO, injury reports, schedule, ESPN/DFF). No hindsight.
- Time-ordered evaluation only: fit on earlier seasons, judge on later ones (rolling origin), never the same games.
- Each signal is tested on its own first, then in combination; keep what survives held-out data, drop the rest (we have dropped plenty before).
- The final judge is **decisions**, not just error: replay last season's weeks with the new projection driving the planner and measure points and wins.
- Report results in plain English; Tommy approves each model change.

## 1. Define the target and the horizon (Phase 1)
Split the value of a player over the next days into two parts, modelled separately and multiplied back:
- **Level**: fantasy points per game *if he plays* in each upcoming game (points-if-plays, matching how starts are chosen).
- **Availability**: chance he plays each upcoming game (today: flat 85% / 81% / 61% by level; tonight uses the injury report).
Targets: per-game level for game k = 1..10 ahead, and per-day expected points over days 1–14. This tells us how fast each signal fades (replaces the
guessed 1.0 / 0.6 / 0.3 fade) and whether "two weeks" is a useful horizon or everything collapses to season DELCO by ~day 7.

## 2. Build the as-of panel and the test harness (Phase 1)
- One row per player per morning (2015-16 .. 2025-26, ~1.5M rows), every feature computed only from information available that morning.
- Sources already on disk: box scores 2010-26 (minutes, full stat lines, DNP reasons through 2023-24), DELCO beliefs before/after every game
  (delco_pregame_all), 5 seasons of hourly official injury reports (injury_reports_hourly.csv), schedules, opponent defense/pace (matchup context),
  league rosters/transactions 2023-27.
- Gaps to fill: starter flag for 2024-25 onward (re-pull box scores with START_POSITION), trade/team-change dates, point spreads if a free history exists
  (blowout risk). ESPN and DFF projections exist only from 2026-10 (projection log / DFF monitor), so they enter later, in Phase 6.
- Harness: one script that scores any candidate projection on the same held-out games, by subset (role up/down, injury returns, rookies, stars,
  fringe players, early/late season), with error, lean (bias) and calibration.

## 3. Baselines (Phase 2)
Season DELCO, career DELCO, v1, naive last-N averages (N = 3, 5, 10, 15), recency-weighted averages with tuned half-lives, ESPN's season projection
+ last-15 blend (the rule v1 replaced; reconstructible from box scores). Every candidate must beat the best of these.

## 4. Signal studies (Phase 3) — one mini-study each, a handful at a time
**A. Role and minutes**
1. Minutes trend at several lengths (last 1/3/5/10 games, weighted), and its slope; how much of a jump persists by game k ahead.
2. Starter vs bench changes (moved into or out of the starting five) — likely a sharper role signal than minutes alone.
3. Minutes stability (a steady 30 vs an erratic 30); rotation size of his team (deep vs short benches).
4. Blowouts, foul trouble, injury exits: re-test with better detection (fouls from PF; injury exits from the injury report the next day) — the first
   attempt (games simply skipped) was worse, so the question is whether *partial* weight beats full or none.
**B. Opportunity and usage**
5. Usage share (shots + free throws + turnovers per minute) and its change; assist share for playmakers.
6. Teammates out — rebuilt forward-looking: who is **listed out for the upcoming games** on the official report, their usage, position overlap,
   and how that production was redistributed last time each player sat (player-specific usage flow), not just "who sat last game".
7. Teammates returning: a starter coming back takes minutes from the fill-in (the reverse of 6; probably where v1 overrates fill-ins).
8. Trades and new teams: role reset after a trade; first games with a new team.
**C. Rate and luck**
9. Per-minute production change separately from minutes change (a real skill/role change vs volume).
10. Shooting luck: field goal / 3-point / free throw percentages vs the player's norm → regression of points over the next games.
11. Hot/cold streaks beyond the above (v1 found only ~11% carries; check by stat category).
**D. Game context for each upcoming game**
12. Opponent defense and pace (already in the daily plan via matchup context): re-measure the size of the effect on a 1–14 day sum.
13. Rest: back-to-backs, 3-in-4, long rests; age interaction (load management of veterans).
14. Blowout risk from team strength gap (bench minutes up, star minutes down in mismatches); point spreads if available.
15. Home/away and travel (small, test anyway).
**E. Player and season traits**
16. Rookies and second-year players (faster learning curves; ties to the rookie in-season model).
17. Post-injury returns: refit the post-return ramp (how many games, how deep) with the injury-report history; unify with short-term logic.
18. Season phase: first weeks (more noise, v1 const), pre/post trade deadline, the last weeks (tanking: tie to late_season.py), All-Star break.
**F. Availability (the second half of the value)**
19. Chance to play each upcoming game from: injury report status and its history for the player, back-to-back, age, minutes load, team
    situation, and "questionable for 3 straight days" patterns. Replace the flat 85/81/61% with a fitted model; check calibration.

## 5. Model building (Phase 4)
Compare, on the same harness:
- **Corrections to DELCO** (v1 style, extended with the surviving signals) — simple, transparent.
- **Structural**: points = minutes × per-minute rate, each with its own short-term model (minutes reacts fast to role, rate slowly); availability
  separate. Most interpretable for the site ("his minutes are up 6").
- **Two-speed DELCO**: a second, fast state inside the Kalman filter (short-term deviation that decays back to the slow season state); DELCO already has
  the machinery, this is "DELCO with a short memory".
- **Gradient-boosted trees** on the full feature set, with monotonic constraints where the sign is known; used as a check on what the simpler models miss.
- **Stacking** of the above, with weights by horizon.
Pick the simplest model within noise of the best.

## 6. Horizon, fade and uncertainty (Phase 5)
- Fit each signal's weight by days ahead → the empirical fade curve; decide the useful horizon (1 week vs 2).
- Spread of outcomes by player type and horizon (for the leverage maths and late-week risk decisions).

## 7. Decision value and deployment (Phase 6)
1. Replay 2025-26 weeks (stream_rules_replay / backtest planner) with the new projection deciding adds, drops and starts; compare weekly points and
   wins with v1 and with season DELCO. Ship only if it wins there.
2. Bring ESPN and DFF in with this season's logs (projection_log.csv, DFF monitor): is there a blend of the new number with ESPN, and does DFF's
   day-specific number improve tomorrow's and the day after's estimate (DFF posts the next day's slate early)?
3. Shadow period: log the new number next to the live one for 2–3 weeks, then switch; the Dec 14 blend study and the monthly calibration task keep
   grading it.
4. Site: show "next few days" on player cards and free-agent rows with the reason ("minutes up 6 over 3 games", "teammate X out").

## 8. Order of work and size
| Phase | What | Sessions (rough) |
|---|---|---|
| 1 | target/horizon definition, as-of panel, test harness, starter-flag and trade-date data | 1–2 |
| 2 | baselines | 0.5 |
| 3 | signal studies A–F (19 tests), a handful at a time, each with a short report | 3–5 |
| 4 | model comparison | 1–2 |
| 5 | horizon, fade, uncertainty | 0.5–1 |
| 6 | decision replay, ESPN/DFF blend, shadow log, site | 1–2 |

Suggested first batch after Phase 1–2: 1 (minutes trend lengths), 2 (starter changes), 6 + 7 (teammates out/returning, forward-looking), 19 (availability).
These are the likeliest big wins and the ones v1 handles worst.

## Results log
### Batch 1 (2026-10-26; research/st_batch1.py, log research/data/st_batch1.log; judged on 2022-23..2025-26 unless noted)
- **Test 1, minutes trend length:** the best pair is minutes weighted toward the last ~2 games (half-life 2) plus points weighted over the last ~4 (half-life 4).
  Typical miss vs v1: game 1 12.33 vs 12.39, games 2-3 12.51 vs 12.57, 4-6 12.72 vs 12.77, 7-9 12.95 vs 13.00 (about 0.05 better at every distance).
  Share of a minutes change that carries: 51% next game, 42% games 2-3, 35% games 4-6, 30% games 7-9. The points gap carries a steady ~34%.
  This is the measured fade, replacing the guessed 1.0/0.6/0.3. Stars still overrated by ~0.9; separate star terms do nothing. Not fixed yet.
- **Tests 6+7, teammates listed out / returning (official report; fit 2021-22..22-23, judged 23-24..25-26):** a teammate listed out helps next game (+0.19 × his level
  × player's minutes share), fading to zero by game 4. A returning teammate costs a little later on. Overall gain ~0.01: real, but too small to matter on top of test 1.
- **Test 19, chance to play:** a fitted model beats the flat 85/81/61% (Brier 0.142 → 0.135, most at game 1: 0.113 → 0.097) and is well calibrated.
  Key signals: recent minutes, level, age, time of season, minutes swings. The flat rule under-predicts (79% vs 82% actual), which made the whole 7-day number run 4 points low.
  The report status adds almost nothing here, because these are mornings after a game he played (rarely listed out then). Players already out are handled by the injury status path.
- **Week (7-day points):** v1 + flat 34.31 / lean +3.9; best level + model 33.60 / lean −0.2. Most of the gain is the chance-to-play model.
- **Test 2 (starters):** waiting on the 2024-27 starter download (NBA site throttled it once; restarted slower).
### Batch 2 (2026-10-26; research/st_batch2.py, log research/data/st_batch2.log)
Each signal added on top of the batch 1 winner (fit 2015-22, judged 2022-26). Typical miss over games 1-6 is 12.574 for the winner.
- **Kept: season DELCO slightly overstates how far a player sits from average.** The next few games come back toward the middle by 3-6% of the distance from 24.
  The pull is larger further ahead. Example: a 50-DELCO star is about 1 point lower and a 12-DELCO player about 0.5 higher.
  This removes the star overrating (lean −0.88 → +0.09), shrinks the rookie and fringe leans, and is the best single gain (−0.018; −0.039 for games 7-9).
- **Not worth adding** (each 0.003-0.005, about +0.007 all together; simplest model kept):
  - Minutes stability.
  - Rotation size: a deeper rotation, oddly, means slightly more points.
  - Usage change: about 27% of a usage jump carries.
  - Shooting luck: hot 3-point and free-throw shooting regresses about 80% back, but it is small in points.
  - Career vs season gap.
- **Did nothing:** per-minute rate change separate from minutes; it is already inside the points gap.
- **Rest (test 13):** no effect on points when a player does play. Back-to-backs −0.6%, older players on a back-to-back −1.6% more.
  Rest matters for whether he plays, which the chance-to-play model already uses.
### Batch 3 (2026-10-26; research/st_batch3.py, log research/data/st_batch3.log)
Base = batch 1 winner + DELCO slope (games 1-6 typical miss 12.556).
- **Test 4, games cut short:** count them in full; every discount was worse.
  - Short games, other than blowouts, foul trouble or injury: 6,968. Halving them costs +0.008; dropping them costs +0.032.
  - Blowouts: 1,836. Dropping them costs +0.007.
  - Foul trouble: 189. No effect either way.
  - Injury exits, listed on the next report (228, 2021+): no gain from discounting.
  - Even after the short games, full weight already runs a little high (lean −0.2 to −0.8). Discounting makes that worse (−1.5 to −2.4).
  - So a short night says something real about the next games. Players often come back limited, and only about 40% of a minutes change carries anyway. Flagg-type nights stay in.
- **Test 8, trades:** for the first 10 games with a new team, add +0.25 and weight his scoring with the new team (about 26% of its gap).
  For traded players: miss 12.37 → 12.21, lean +0.46 → 0.0. Tiny overall (−0.004); keep it, since it is cheap and fixes a known blind spot.
- **Test 17, back from 5+ days off:** about +0.4 to +0.5 for the next 5 games (they bounce back more than the short window says). Small (−0.002), and the lean goes to zero.
- **Test 18, season phase:** −0.016 overall.
  - First month: −0.55, and the recent window counts a bit less. First-month lean −0.6 → 0.0.
  - From day 150: +1.0 for most players, −1.6 for DELCO 35+ (late-season rest of stars).
  - The late part overlaps late_season.py's playoff multipliers, so pick one in Phase 4, not both.
- **Tests 12 + 14, opponent and blowout risk per upcoming game:** opponent is the biggest single gain so far (−0.042 at every distance).
  - +10% fantasy points allowed by the opponent → +6% for the player. Lean by opponent quintile runs −1.5 (toughest) to +1.5 (easiest).
  - Blowout risk (net-rating mismatch) adds nothing for starters or bench.
  - The plan already applies an opponent adjustment day by day (matchup_context.py, additive, capped ±2.5). Phase 4 compares that existing adjustment with this one on the harness and keeps one.
### Batch 4 (2026-10-26; research/st_batch4.py, log research/data/st_batch4.log)
Base = all kept so far (games 1-6 typical miss 12.535).
- **Test 11, streaks by category:** −0.007 overall. Most of a hot streak in points or assists carries; this is already counted in the points gap.
  Threes (−0.7) and turnovers (−0.6) mostly fall back. Steals+blocks, rebounds and free throws fall back a little.
  Real, but small. Keep threes and turnovers only if Phase 4 finds them worth it.
- **Test 16, rookies and second-year players:** rookies are still underrated by about +0.5 (+0.66 in their first 20 games).
  Rookie terms cut that to +0.37 / +0.54 (−0.002 overall). Second-year players are already right. A small keep, pending Phase 4.
- **Test 15, home/away:** about +1.4% at home and −1.4% away; travel (2+ straight road games) adds nothing. −0.003. Tiny; a candidate only as a day-by-day term next to the opponent.
- Phase 3 is complete except test 2 (starters), which waits on the download.
