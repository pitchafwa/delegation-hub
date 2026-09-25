# Start/sit + waiver strategy — research (2026-09-24)

Goal: maximize points scored in each weekly matchup of Tommy's ESPN league (12 teams, H2H points, league 600271905).
Research first, feature second. Everything below is either read from the league's real ESPN settings, measured on this
league's real 2025-26 lineups, measured on NBA game logs, or taken from a cited outside source. Where something is
unverified it says so.

## 1. The league's actual rules (ESPN settings + 2025-26 history)

| Rule | Value | Status |
|---|---|---|
| Format | H2H points, 12 teams, 6 make playoffs | read from settings |
| Matchups | Mon–Sun weeks. 2025-26: period 1 = 6 days, periods 2–16 and 18–19 = 7 days, **period 17 = 14 days** (All-Star), then 3 playoff rounds (periods 20–22). Rest-of-2026-27 dates (RotoWire/Yahoo): opener Tue Oct 20 (6 days), 14-day matchups at Week 7 (NBA Cup) and Week 17 (Feb 15–28), playoffs Mar 15–Apr 4 | 2025-26 measured; 2026-27 ESPN calendar not published in settings yet — **verify** |
| Starting lineup | **10 slots: PG, SG, SF, PF, C, G, F, 3 UTIL.** Bench 5, IR 4 | read from settings |
| Lineup changes | Daily, and each player locks at **his own game's tip-off** (individual-game lock) → late swaps are possible until each game starts | read from settings |
| Adds | `matchupAcquisitionLimit` = 1.143 per scoring period ≈ **8 adds per 7-day matchup**. Drops/trades presumably free | setting read; scaling for the 6- and 14-day periods and exact counting rules **unverified** |
| Waivers | Traditional; claims process **Sundays 8am**; 24-hour waiver period | setting read; ESPN's help pages describe daily processing, so how a Sunday-only league treats players dropped mid-week is **unverified** — test with a real add/drop in the first week |
| Games-played cap | **40 started player-games per standard 7-day matchup**, scaled by length (≈34.3 for a 6-day week, 80 for a 14-day week). Checked at the **start of each day**: if you begin a day under the cap you may start anyone that day and go over; if you begin a day at or over it, nothing counts for the rest of the matchup. So a team at 39 entering Sunday can start 10 players and finish on 49 | **Confirmed** by ESPN's own help page (Games Played Limit) and by the league's 2025-26 lineups (see section 2b). Tommy's reading was exactly right; my first pass was wrong to call it non-binding |
| Keepers | 3 in 2026, 5 planned later; keeper cost = a roster slot | from earlier work |

## 2. What actually decides weeks in this league (measured on 2025-26 lineups)

Script: `ingest/research/league_lineup_analysis.py`, `pull_league_history_2026.py` (regular season, 1,668 team-days).

* **Games started is the main driver.** Weekly points vs games started: correlation **0.90**; every extra started game is worth
  about **36 points** (an average started game scores 38). Teams start on average **39 of a possible 70** slot-days a week.
* **Top vs bottom.** The three best teams started 43–44 games a week; the three worst started 27–32. That ~15-game gap is
  worth ~570 of the ~700 points per week separating them — far more than a difference in player quality.
* **Leakage is common.** In **82%** of team-weeks at least one player-game was lost to lineup management (a player played
  on the bench while a starting slot was empty or held a non-player): 4.1 per week on average, 5–11 per week for the
  worst-run teams, ~2 even for the best. (Approximate: uses each player's default position, so it slightly overstates.)
* **Weeks are close.** Median matchup margin 228 pts; **42% of matchups decided by <200 pts (~5 started games), 18% by <100.**
  Within-team weekly swing (SD) 253 pts vs a between-team spread of 222. A team that was better over the season by 100–300 pts
  won 84–85% of weeks; by 300+ it won every time.
* **Free-agent value.** Ranking 2025-26 players by fantasy pts/game: ranks 200–280 (roughly the free-agent pool, since 180–196
  players are rostered) average **21.6–25 pts/game** at 20–22 minutes. So filling an otherwise **empty** starting slot with a
  free agent is worth about **+22 points per game**, vs 38 for a typical started game.

### 2b. The games cap, measured on the league's real lineups (`ingest/research/games_cap_check.py`, 2025-26)
* In 7-day matchups, **every one of the 12 team-days that began at 40+ starts had zero starts** (lineup slots locked); every
  day that began at ≤39 could start up to all 10. The highest weekly total in any 7-day matchup was **49 = 39 + 10**.
* In the 6-day opening week totals cluster at 33–35 (cap ≈34.3) and the highest was 41; in the 14-day All-Star matchup
  nobody got near the ~80 cap (best 55), so it does not bind there.
* The cap rarely binds *early*: only **5 of 204** team-weeks crossed 40 before the final day (and each then lost the remaining
  days' games). But **51%** of team-weeks finished above 40, nearly all by crossing it on the final day.
* So the cap shapes strategy in two ways: it puts a ceiling of ~40–49 on any week, and it makes the *order* of starts matter:
  low-value starts spent early can push you to the cap before better games later in the week are available.
  For the average team (38.6 starts) volume is still the limit, not the cap.

## 3. What single-game information is worth (NBA game logs, out-of-sample)

* **Single-game noise is huge.** Predicting one player's fantasy points from his own recent form has RMSE ≈ **13 pts**
  (mean ~30). Best predictor: an exponentially-weighted average with a half-life of 5–10 games (RMSE 13.09) vs season-to-date
  13.46, last-20 13.34, last-5 13.67 — recent form deserves weight, but only ~3% better. (`form_window_test.py`)
* **Vegas game environment adds nothing measurable once you have a decent baseline.** 20,340 player-games from 2025-26 with
  real DraftKings/ESPN BET lines, fit on the first 60% of the season and scored on the rest: adding implied team total, game
  total, spread size and favorite status changed error by **−0.2% to −0.4% (slightly worse)**. Effect sizes are small: starters
  on teams favored by 12+ lose ~2.7 minutes but only ~0.7 fantasy pts; a high game total is worth about +0.06 pts per point of
  total. (`vegas_environment_test.py`; effectively one season: ESPN's free odds feed only had usable lines for ~200 games of 2024-25, and adding those changed nothing. Free historical archives (Sportsbook Reviews Online, Kaggle) could extend it to 2008-2025 if we want more certainty.)
  Outside sources agree it is directional at best: RotoGrinders warns not to double-count information already in
  projections; a public test of ~41k props found overs hit 46.7% in close games vs 40.9% in 20+ point blowouts (that is a
  betting-line effect, not a fantasy-points effect).
* **Defense-vs-position is noise** per an outside backtest (switch-heavy modern NBA); not tested by us.
* **Back-to-backs matter, through availability.** Rotation players (24+ mpg) sit a short spell on the **second night of a
  back-to-back 10.9% of the time vs 6.1% otherwise** (37,728 player-games, 2022-23 to 2025-26, same pattern in every season
  and minutes tier, stars included). Expected cost ≈ 5% × 40 pts ≈ **2 points per star game** — real but small; the NBA's
  own load-management data says players are also less efficient on the second night. (`rest_absence_test.py`)
* **Teammate absences shift roles** (outside sources: Nembhard 28.9 → mid-30s minutes without Haliburton; RotoWire minutes
  projections update with every injury). This is the big *information* edge in daily fantasy, but it needs injury news as
  fast as the market — not yet tested here.

## 4. The strategy this implies (ranked by size)

0. **Plan the week as a budget of ~40 starts plus a final-day overflow of up to 10.** Never cross 40 before the last day unless you are
   happy to forfeit the remaining days; aim to enter Sunday at ≤39 and use every slot that Sunday's slate allows (max 49); in a 6-day
   week the target is ≈34 entering the last day. A game's real value is its points minus the cap's opportunity cost (the points of the
   best game you'd otherwise be locked out of), so once you can reach the cap, quality of the starts matters more than filling every slot.
1. **Fill every starting slot every day** (while you are below the cap). Worth ~36 pts per game recovered, ~4 games/week of leakage on average, far more
   for a careless week. This is a discipline/tools problem more than a modeling problem: an assignment of players to
   the 10 position slots each day that maximizes expected points, with the day's injury news applied and late swaps used.
2. **Use the 8 weekly adds on idle slot-days.** Adds do not roll over. The typical roster has ~30 empty slot-days a
   week, and one add covers a player's whole week of games (3–4). Best targets: free agents with games on the days your
   healthy players are not playing, and whose positions fit an open slot. Value of an add = points it adds on idle
   slot-days (~22/game) + upgrades over your worst starter that day − the games of whoever you drop. Streaming only pays
   because adds are capped and roster/bench spots have an opportunity cost, so rank adds by *points per add used*.
3. **Build the roster for schedule coverage.** Bench players who play on the days your stars don't are worth more than
   their per-game numbers; prefer players on teams with more games that week (the same thing streaming does, planned in
   advance). Use the 4 IR slots so injured stars don't eat bench spots.
4. **Availability over quality at the margin.** Never start an OUT player; check status before each tip-off; expect
   second-night rest; keep flexibility at the end of each day (individual-game lock lets a late game's player replace an
   early scratch).
5. **Quality still matters, but weakly at the single-game level.** Use recency-weighted form on top of the season model;
   don't chase last-5-game hot streaks.
6. **Matchup state.** Weekly outcome ≈ a coin flip adjusted by the gap: with a 253-pt weekly SD per team (~360 on the
   difference), being up 150 is ~66%, 300 ~80%, 500 ~92%. Mid-week, with the opponent's remaining games known, use these odds.
   Variance-seeking (or -avoiding) picks barely move this compared with game count — single-game noise averages out over ~40
   games (~80 pts of weekly SD from single-game luck vs 253 observed; the rest is volume and injuries). So the sensible
   rule: when trailing, maximize *games* (streamers on open days); when leading, protect against injuries (keep depth).
7. **Playoffs (weeks 20–22).** Two-game weeks are decisive (RotoWire's 2026-27 note: some teams get only 9 games over the
   three playoff weeks); pick and hold players by playoff-week game counts from ~week 17.
8. **Not worth building now:** Vegas-driven projections (no measurable lift), paid player-prop feeds (NBA props need a
   $99/mo API tier), defense-vs-position tables.

**Answer to "which approach first?"** Phase 1 ("this week": daily best lineup + adds) should come first, and it has to include the cap
logic from the start. It captures the largest measured leaks (unfilled slots and lineup errors) and already ranks adds by schedule,
which is most of what roster planning does; the cap makes the weekly plan a small optimization (choose which player-games to
start, day by day, so the total lands at ≤39 entering the final day and as high as possible after it). Roster planning around
the schedule follows as Phase 2.

## 5. Data we can get (free, CI-friendly)

* ESPN league API (already wired in): rosters, lineup slots, injury status, free-agent pool, transactions, settings.
* ESPN scoreboard/core API: NBA schedule by date; DraftKings/ESPN BET spread/total/moneyline (history to ~2020-21).
* NBA game logs (have 2010–2026) for form/rest/absence models; our season-level model (`hub_data.json`) for player level.
* Not free: player props, real-time injury feeds beyond ESPN's status field, minutes projections (RotoWire).

## 6. Proposed build (after Tommy's approval — nothing built yet)

**Phase 1: "This week" tab.** For the selected team (default Tommy's: DRNK, "Most of us Can't Legally Drink"): day-by-day lineup grid with the optimal assignment,
idle slot-days highlighted, adds remaining, ranked add/drop suggestions with points gained per add, plus the opponent's
projected total and a win-probability read. Expected points per player-game = P(plays) × recency-weighted level (season model
blended with an EWMA of recent games) with small home/back-to-back/blowout adjustments. Refreshed daily by the existing
GitHub Action; a morning-of update near tip-offs is a stretch goal (no reliable free source of lineup news).
**Phase 2:** lineup-lock reminders/late-swap suggestions, week-ahead schedule view, playoff planner.
**Before trusting it:** replay it on the league's 2025-26 weeks and compare what its lineups/adds would have scored vs what
each team actually scored; verify the add-limit and waiver behavior in week 1.

## 7. Open questions for Tommy
1. Which fantasy team is yours (so the tab defaults to it and opponent logic works)?
2. Do you want reminders (email/push) or is a page you open in the morning enough?
3. Priority between "set my best lineup today" (Phase 1) and "plan my roster around the schedule" (Phase 2)?

## Sources
* [Fantasy Projection Lab — Vegas lines as projection inputs](https://fantasyprojectionlab.com/vegas-lines-and-fantasy-projections) (no basketball evidence)
* [RotoGrinders — Betting markets and NBA DFS](https://rotogrinders.com/fantasy/lessons/dfs-how-to-use-nba-odds)
* [DataStreak — Do blowouts kill NBA prop overs?](https://datastreak.com/insights/nba-blowouts-4th-quarter-props)
* [RotoWire — 2026-27 fantasy basketball schedule analysis](https://www.rotowire.com/basketball/article/2026-27-fantasy-basketball-schedule-analysis-136183)
* [RotoWire — NBA projected minutes explained](https://www.rotowire.com/basketball/article/nba-projected-minutes-explained-fantasy-basketball-97473)
* [theScore — Can a streaming strategy work in fantasy basketball?](https://www.thescore.com/news/1085822)
* [ESPN — Rules: Roster settings (games-played limits, acquisition limits)](https://www.espn.com/espn/print?id=4333729)
* [ESPN Fan Support — Games Played Limit](https://support.espn.com/hc/en-us/articles/360056369011-Games-Played-Limit)
* [ESPN Fan Support — Waiver period](https://support.espn.com/hc/en-us/articles/360012531592-Waiver-Period)
* [OddsPapi — Odds API pricing comparison](https://oddspapi.io/blog/odds-api-pricing-2026-comparison/)
* Scripts (all in `ingest/research/`): `league_lineup_analysis.py`, `pull_league_history_2026.py`, `vegas_environment_test.py`, `pull_espn_game_odds.py`, `rest_absence_test.py`, `form_window_test.py`

## Update 2026-09-24 (later): drop rules, projection source, props
* **Drop rules (keeper league, 5 keepers):** each team's top-6 dynasty assets and anyone projecting 35+ are never suggested as drops; every other
  suggestion is scored NET of a future cost = max(0, dropped level - added level) x 3.3 games x 6 weeks. Injury/short schedule this week never reduces it.
* **Single-game projections:** ESPN's projection first (blended with the last 15 games in season); our model only as a fallback.
* **Sportsbook player props -> league scoring:** `ingest/research/props_projection.py` converts lines + over/under prices into stat means
  (negative-binomial with variance fitted on 2023-26 logs) and scores them with the league formula; FTM/FTA/TD3 (no prop markets) come from ESPN's
  stat line. Needs `ODDS_API_KEY` (The Odds API; player props for NBA, ~7 credits per game per day). Free data sources for props are thin: ESPN has none;
  historical prop archives are paid. So the plan logs ESPN / model / props projections for every player each game day to
  `dashboard/projection_log.csv` (shadow mode) so the sources can be scored against real results after a few weeks before we trust any one.
