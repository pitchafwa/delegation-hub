# Injured-player advisor and live matchup tracker (2026-09-26)

## Injured-player advisor (This week tab, "Injured players: hold, stash or drop"; return estimate inside status alerts)
* **Time out.** From five seasons of official reports (episodes of Out listings merged across gaps with no game played): P(back within k more games | out s games so far) by body-part group and injury tier
  (major / moderate / minor), with fallbacks to tier and to all injuries when a cell has under 40 episodes. Fit on 2021-24 and tested on 2024-26 it was calibrated within about 3 points (e.g. 6+ games out, back within 8 more: predicted 38%, actual 41%).
  Typical: a moderate ankle/foot injury out 2-3 games is back within 8 more games 56% of the time; a major knee injury out 8+ games only 6% (and 23% within 45). Right-censoring at season end slightly overstates the long tail.
* **ESPN's estimate.** ESPN's public injury feed gives type, side, detail, its own return date and the news text. In season the plan uses the LATER of ours and ESPN's (team timelines run optimistic; not yet graded); in the offseason ESPN's date leads.
* **The return ramp** (about 1,000 returns after 5+ missed team games by rotation players, 14 seasons of box scores): first game back about 70% of usual minutes and points, games 2-3 about 85%, later about 90%, and he plays only about 75% of his next 10 games.
  (Absences of 25+ games are rare in the box-score data, so the 10-24 game ramp is used for long ones.)
* **Verdict.** Value after return = (level x 0.88 - 24) x games left after return x 0.8, plus a keeper term (5-keeper asset x 73 x 0.10). Cost of holding = nothing if an IR slot is free, otherwise the best free-agent swap's weekly gain x weeks out x 0.6.
  ACTIVATE / KEEP ACTIVE (back within a game or two), HOLD on IR, MOVE TO IR, STASH (out up to 20 games and worth the spot), HOLD, or DROP.
* Untested live (offseason feed only). Test with `INJURY_TEST="Name|Type|Detail|Side|ReturnDate;..." uv run python research/build_week_plan.py`. To grade ESPN's dates and ours, log them in season (ledger).

## Live matchup tracker (top of This week when the plan's matchup is live)
* **Data path.** The alerts job (already run every 20 minutes by cron-job.org) runs `live_matchup.py`, which reads ESPN's live matchup totals and today's box scores, and force-pushes `live_matchup.json` to the throwaway `live-data` branch (one commit, no site rebuild). The page reads it from raw.githubusercontent.com every 2 minutes.
  No extra cron job.
* **Win probability model** (`win_prob_study.py`, 2025-26 real matchups, both sides of every pairing): win = Phi((margin now + 0.72 x expected remaining margin) / sigma(days left)), sigma (points of margin) 135, 138, 168, 203, 232, 256, 277 for 1-7 days left.
  The 0.72 shrink is because projected differences between teams overstate real ones (the same slope, 0.72, appears for the cap-aware forecasts of Phase 2). Calibration by bin was within about 2 points and Brier 0.127 (0.25 = coin flip).
* **Variance advice was dropped:** raising a lineup's spread by 5% moves win probability by about 1 point, far smaller than the weekly spread (about 250 points). The tracker says so instead of giving "take risks" advice.
* **Expected rest of the week:** players in each team's ACTUAL ESPN lineup who have not played today (plan level x chance of playing) plus the plan's expected points for later days. The plan is refreshed overnight and about 3pm and 6pm ET.
* **Caveats to verify on opening night (Oct 20):** ESPN's live scoring fields (`totalPoints`, `pointsByScoringPeriod`, `rosterForCurrentScoringPeriod` stats during games), whether a player mid-game counts as played (his live points are counted, remaining minutes ignored), and the `live-data` push.
  Tested against last season's finished games (dry run).
* **Optional add-on in the backlog:** alerts when win probability crosses 25% or 75%.
