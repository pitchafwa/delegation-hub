# Injury research (2026-09-26): what the injury type tells us, and what it doesn't

**Data.** Five seasons (2021-22 to 2025-26) of the NBA's official injury reports, both the late-morning and the 5:30 PM ET report for every game date, with the
reason text (138k rows), joined to box-score logs (played = minutes > 0). Earlier work (`nba_injury_reports.py`) only found the newer 15-minute file names, which
exist from Dec 2025, so its play rates came from ~100 game days of ONE season (the earlier note that said two seasons was wrong). The hourly-named files
(`Injury-Report_2022-01-20_05PM.pdf`) go back to 2021-22; `pull_injury_reports_hourly.py` reads them from word positions. Analysis: `injury_common.py`
(reason parsing), `injury_playrate_study.py`, `injury_recurrence_study.py`. Prosportstransactions (the usual multi-decade injury log) is behind a Cloudflare
challenge and was not scraped.

## 1. Probability of playing, given the status tag (rotation = 20+ mpg over the previous 15 games)
| Status | rotation | bench |
|---|---|---|
| Available | 94% | 70% |
| Probable | 92% | 76% |
| Questionable | 55% | 42% |
| Doubtful | 2.7% | 2.7% |
| Out | 0.4% | |

These match the numbers already in the plan (which came from one partial season), so nothing there was badly off.

**The injury type barely matters; recent participation matters a lot.**
* Questionable, rotation players, by injury tier: minor 52%, moderate 57%, major/surgery-return 75% (n=158), recovery/management 58%. By body part: 49-58%, no
  pattern. So a "minor" Questionable is NOT more likely to play than a "major" one; the major-injury Questionables are players teams are easing back, and they
  play more often. Out-of-sample log loss (train 2021-24, test 2024-26): status only 0.535, + rotation 0.524, + injury tier 0.524, + body part 0.524.
* **What does predict:** did he play his last game / how long has he been listed.
  Questionable, rotation: first day listed 44.5%; listed before and PLAYED his last game 81.4%; listed before and MISSED his last game 41.6%.
  Questionable bench: 35.8% / 83.0% / 29.1%. Probable: 87.7% / 97.9% / 77.9%. Available: 95.7% / 99.1% / 79.3%. Adding this lifts out-of-sample log loss from 0.524 to 0.452.
  The pattern is stable across all five seasons (missed-last 34-43%, played-last 76-84%).
* Practical rule: a Questionable player who played through it last game almost always plays (81%); one who has been sitting is a coin flip minus (42%); a brand-new
  Questionable is 45%.

## 2. Recurrence (does the same body part come back?)
A first pass counted listing fragments (a player skipped for a few days by the report) as separate episodes, which overstated recurrence. Fragments with no game played in
between are now merged, so a new episode needs a real return to play. Corrected results:
* A prior episode (3+ game days out) in the same body-part group over the previous two seasons roughly doubles the chance of a new one this season: 22% vs 11.5% with only
  other-group history and 7.4% with none. Holding the TOTAL number of prior episodes equal, the odds ratio for same-group history is 1.45; each extra prior episode of any kind adds only 1.15.
  So "injury-prone" is mostly body-part specific, and a broken thumb after a broken rib is only weakly predictive.
* **Same side.** Among players with two episodes of the same part: 71% are on the SAME side (50% = no effect): knee 73% (n=119), ankle 68%, hamstring 74%. Within six months of the
  previous episode it is 82%. Your hamstring-on-the-same-leg hunch holds, less strongly than the first pass said.
* **Longer when it recurs: NOT supported** after merging (first episodes average 17.5 game days out, same-side recurrences 13.9). The earlier "5 games longer" was an artifact.
* Season level: last season's games missed predicts next season's only weakly (R-squared 0.05); injury counts, repeat parts and major injuries did NOT improve it.

## 3. Five-tier injury risk for the board (built)
Predicted chance of missing 30%+ of next season from games missed in the last three seasons, age and minutes (logistic, leave-one-season-out over 1,362 player-seasons,
2022-23 to 2025-26): AUC 0.686 vs 0.628 for the old rule (count of seasons with 25%+ missed), calibrated (top quintile predicted 73%, actual 72%). Tiers: Low, Below average,
Average, Elevated, High (cut points 27%, 36%, 48%, 62%); players under 15 mpg get no tier (their missed games are mostly coaching decisions). Injury DETAILS (episodes, days out,
repeats, big injuries, still-out-at-season-end) did not improve the season-level prediction (AUC 0.677-0.686), so they are TAGS on the player: "recurring <side> <part> xN" (2+ episodes
in two seasons) and "returning: still out at end of last season" (31% missed half of the first ~16 games vs 18% otherwise, though the history already captures most of that).
Script: `build_injury_risk.py` (coefficients from `injury_risk_model.py`); the board shows a coloured dot (green = low, gold = elevated, red = high, none = average) with the reason in the tooltip and a
detail line when a player is opened.

## 4. Plan availability (built)
`availability_table.json` (from `build_availability_table.py`) gives P(play) by official status x rotation/bench x state, and `build_week_plan.py` uses it for today's game:
state = first day listed / listed before and played his last game / listed before and missed it, worked out from the previous 7 days of official reports plus ESPN's per-player
game log (`availability_state.py`; 54 of 60 sampled historical cases matched the study labels, the rest are report duplicates). Everything falls back to the flat rates if a
step fails. Tomorrow's designations use the pooled rate for that status. Untested live until the season starts (no reports in the offseason).
Caveats: episodes are defined from report listings, body part comes from the printed reason, and the 2025-26 11AM reports are thin (61 days). Results are descriptive, not causal.
