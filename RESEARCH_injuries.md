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
* A prior episode (3+ game days out) in the same body-part group over the previous two seasons roughly doubles the chance of a new one this season: 23% vs 11.5% with
  only other-group history and 7.4% with none. Holding the TOTAL number of prior episodes equal, the odds ratio for same-group history is 1.57; each extra prior
  episode of any kind adds only 1.13. So "injury-prone" is mostly body-part specific, and a broken thumb after a broken rib is only weakly predictive.
* **Same side.** Among players who have one episode of a part then another: 79% are on the SAME side (50% would be no effect): knee 82% (n=196), ankle 74%, hamstring 77%,
  calf 67%, foot 97%. Within six months of the previous episode it is 89%. Your hamstring-on-the-same-leg hunch holds.
* **Longer when it recurs on the same side:** median 13 game days out (mean 19.4) vs 8 (mean 14.2) for a first episode. Opposite-side recurrences are no longer (13.1 mean).
* Season level: last season's games missed predicts next season's only weakly (R-squared 0.05, MAE 15.0 vs 15.5 for "everyone misses 22"), and adding injury counts,
  repeat parts, or major injuries did NOT improve it (14.99). Injury history is worth a few games a year, not a big risk score, and not more than games missed already says.

## What this suggests building (not built yet)
1. **Plan availability**: replace the flat Questionable rate with the state table above (needs each listed player's last-game participation and whether he was
   listed the day before: previous day's report plus ESPN's per-player game log).
2. **Board injury risk**: flag players with a repeat SAME-side episode in the past two seasons (e.g. "left hamstring x3"), worth roughly +11 points of probability of another
   3+ game absence in that region and about 5 extra games when it happens (about 2-3 games a season); ignore unrelated injuries.
3. Do not weight injury type in day-to-day availability: the data says it does not help.
Caveats: episodes are defined from report listings (a long absence is one episode; a gap over 7 days starts a new one), body part comes from the printed reason, and 2025-26
11AM reports are thin (61 days). The recurrence and side results are descriptive, not causal.
