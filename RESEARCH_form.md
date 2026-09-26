# "Why is he hot?" (form split)

**Question.** When a player's last 10 games are far above (or below) his norm, how much of it will last, and does the *reason* matter?

**Data.** Every player-game 2010-11 to 2025-26 (404k games; full box scores: minutes, FGA, 3PA, FTA, rebounds, assists, steals, blocks, turnovers). Panel of 38,211 checkpoints (a checkpoint every 5 games for every player with 30+ baseline games): window = last 10 games, baseline = his games in the previous 365 days, outcome = his fantasy points per game (this league's scoring) over the next 12 games in the same season.

**The split** (adds up exactly to recent minus baseline points per game):
* minutes: (minutes change) x baseline points per minute
* shooting luck: makes vs his own shrunk baseline percentages, by shot type (3-pt, 2-pt, free throw)
* shot volume: change in attempts per minute, valued at his baseline percentages
* rebounds, assists, steals+blocks, turnovers/other: change in those per-minute rates

**How much of each lasts** (fit 2010-2024, tested on 2024-25 and 2025-26): minutes 0.69, assists 0.57, shot volume 0.60, rebounds 0.38, steals+blocks 0.30, turnovers/other 0.21, 2-pt luck 0.30, free-throw luck 0.24, 3-pt luck 0.18. (A plain "blend" keeps 0.61 of everything.) Also: last-3-game points +0.05 and last-3 minutes +0.16 (small extra signal), young players (<24) +0.58/yr, players over 31 -0.38/yr.

**Out-of-sample error** (points per game, next 12 games): no form 7.10 RMSE; best plain blend 5.70; split model 5.55 (2.8% lower; 4.33 vs 4.45 MAE), better in both test seasons separately (5.59 vs 5.73 and 5.51 vs 5.67). Predicted vs actual change is well calibrated across 8 buckets (predicted -6.3 -> actual -5.7 ... predicted +8.4 -> actual +8.9). A gradient-boosted model with more features did no better (5.56), so the plain linear split is what ships. Results do not depend on the shrinkage used for baseline percentages (0.4x to 2.5x: same coefficients and error).

**Findings worth knowing.**
* Minutes changes are the real thing (69% lasts); shooting luck mostly fades (18-30%).
* Minutes gains that came while teammates were out persisted as well as other minutes gains (69% vs 61%), and the absent-teammate minutes were a weak signal in aggregate (0.06 points per absent minute), so the usage-flow model, not this study, remains the tool for "who benefits from an absence".
* The gain from splitting is modest: a typical player's forecast moves under a point; it matters for the tails (a player +8 on 3-pt luck, a role player whose minutes doubled).

**What ships.**
* `build_form_split.py` (daily local refresh, in season) -> `dashboard/form_split.json`. The weekly plan ignores it if older than 4 days or from another season.
* Plan level adjustment: `keep` (the split's expected lasting change) minus what the level already credits (`RECENT_FORM_MAX` x games/15 x change), shrunk 30% and capped +-3 points a game. Age is left to ESPN's projection.
* UI: a "Form (L10)" column on the roster (minutes and points per minute vs baseline), and a one-line reason under a player's name in suggested moves, the long-term adds and the roster when he is 3+ points off baseline.
* Ledger logs `level_pre`, `form_d` and `form_keep` next to `level`, so in season we can check whether the adjustment beat the unadjusted level.

**Caveats.** Rookies and anyone with under 30 baseline games (or under 10 games this season) get no split. The adjustment is validated against a trailing-average baseline, not against ESPN's in-season projection (no history of it), so it is deliberately shrunk and capped; grade it with the ledger after a month.
