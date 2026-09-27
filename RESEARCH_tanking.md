# Team motivation and player rest: preliminary research (2026-09-27)

**Why.** Team-motivation flags were shelved for this season because the new lottery rules punish bottom-3 finishes (a "relegation zone"), unlike 2019-2026 where the worst record barely mattered (top-3 odds were flattened in 2019). Before deciding whether/how to flag it in-season, this establishes a historical baseline: how much do teams actually rest star/rotation players down the stretch depending on where they sit in the standings, and did the 2019 reform change that. Tommy's caveat (not yet testable — no pick-ownership-history source found): a team that has traded away its own first-round pick has no lottery incentive at all, whatever its record.

**Data.** Team game logs (nba_api `leaguegamelog`, team level) for every regular season 2010-11 to 2025-26 (`pull_team_gamelogs.py`); teams ranked 1 (worst record) to 30 (best) by final season win%. Player game logs from the "why is he hot" dataset (404k player-games). Rotation players only: 20+ mpg and 40+ games that season (about 3,200 player-seasons). `tank_study.py`.

**Method.** For each qualifying player-season, compare his **last 20 games** to the rest of the season on: minutes per game he actually played, fantasy points per game he played, and — separately and more importantly — **availability**, his share of his TEAM's last 20 games actually played (using the team's real schedule, so games he sat out count). Grouped by final-standings bucket (1-3 / 4-8 / 9-16 / 17-30) and era (pre-2019: 2010-11..2018-19; post-2019: 2019-20..2025-26). The lockout-shortened 2011-12 and the covid-disrupted 2019-20/2020-21 seasons are reported separately, not folded into the headline numbers.

## Findings

**1. Per-game rate barely drops. Availability does.** Bottom-standings rotation players who DO play in the last 20 games are not sandbagged — their minutes and fantasy points per game are flat to slightly UP late in the season (bottom-3 pre-2019: minutes +1.2/g, +5.8%; post-2019: +1.5/g, +7.3%). The real mechanism is games missed, not reduced role: bottom-3 teams' rotation players played about 84-86% of the team's early-season games but only 62-67% of the last 20 (a 17-20 point drop), versus a 5-6 point drop for teams in the top standings tier. **Tanking shows up as rest/DNPs, not as quietly bad stat lines.**

**2. The gap is bigger than the league-wide "load management" trend, and it widened after 2019 — for the 4-8 tier specifically, not bottom-3.** Subtracting the top-tier (contending team) drop as a league-wide baseline, the EXCESS availability drop for bottom-standings players:
| Bucket | Pre-2019 excess | Post-2019 excess |
|---|---|---|
| 1-3 (bottom) | -12 pts | -14 pts |
| 4-8 | -11 pts | -20 pts |
| 9-16 | -5 pts | -6 pts |

Bottom-3 barely changed. The 4-8 bucket — teams still bound for the lottery but with no shot at (and no reason to specifically chase) the very worst record — shows a much bigger excess drop after the reform than before.

**3. Stars specifically drive that 4-8 shift.** Splitting into stars (each team's top-2 by total minutes) and role players (rest of the rotation): bottom-3 stars are NOT rested unusually more than average (availability drop -7.5 to -8.5 points, both eras — barely different from a top-standings star's -5 to -6 points). But stars on 4-8 teams went from a -9.7 point drop pre-2019 to a **-30.8 point drop post-2019** (n=40, small sample — flagged, not a firm number). Role players show the same direction, smaller (-18.9 → -24.0 points).

**Read (a hypothesis, not a proven cause):** flattening the very-worst-record odds in 2019 removed the incentive to fight for the single worst record, but did nothing to reduce — and this data suggests it may have spread — the incentive to be anywhere in the lottery rather than sneak into a meaningless late playoff/play-in spot. Tanking behavior looks like it migrated from "be the worst" to "be somewhere in the bottom half," concentrated in star rest.

**Robustness / caveats.**
* Covid (2019-20 bubble cutoff, 2020-21 72-game) and the 2011-12 lockout season show the same shape but are excluded from the headline numbers (weird schedules and, for the bubble, a hard stop that mechanically ended "last 20 games" early for non-bubble teams).
* No causal claim — this is standings correlated with rest, not proof the reform caused the shift; league-wide load-management culture also grew over this period (captured partly, not fully, by the top-tier baseline).
* The rotation cutoff (20+ mpg, 40+ GP) drops players who were rested out of the rotation entirely or traded away mid-tank, which would understate the true effect.
* Pick-ownership (a team resting a good player for a pick that goes to someone else, or NOT resting because their own pick is protected/owed out) is not modelled — no historical source found yet. If a "team motivation" flag is built in-season, this is the first thing to add if a data source turns up (Real GM / Basketball-Reference trade pages have pick ownership; not yet pulled).
* This only covers 2010-2025. Nothing here is compared against the true PRE-2019 lottery system further back (same odds structure existed back to 1994, so 2010-2018 already reflects it) — no pre-2010 data was pulled.

**What this is for.** A 2026-27-season baseline: once real games are played under the new bottom-3-punished rules, compare this season's late-stretch availability drop by bucket against these historical numbers (especially the bottom-3 vs 4-8 gap) to see whether bottom-3 teams now fight harder to avoid the relegation zone (a NARROWING or reversal of the bottom-3-vs-4-8 gap), and whether 4-8-style "quiet tanking" continues unchanged. Scripts and panels: `ingest/research/tank_study.py`, `ingest/research/pull_team_gamelogs.py`, `ingest/research/data/tank/`.
