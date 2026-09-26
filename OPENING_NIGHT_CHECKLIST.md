# Opening night checklist (season opens Tue Oct 20, 2026; first matchup is 6 days, Oct 20-25)

Claude: on or before Oct 20, flag this file to Tommy, walk through it, and say exactly what to look at. Tommy asked for this on 2026-09-26.

## A. Before Oct 20 (build or verify)
1. **Keeper deadline aftermath (Sun 9/27 5pm ET).** In ESPN: are all 12 teams' keepers recorded? Does the Keepers tab now say "ESPN shows the real keepers"? Did the commissioner fix the draft order (Tommy first in the first drafted round, then snake)? Re-pull draft data (`build_draft_data.py`) and re-check the Draft tab pick labels (Savion's pick, "Rd 1.1").
2. **Accountability ledger logging (not built yet).** Log every plan suggestion daily from opening night (moves, verdicts, win probability, injury estimates and ESPN's dates) so a ledger can grade them later. Build before Oct 20.
3. **Alerts timers.** cron-job.org jobs (check every 20 min 8:00-22:40; daily 8:30; weekly 9:30), all seven days, show HTTP 204 and a new "Phone alerts (ntfy)" run in GitHub Actions. `alerts/SETUP.md` has the steps.
4. **Breakout ledger freeze** happens automatically on Oct 20 (`freeze_breakout_ledger.py` via the daily task): confirm the file exists in `data/breakout_ledger/` afterwards.
5. **DFS projections.** Run `probe_dfs_projections.py` on a game day, read the saved HTML in `data/dfs_probe/`, write the DraftKings parser (league points = 1.206 x DK), log `dk_proj` next to ESPN's in `projection_log.csv`.

## B. What to look at on opening night and the first week (and what "good" looks like)
6. **ESPN's real matchup calendar.** Compare with our assumption (6-day opener, all other weeks 7 days except one 14-day All-Star matchup #17, playoffs Mar 8-28). Yahoo's calendar has a second 14-day matchup in Week 7 (NBA Cup, Nov 30-Dec 13) and playoffs Mar 15-Apr 4. If ESPN differs, update `LENGTHS` in `build_week_plan.py` and `build_schedule_plan.py`.
7. **Games cap and adds.** Cap 40 per 7-day week (about 34.3 for 6 days), checked at the start of each day; adds limit scaling (7 for 6 days? 8 for 7?); does ESPN's transaction counter (`matchupAcquisitionTotals`) match our "adds used"; do dropped players go to waivers for a day and clear Sunday (or daily)?
8. **Live matchup tracker.** The card appears at the top of This week (needs the alerts job to run once during a game day). Check: score matches ESPN within a few points, starts used matches the lineup, "who is still to play today" is right, win chance moves sensibly, the `live-data` branch exists in GitHub, and history builds through the day. Known unknowns: how ESPN reports a player mid-game, and ESPN's live field names.
9. **Lineup plan vs ESPN.** Tue lineup in the plan matches what you would set; the lineup-check alert fires only when something is wrong (Out player in the lineup, planned starter benched).
10. **Availability model.** Roster badges show "first day listed / played his last game / missed his last game" for Questionable players; afternoon refreshes (about 3pm and 6pm ET) run and the official report parses (`availability: N players listed` in the workflow log).
11. **Usage flow.** When a rotation player is ruled out, the "Injuries creating opportunity" box lists sensible beneficiaries and the beneficiary alert fires only when the plan itself recommends the add.
12. **Injured-player advisor.** ESPN's estimated return dates vs ours vs what happens; verdicts make sense (IR slot rules: does ESPN only allow OUT/IR-status players in IR? 4 slots).
13. **Automation health.** GitHub workflows (`refresh-league`, `refresh-gameday`, `alerts`, `pages`) green; local `FantasyHubDailyRefresh` task pushed on Oct 20 and 21; hub data and `schedule_plan.json` updating; the Pages site loads This week first and the nav shows Week / Players / League / About.
14. **Per-day projections.** Does ESPN publish per-game projections once the season starts (our level is ESPN season average blended with last 15 games)?

## C. Later (mid-season)
* Grade the availability, usage-flow and injury-return numbers against real results using the ledger.
* Replay the usage-flow add/drop decisions and the Phase 2 features on real 2026-27 weeks.
