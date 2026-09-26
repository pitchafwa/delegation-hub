# Fantasy basketball strategy playbook for this league (2026-09-26)

League: ESPN, 12 teams, head-to-head POINTS (PTS 1, REB 1.5, AST 2, STL 3, BLK 3, 3PM +1, FTM +2, FTA -1, TOV -1, TD3 +3), 10 starting slots (PG SG SF PF C G F + 3 UTIL), 5 bench, 4 IR, daily lineups with a
per-game lock, about 8 adds per matchup, a 40-started-game cap per 7-day matchup, 6 of 12 make the playoffs, keepers 3 (2026) then 5.
Sources: outside articles (RotoWire, ESPN, Yahoo, Athlon, Fantrax, NBC, FantasyPros) plus our own measurements. Where an outside claim conflicts with our data, ours is noted.

## What decides a points league (ranked for THIS league)
1. Games started / volume (our 2025-26 data: correlation with weekly points 0.90; about 36 points per extra started game; top teams start about 43 games a week, bottom about 29).
2. Availability (injury and rest risk beat talent early; see the injury studies).
3. Schedule density (4-5 game weeks vs 2; playoff weeks 20-22).
4. Adds used well (about 8 a week; timing; streaming spots).
5. Keeper/dynasty value (5 keepers; age windows).
6. Trades (consolidation, contender/rebuilder windows, buying injured stars).
Scoring quirk: missed field goals cost nothing (only FTA and TOV are penalised), so volume scorers and rebound/steal/block producers gain, and efficiency specialists lose value relative to standard rankings.

## Strategy list, applicability, and where it lives on the site
| # | Strategy or tactic | H2H points? | On the site now | Suggested feature |
|---|---|---|---|---|
| 1 | Fill every starting slot every day; volume beats quality (a half-as-good player with twice the games matches a star) | Core | This week lineup plan (cap-aware DP), lineup-check alert | -- |
| 2 | Stay under the games cap until the last day, then use the overflow (up to 49) | Core (ours) | Cap logic in the plan | -- |
| 3 | Know schedule density; avoid 2-game weeks for stars; stack teams with 4-5 games; playoff weeks matter most | Core | Schedule tab: heat map, roster +/- games, playoff planner, streamers | Verify ESPN's real calendar in week 1 (below) |
| 4 | Keep 1-2 open bench spots for streamers; chain streamers across the week | Core | Suggested moves + add timing + budget rule | Streaming-slot manager: multi-day chain (Mon-Tue guy, then Wed-Thu guy, then weekend) within the add budget |
| 5 | Adds are a budget: do the urgent ones first, keep one in reserve | Core (ours) | Add budget rule, timing table | -- |
| 6 | Second night of a back-to-back: stars sit about 50% more (ours: 10.9% vs 6.1%) | Yes | p(play) factor in the plan | B2B and rest-risk icons on the Day table |
| 7 | Official injury report is the most reliable availability signal; participation matters more than injury type (ours) | Yes | Availability table, afternoon refreshes, status alerts | -- |
| 8 | Injury beneficiaries: backups and same-position teammates absorb usage; buy the beneficiary before the market | Yes | Usage-flow model in every projection, "Injuries creating opportunity", beneficiary alert | -- |
| 9 | Draft/hold availability over talent; injury-prone stars carry a discount | Yes | Five-tier injury risk and tags on the Board | -- |
| 10 | Stash vs drop injured players: top-100 talents are stashes, marginal players are droppable, IR spots are a farm system; watch minutes restrictions on return | Yes | IR moves box; absence-length curve | Injured-player advisor: stash/drop/trade-for by expected return date and post-return minutes ramp (needs a small return-ramp study) |
| 11 | Late season: tanking teams shut stars down; playoff-seeded teams rest stars; youth get minutes; buy motivated teams | Yes | Not modelled | Team-motivation flags from standings (tank, seed locked) feeding rest risk and youth-streamer boosts |
| 12 | Hot streaks: chase only if minutes/role changed; regression is real early | Yes (mild) | Buy-low/sell-high panel (weak evidence, said so) | "Why is he hot" split: minutes vs teammate-out boost vs shooting luck, plus FPPM and minutes-path columns on free-agent cards |
| 13 | Fantasy points per minute x projected minutes; path to playing time | Yes | Level and usage flow | Add FPPM and minutes trend (last 5 vs baseline) to add cards |
| 14 | Individual-game lock: keep late-tipping players in flexible slots until news arrives | Yes (ESPN) | Lineup-check alert 90 minutes before first tip | Late-swap planner: which slots to leave open for the 10pm games |
| 15 | Matchup awareness: underdogs want variance, favourites want safe floors; track the opponent's remaining games | Yes | Win probability in the plan | Live matchup tracker with swing-game and aggression advice |
| 16 | Trades: consolidate 2-for-1 when depth is surplus; stars are worth more than their sum | Yes | Star-premium package model in Trades | -- |
| 17 | Trades: buy injured stars at a discount (returns before the playoffs); contenders overpay at the deadline; rebuilders sell vets in season | Yes | Contender/rebuilder pricing in Trades | Injured-star buy list; trade-deadline calendar |
| 18 | Each manager values differently (age, rookies, injury appetite, window); anchoring on last season | Yes | Team draft profiles, partner-view pricing | -- |
| 19 | Draft picks are currency; sell in summer, buy at the deadline | Yes (we trade picks) | Draft-pick record only | Pick valuation in the trade finder (value of round/slot from ADP and our asset values) |
| 20 | Draft best player available; guards and usage first; positional scarcity is minor with 3 UTIL slots | Yes | Draft assistant (lineup fit + dynasty weights), % left | -- |
| 21 | Keeper window: contend with vets, rebuild with youth; youth are the currency; 22-24 is the sweet spot | Yes | Asset value (5 keepers), dynasty adds, age windows | Keeper decision tool (deadline Sun 9/27 5pm ET) |
| 22 | Late-round darts with easy drops; two-way players (50-game cap) and G League call-ups are volatile | Yes | Dynasty adds list | Two-way/call-up flag on free agents |
| 23 | Waiver mechanics: players dropped more than 24h ago go to waivers, first-come free agents are instant; Sunday processing here | Yes | Assumption noted | Just-dropped watch with waiver clear time |
| 24 | Position eligibility grows with games played (ESPN needs 10 games at a position) | Minor | Uses ESPN slots | Eligibility-gain tracker (small edge) |
| 25 | Playoff seeding: 6 of 12 qualify; points-for is the tiebreaker; win this week first, look ahead 2-3 weeks at most | Yes | Counterfactual replay | Playoff-odds and seed simulator |
| 26 | Do not rely on preseason ADP once games are played; update to current role | Yes | Daily model refresh, in-season blend | -- |
| 27 | Information speed: multiple outlets confirm; act before the market | Yes | Alerts (15-20 min), official report | -- |
| 28 | Category-league tactics: punting, category balancing, percentage protection, benching to protect FG%/FT% | NO in points | n/a | n/a |
| 29 | Matchup exploitation (defense vs position, pace, Vegas) | Weak | Tested: Vegas adds nothing; DvP is noise per others | none (skip) |
| 30 | Opponent scouting and an accountability ledger of our own recommendations | Yes | Not built | Opponent scouting; ledger that grades each suggestion |

## Calendar assumptions to verify (risk)
* RotoWire's 2026-27 analysis (Yahoo calendar) has Week 7 (Nov 30 - Dec 13, NBA Cup knockout rounds) as a single 14-day matchup, plus the All-Star 14-day matchup, with playoffs Mar 15 - Apr 4. Our plan assumes ESPN has only the All-Star
  14-day matchup and playoffs Mar 8-28. ESPN's real matchup calendar must be read when published; the 40-game cap scales with matchup length (80 for 14 days, which almost never binds), so a second 14-day week changes planning.
* NBA Cup knockout games are still shown as TBD in our schedule feed; Cup-advancing teams get extra games in that stretch.
