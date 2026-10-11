# Trade value research and proposed model (2026-10-11)

Goal: a trade ANALYZER (enter any trade, get a verdict for both sides) built on a trade-VALUES model. Draft-pick values come second and plug into the same model.
Status: research done, model PROPOSED, nothing built yet. The existing Trades tab is a finder (`build_trades.py`, v1, never backtested), not an analyzer.

## 1. What the outside world does (and where it falls short)

| Approach | How values are made | Weakness found |
|---|---|---|
| KeepTradeCut | Crowd votes ("keep/trade/cut"), then a hidden per-player "raw adjustment" (10-42% of value, convex in value, depends on the best player in the trade) is added to the side giving more bodies; fair band 5%. Community reverse-engineering: [Javelin](https://www.javelinfantasyfootball.com/2022/09/30/how-the-ktc-adjustment/); KTC's own [FAQ](https://keeptradecut.com/frequently-asked-questions) only says it accounts for roster spots and "stud" factor. | Throw-ins add adjustment, so padding a trade distorts it; very high values are nearly unbuyable. |
| FantasyCalc | Values solved from millions of completed trades (convex optimization, recent trades weighted more, outliers removed, a fixed waiver credit for the side receiving fewer players). [About](https://fantasycalc.com/about) / [FAQ](https://fantasycalc.com/frequently-asked-questions). | Needs huge trade volume. |
| RosterAudit | Elo-style engine on real completed trades. [About](https://rosteraudit.com/about/). | Same: needs volume. |
| DynastyProcess | Expert rank consensus through an exponential decay curve. [Values](https://dynastyprocess.com/values/). | Pure market ranks, no roster context. |
| Roster-aware analyzers | Compare each team's best lineup before and after for every remaining week; charge the cut player for the 2-for-1 roster spot; some run 1,000-10,000 season simulations to show playoff/title odds before and after (Smash Accept, Dynasty Daddy). [RationalGo](https://rationalgo.ai/resources/app-builder/community-tools/fantasy-football-trade-analyzer-build-free), [FantasyPros](https://www.fantasypros.com/nfl/myplaybook/trade-analyzer.php). | Mostly vendor claims; formulas unpublished. |

Evidence worth keeping:
* A test of calculators on 3,003 community-judged dynasty trades ([Stats Guy](https://statsguyfantasy.com/methodology/comparing-trade-calculators), written by a competitor, so weigh it) found values learned from real trades named the community's winner 82-85% of the time, KTC only 57%. The gap is almost all in package (several-for-one) trades. Correction layers (KTC's adjustment, FantasyCalc's waiver credit) lifted raw sums from 43% / 61% to 57% / 82%. Nobody handled contender-vs-rebuilder or picks from specific teams well.
* The 2-for-1 discount rule of thumb (15-20% off every asset after the best) is a vendor assumption, not a measured number. The mechanism is real: the extra player forces a cut, so he is worth roughly his value OVER replacement minus the value of the player you must cut. In a value-over-replacement framework it needs no fudge factor.
* Contender/rebuilder is usually a toggle (age factor, pick weights). The more rigorous version is to simulate playoff and title odds before and after the trade for BOTH teams (my synthesis; no source publishes a formula).
* Behavioral: the endowment effect is real in sports trading. A 2022 [Economic Inquiry study](https://onlinelibrary.wiley.com/doi/full/10.1111/ecin.13102) of pro draft picks found traded picks roughly 10% more likely to trade again than matched picks (the original owner values his asset about that much more). Fantasy-specific acceptance models do not exist in the literature I found; the mechanism is contested (loss aversion vs asymmetric information), so treat ~10% as a prior to test, not a constant.
* Pick values in dynasty guides: roughly 15-20% off per year of distance, and the sending team's projected finish and the class strength matter (vendor rules of thumb, unvalidated). This belongs to part 2.

## 2. What is different about THIS league (why off-the-shelf values would mislead)

* H2H points with a games cap: 40 started games per 7-day matchup, checked at the start of each day. Depth beyond ~10 playing starters has little or no value; a player's worth depends on which days he plays and on the cap, not just his per-game level.
* Instant free agents (see waiver work 2026-10-10): the replacement level is the 3rd-best free agent (about 27 pts/g shrunk, 30+ raw), and 8 adds a week make bench depth cheap. So "2 for 1" costs less than in a no-streaming league, and depth players are worth about zero.
* Keepers cost nothing, only a roster slot: next year's value is the sum of each team's top 5 assets (3 this year). A star who would be a team's 7th asset is worth little to that team as a keeper; the same star is a top-3 keeper elsewhere. That gap is where win-win trades live (the v1 finder's core idea, which is sound).
* Trade volume is tiny: last season 35 proposals, 21 declined, 24 cancelled, 11 never resolved, and roughly 3-8 accepted (ESPN's history shows only a fraction of the accepted trades with their players). There is no trade history to learn values from the FantasyCalc way.
* But there IS a league-specific price history: six drafts. The availability model (2026-10-04) already estimates how THIS league prices players, including the rookie premium (rookies went around picks 1-9 for the top NBA picks since 2023). That is revealed preference and the best proxy we have for how these owners value players.

## 3. Proposed model: "value to you" and "will they accept" are two different questions

### 3a. Value to a team = change in that team's own chances, not a sum of player prices
For team T and a trade X (players and, later, picks in and out):

  V(T, X) = ΔTitle%(T) x PRIZE  +  ΔFuture(T)

* **Season part (ΔTitle, ΔMade):** rebuild T's best lineups for every remaining week with the planner we already have (`plan_team`: 10 slots with eligibility, the 40-game cap, availability, schedule, per-day opponent adjustments), with the roster after the trade, forced drops chosen to lose the least, and free agents as replacement. Then convert weekly point changes into playoff and title odds with the simulator we already have (`playoff_sim.py`; it ran 20,000 seasons). From today's odds file: DRNK is 80% to make the playoffs and 26% for the title; adding 25 points a week raises that to 85% and 31%, so about +0.19 percentage points of title chance per point per week. Playoff weeks (20-22) count more than regular weeks automatically, because the simulator knows they decide the title.
* **Why odds, not points:** points are not what you want. A contender and a team already out of it value the same 10 points differently, and the simulator captures that. "Contender vs rebuilder" falls out of the model instead of a toggle (a manual override slider stays).
* **Future part (ΔFuture):** the change in next year's top-5 keeper assets (and the 10-year asset path, discounted 0.95) from our DELCO asset model, converted to the same units with the same odds sensitivity per asset point (1 asset point = 1 point per game for a season = about 73 points). Default future weight stays adjustable (the keeper slider already on the site).
* **Roster and risk effects come for free:** a 2-for-1 forces a drop whose value over replacement is lost; IR use, add-budget use and bench cover (the new `starts_cover` logic) are in the lineup rebuild. Injury risk is already inside availability (injury tiers).
* **Consolidation needs no fudge factor:** because both sides are measured through a lineup, two 40s for one 55 loses exactly what the lineup says it loses.

### 3b. Will the other owner say yes? A separate "perceived value" model
Owners do not run a lineup optimizer; they react to market-style value. So acceptance is modeled on perceived value, not on the true model:
* Perceived value of a player = a blend of the league-specific draft price (our availability model's expected pick, with the rookie premium), Hashtag dynasty rank and ESPN ADP; value is a steeply convex curve in rank above a replaceable player (Hashtag's own curve: rank 1 ~2530, rank 100 ~1336, rank 430 ~998 on a baseline of 1000). The subtraction of replacement is what makes two bench guys worth ~0 for one star.
* Acceptance score = perceived gain for them, with an endowment margin (prior ~10% from the literature: they need to receive more than they give up) and adjustments from the owner profile we already have (who loves rookies, who gambles on injuries). Shown as bands (unlikely / possible / likely), not a fake-precise percentage, because the data to fit it is thin (about 35 proposals a season).
* Their TRUE gain (the same model run for them) is shown too. The best trades are those good for you in the model and acceptable in perceived value: the gap between "what they think" and "what it does for them".

### 3c. What the analyzer shows
You pick a partner, add players (and later picks) on each side. It shows, for you and for them: change in playoff odds, title odds, expected season points, next year's keeper strength, who gets dropped, the lineup before and after; the perceived-value balance and acceptance band; a verdict (win-win / good for you but likely refused / fair / bad for you) and a suggested balancing throw-in from their roster that helps acceptance at the least cost to you. The finder (v2) reuses the same engine to search.

## 4. Validation (the model must earn its place)
1. **Counterfactual replay (the season part):** at past checkpoints (weeks 5/10/15 of 2024, 2025 and 2026) take the actual rosters (we have every team's daily lineups for all three seasons), apply hypothetical trades, and compare predicted gain with realized points over the next 8 weeks. Baselines to beat: sum of market values; sum of per-game levels. Metrics: rank correlation and sign accuracy (the same "named the right winner" test the calculator comparison used).
2. **Acceptance:** fit the endowment margin and noise on every proposal we can reconstruct from 2024-2026 (accepted vs declined), report AUC. Expect weak identification; keep bands.
3. **Case studies:** the Tatum-for-Quickley trade last October and the vetoed trade, and spot checks you choose.
4. **Honest limits to state in the tool:** projections are the inputs (same for both sides); the keeper side is the least validated; acceptance is a prior.

## 5. Build plan
* Phase A: `trade_value.py` engine (lineup rebuild + odds sensitivity + asset change + forced-drop logic) with the validation scripts. 
* Phase B: the analyzer. Interactive evaluation of arbitrary trades needs the Python engine, so run it like the lineup tool: a button dispatches a GitHub workflow (about a minute, works on the phone with your token) with an instant rough estimate in the browser from precomputed per-player marginals. 
* Phase C: draft-pick values (pick -> expected asset of the player drafted there, from this league's draft pricing and the incoming class strength, discounted by distance and by the sender's projected finish), plugged into 3a. Then finder v2.

## 6. Decisions needed before building
1. Objective: title/playoff odds as the primary currency (recommended), or season points?
2. Analyzer runs: workflow (exact, ~1 min) plus instant estimate (recommended), or a simplified in-browser model only?
3. Contender/rebuilder: automatic from the simulator with a manual override (recommended)?
4. Perceived value source: league draft pricing + Hashtag + ADP blend (recommended).
5. Acceptance shown as bands, not percentages (recommended).
