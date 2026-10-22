# Streaming, using your moves, and roster flexibility (2026-10-22)

Question (Tommy): are we undervaluing moves? Should most matchups use several of the 7-8 allowed adds, even when that means dropping a 30 ppg player for a 25 ppg player to get an extra game played, given how uncertain season-long projections are?

## 1. What the outside advice says (web search; no one quantifies it)
* Schedule volume beats player quality in weekly formats: "five games from an average player are often going to beat out three games from a good player" ([theScore](https://www.thescore.com/news/1085822)). An idle roster day is costly; replacement-level production in an otherwise empty slot beats nothing (same source).
* Size the streaming effort to the move allowance: about 1-2 streaming spots for 2-3 weekly moves, more for 6-7. Drop fringe players with only 2-3 games that week or out of form; do not drop your best players, except in a final week where they have no games left ([Hashtag Basketball](https://hashtagbasketball.com/fantasy-basketball/content/how-to-stream-fantasy-basketball)).
* In head-to-head formats, pick up and drop weekly by how many games each player has ([ESPN](https://www.espn.co.uk/fantasy/basketball/fba/story/_/page/wtw120406)). One add can yield 3 games; 2 adds can yield 6 starts, so plan adds around days and back-to-backs ([CBS Sports](https://www.cbssports.com/fantasy/basketball/news/week-7-fantasy-basketball-waiver-wire/)); "nothing worse than picking up a player you can't fit into your lineup" ([AOL/Yahoo](https://www.aol.com/articles/fantasy-basketball-week-22-weekend-143300472.html)).
* Gaps: nothing found on how a games-played cap changes this, on points leagues, or on the dollar value of a move. The rest is our own data.

## 2. Our own data (this league, 2024-2026 seasons, 744 team-matchups)
* Starts matter and teams differ: league average 38.4 started games per 7-day matchup (cap 40); the five best-using teams average 42-43; Tommy's slot averaged 33.8 over its history (a different owner for much of it). Points per started game average 38, so every start you fail to make is expensive if a replacement can fill it at even 22-25.
* Adds and results move together across teams (36 team-seasons, correlation 0.79 with weekly points; teams with 100+ adds a year average 41-44 starts) but NOT within a team across years, so that is mostly "good, active owners", not proof that adds cause points. Treat as consistent with Tommy's view, not proof.
* Projection persistence (2010-2024, 2,687 established players, level = last season's points per game, realized = games 11-30 of the next season): slope 0.95. A 4-6 point level gap realizes as 4.2 points and the higher-level player outscores the other 65% of the time over six weeks. So for established players a 30 vs 25 gap is NOT mostly noise; Tommy's doubt about long-run accuracy does not hold for those players (it does hold for free agents and young players, already shrunk 40% in the plan).
* Getting a dropped player back: of players dropped after 4+ recent games, the share taken by ANOTHER team within 48 hours was 14-16% for players at 27-34 points per game (25% within 72 hours), 25% for 34+. Waiver data from 2024-26 gives about 18% claimed for a good starter.

## 3. Model results for a typical full week (Mon Oct 26 - Sun Nov 1, cap 40; same roster and free agents)
Baseline with no moves: 1806 expected points.
| Chance the dropped player is lost (never re-added) | Moves chosen | Expected gain this week |
|---|---|---|
| 25% (my old setting) | 0 | 0 |
| 18% | 0 | 0 |
| 15% (measured) | 3 | about +48 net of risk |
| 10% | 3 | about +64 |
| 0% | 5-7 | about +91 |
Gains concentrate on thin-slate days (Mon, Fri, Sun) when your starters have no game.

## 4. Verdict
Tommy is right on the part that decides it: (a) an add you do not use disappears at the end of the matchup, so its shadow price is zero and the per-move bar should be small; (b) the plan priced losing the dropped player too pessimistically (25% vs about 15% measured); (c) the right way to evaluate a rental is "this week's points gained minus the expected cost of not getting the dropped player back", not "swap a 30 for a 25 forever". Using 3 or more moves in most full weeks is correct (and worth roughly +50 points a week, about 1 extra win a season at a 250-point weekly spread).
He is not right that season-long projections are too noisy to justify caring: established players' gaps persist almost fully, so permanently swapping a 30 for a 25 loses about 4 ppg for real. The way streaming wins is rotation and reclaim, not a permanent downgrade.
The remaining real constraint: with 14 healthy players all above replacement, there is no free drop. Best practice (Hashtag) is to hold one or two deliberately fungible streaming spots; the cheapest candidates on this roster are Sharpe, Allen and Bey.

## 5. Changes made
RECLAIM_LOSE 0.25 -> 0.15; MIN_NET_GAIN 10 -> 6 (CHAIN_STEP_MIN stays 5). Open: log real re-add successes to refine 0.15; consider a "stream slot" roster-construction recommendation; model back-to-back and injury-report timing for rentals.
