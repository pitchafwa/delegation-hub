# Why do experts rank top prospects above our model? (research notes, 2026-09-24)

Question (Tommy): experts have refined dynasty valuation for decades and would have been higher on top prospects than us in each recent
season. Why is there a gap, and should/how do we close it?

## What the literature/practice says (web research; some pages paywalled, a few details are from search summaries only)
1. **Horizon is long (~10 seasons).** Dynasty valuation is "a discounted sum over 10 years (this year plus nine future seasons)" in at least one
   published method; another describes "a multi-year horizon modeled across the remaining peak window"; RotoWire's blend is "current production,
   age and upside" with players under 24 "weighted heavily for upside". Ours stopped at 6 seasons (age 24-25 for a 19-year-old = before his prime).
   Sources: draftexpertpro.com/dynasty-auction-values, draftvalueanalytics.com/dynasty-draft-value, rotowire.com dynasty rankings.
2. **Production-based methods systematically under-read youth -- the practitioners say so.** FantasyVerdicts' methodology page: "Ascending young
   quarterbacks read low" / "Elite young tight ends read low" because redraft-style value prioritizes immediate impact; correcting needs
   dynasty-specific market data, not multipliers. Published expert/crowd dynasty rankings (Hashtag: 205k crowd votes; KeepTradeCut: crowd + real
   trade data; RotoWire) are MARKET/CONSENSUS prices, not production forecasts.
3. **Convexity at the top.** Dynasty trade calculators "weight elite assets more heavily, because the top of a dynasty roster is harder to replace
   than the middle" (affects trade math; a monotone transform does not change one-by-one rankings).
4. **Uncertainty is priced but not formalized.** "Half potentially leapfrogging into the top tier within two seasons, half stalling as role
   players." No source publishes an option-value formula.
5. **Situation matters for rookie timelines** ("a rookie with elite draft capital in a crowded room faces a depressed dynasty timeline";
   "productive struggle" arbitrage on bad teams) -- now in our model.
6. **Early-career studies stop before prime**: e.g. SportsEthos' first-round-pick study looks only at seasons 1-3 and stresses prospects' value is
   future peak production.

## What we tested with our own data
- Horizon 6 -> 10 seasons (empirical yearly change by age + survival by age for seasons 7-10): Boozer #44 -> #34 (d=0.92) -> #30 (0.96) -> #25 (1.0);
  Dybantsa #26 -> #20 -> #18 -> #16; Wilson #83 -> #65 -> #59 -> #54; Harper #85 -> #79 -> #72 -> #66 (5 keepers). Agreement with Hashtag crowd
  0.830 -> 0.839; with RotoWire top-30 0.747 -> 0.705 (no overall improvement). horizon_test.py
- CORRECTION (Tommy): comparing a dynasty rank with ROOKIE-YEAR production is the wrong target (a dynasty rank is a claim about future value).
  Fair check = does the value HOLD, i.e. how the same players are ranked later (expert_rank_drift_check.py):
  A) FantraxHQ 2023 preseason -> RotoWire Aug 2025: young/still-developing players' median rank change -9 (moved UP), 68% held or improved;
     established players +16 (moved down), 30% held. (Wembanyama 23->1, Amen Thompson 75->14, Ausar 87->41, Miller 90->39; Scoot 49->78.)
  B) RotoWire Aug 2025 -> Hashtag Sept 2026: young median +2 (flat), 52% held; established +17, 34% held. (Flagg 5->2, Harper 40->13,
     Edgecombe 55->33, Knueppel 100->37; Risacher 99->151.)
  => the market's own later verdict does NOT show youth premiums were too high; if anything young players gained ~25 spots on the established
  over two years. Caveats: same crowd re-ranking (self-referential); part of the drift is veterans aging; survivorship of the later list.
- Where we differ from the Hashtag crowd (expert_gap by age, ours minus crowd, within the matched set): <=20: +11 (K=19) / +17 (K=5) spots too low;
  21-22: +5 / +10; 23-24: -13 / -8 (we rank them too high); 25-30 ~0; 31+ mixed. Biggest gaps: Harper (crowd 11, ours 78 before horizon fix),
  Murray-Boyles, Mikel Brown, Wilson, Burries.
- A uniform youth premium fit on one outlet does NOT transfer (youth_premium_test.py): with the 10-season horizon, adding beta*(24-age)/4:
  agreement with Hashtag is flat (0.839 -> 0.841 at beta 0.1, falling after) and agreement with RotoWire falls steadily (0.70 -> 0.44 at beta 1.0).
  The remaining gap is PLAYER-specific (Boozer 15/15 vs Dybantsa 25/42 vs Peterson 34/38; outlets disagree on Harper 13 vs 40), not a uniform age effect.
- Convex/upside tilt (E[(v-c)^g]): small effect (Boozer #39 -> #35), tried and reverted.

## Conclusions
- Part of the gap is structural: experts price a 10-season horizon (ours was 6) and a market (which includes hype/resale), we forecast production.
- Horizon extension is standard, principled, and worth ~10 spots for the top teens; it does not close the gap alone.
- Remaining gap = player-specific scouting/market information (and outlets disagree with each other). Best handled by SHOWING the
  market rank next to our value (so the gap is visible and tradeable) and keeping the transparent override lever, not by hard-coding a premium.
