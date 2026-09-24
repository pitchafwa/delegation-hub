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
- Did experts' youth premium pay off? expert_premium_check.py.
  RotoWire preseason 2025-26 (Aug 2025), one-season outcome: players <=22 ranked on average 50.5, realized 57.3 (68% realized worse than ranked; e.g.
  Harper 39->89, Ausar Thompson 40->82, Bailey 51->87, Flagg 5->32 by 2025-26 fpg among the list; under-ranked: Knueppel 93->62, Castle 71->41).
  FantraxHQ preseason 2023-24, three-season outcome: young players ranked 57.1 vs realized 56.1 (fair); Wembanyama 23->5, Maxey 35->11, but
  Scoot Henderson 48->88, Jaden Ivey 45->92. Overall Spearman(expert rank, realized rank) 0.68-0.74.
  => No evidence experts' youth premium is systematically RIGHT on realized production; their individual calls are noisy. FantraxHQ had Wembanyama only #23.
- Convex/upside tilt (E[(v-c)^g]): small effect (Boozer #39 -> #35), tried and reverted.

## Conclusions
- Part of the gap is structural: experts price a 10-season horizon (ours was 6) and a market (which includes hype/resale), we forecast production.
- Horizon extension is standard, principled, and worth ~10 spots for the top teens; it does not close the gap alone.
- Remaining gap = market premium for youth/scouting info we cannot measure. Not shown to be profitable historically. Best handled by SHOWING the
  market rank next to our value (so the gap is visible and tradeable) and keeping the transparent override lever, not by hard-coding a premium.
