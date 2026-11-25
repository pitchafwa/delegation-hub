# Plan: trade deadline calendar + injured-star buy list

Status: PLAN ONLY (Tommy, 2026-11-25: "make a plan for this but don't implement yet"). Backlog item #3.

## What we know

- Trade deadline: **Fri Feb 19 2027, 12:00pm ET** (ESPN league setting `trade_deadline`). That is matchup week 16 (Feb 15-21), about 12 weeks from now.
- League trade review: 5 veto votes required. Not yet confirmed: whether a trade accepted shortly before the deadline still goes through if its review period ends after it. Check this on ESPN before building the countdown, because it sets the real last day.
- Already built and reusable:
  - The Windows tab already lists "a 38+ starter out" (capped at 5) and your players that fit.
  - The injury advisor gives an expected return, a return curve, and the chance he's back for the playoffs (`p_back_playoffs`). Today it only runs for your roster and free agents.
  - The trade engine prices any deal: league price, the acceptance model, and the full title-odds check.
  - Title weights: a playoff-week point is worth about 17 regular-season points to your title odds.
  - Owner posture: contending / bubble / out of the race.

## 1. Deadline calendar (Trades tab header + Windows)

- A countdown strip at the top of Trades: "Trade deadline Fri Feb 19, noon · 12 weeks". It turns urgent in the final 2 weeks.
- Below it, three phases, each with a short "what usually moves now":
  - **Now to early January: buy injured stars.** Prices are lowest right after an injury, and their owners are still contending and need production now.
  - **January: sellers appear.** Teams whose playoff odds fall under ~35% (odds history) become sellers. The Windows tab already tracks race movers.
  - **Feb 1-19: final window.** Last chance for playoff-schedule buys (title-week games) and for your own sell decisions.
- The deadline date comes from ESPN each morning (`league_settings` pull into `season_calendar.json`), so next season updates itself.

## 2. Injured-star buy list (a full list on Windows, replacing the capped injury rows)

For every injured player on another team whose healthy level is 35+:

| Column | Source |
|---|---|
| Expected return + chance back by the playoffs | Injury advisor, extended to all rostered injured players (cheap: same curve) |
| Playoff-week and title-week games after he's back | schedule_plan.json |
| Value to you | Expected playoff points x title weight, plus rest-of-season points x the regular weight. These are title-odds points, the same scale as the trade sheet. |
| What it should cost | The trade engine's best package for him (tcIdea), and his price now vs healthy (the "discount") |
| Will they sell? | Owner posture, whether their IR is full (roster crunch), and whether they need his position now |
| Your fit | Your players outside your top keepers at positions they're thin at (ownFits) |

- **Ranked by title-odds gain per unit of cost.** Each row opens the trade sheet with him as the target.
- **Built in** `build_trade_board.py` as `injured_buys`, daily, the same in the cloud and on the laptop.

## 3. Checks before building (research first)

1. **Return-curve accuracy for long absences (4+ weeks):** check the advisor's "back by the playoffs" calls on 2011-26 injuries, since the list leans on them.
2. **Are injured stars really cheap in this league?** There are only ~9 completed trades, so this can't be measured here. Use the league-price model's discount and be upfront that it's a model price, not a market record.
3. **A replay of 2025-26:** what the list would have ranked at three dates, and how those players did in the playoff weeks.

## 4. Questions for Tommy before building

- **Phone alert:** a single alert a few days before the deadline (only if a list target is still open), or site only? The default would be site only, per the actionable-alerts rule.
- **Season-ending injuries:** include them as keeper buys for next season, or this season only?
- **Your own injured players:** also show "sell" candidates (injured players on your team who won't be back for the playoffs), or buy side only?

## Rough size

- Calendar: small (half a session).
- Buy list: one session for the data and page, plus the three checks above.
