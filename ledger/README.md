# Accountability ledger

Append-only records of what the site predicted and recommended, written automatically from opening night (Oct 19, 2026) so every model can be graded against what really happened.

| File | Written by | One record per |
|---|---|---|
| `plan-YYYY-MM.jsonl` | `build_week_plan.py` (overnight, ~3pm and ~6pm ET runs) | run: kind `plan` (expected points, win chance, suggested moves with timing, lineup, every roster player's status / availability state / level / usage boost / chance of playing, top free agents), kind `injury` (advisor verdicts, our estimated games out vs ESPN's date), kind `usage` (who was out and who was predicted to gain) |
| `alerts-YYYY-MM.jsonl` | `alerts.py` | notification sent |
| `live-YYYY-MM.jsonl` | `live_matchup.py` | 20-minute snapshot of the live win probability, so it can be checked against final results |

Grading (to be built): join these to ESPN's real results by date and player id: did the suggested adds gain what was predicted, did players listed Questionable play as often as `p_play` said,
did usage-flow beneficiaries gain, were the injury timelines right (our median vs ESPN's date vs reality), was the live win probability calibrated.
Only Tommy's team is logged in detail; nothing here is secret.
