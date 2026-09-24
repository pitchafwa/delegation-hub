# Delegation League Fantasy Hub

Dynasty/keeper valuation for a 12-team ESPN points league (keepers cost nothing, only a roster slot):
Kalman-filter player trajectories, rookie/prospect projections, keeper VOR, a keeper-count-aware
asset value, and a breakout / bust / ADP-edge model with walk-forward backtests and a frozen
preseason prediction ledger.

**Live board:** https://pitchafwa.github.io/delegation-hub/

## Layout
- `dashboard/` - the static site (`index.html` + `hub_data.json` + `breakout_history.json`). Deployed to GitHub Pages by `.github/workflows/pages.yml` on every push that touches it.
- `ingest/research/` - all model/pipeline code (Kalman engine, breakout models, asset value, freeze/grade scripts).
- `BACKLOG.md` - findings, caveats, and the approved-one-at-a-time to-do list.

## Not in this repo (by design)
Raw and third-party data pulls (`ingest/research/data/*`: game logs, college/Torvik data, ESPN pulls) and
secrets (`.env` with ESPN cookies). Regenerate data with the `pull_*` / `prep_*` scripts; the pipeline needs
a local `.env` with `LEAGUE_ID`, `ESPN_S2`, `SWID` (see `ingest/config.py`).

## Rebuild the site
```
cd ingest
uv run python research/build_hub_data.py   # writes dashboard/hub_data.json + breakout_history.json
```
then commit and push; Pages redeploys automatically.

## Preseason ledger
`ingest/research/freeze_breakout_ledger.py` locks the breakout predictions at the 2026-10-20 season opener into
`ingest/research/data/breakout_ledger/`. Git history is the proof they weren't edited afterwards.
