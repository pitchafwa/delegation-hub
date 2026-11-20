# Delegation League Fantasy Hub

Dynasty/keeper valuation for a 12-team ESPN points league (keepers cost nothing, only a roster slot):
Kalman-filter player trajectories, rookie/prospect projections, keeper VOR, a keeper-count-aware
asset value, and a breakout / bust / ADP-edge model with walk-forward backtests and a frozen
preseason prediction ledger.

**Live board:** https://delegation-hub.tommy-valtin.workers.dev (Cloudflare Worker delegation-hub; the old https://pitchafwa.github.io/delegation-hub/ still deploys as a backup)

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


## Design system (2026-11-19, following the Backlog design review)

All look-and-feel lives in the `<style>` block of `dashboard/index.html`; the last block ("Design system") holds the tokens and wins over older rules. Keep new UI inside these rules:

- **Type:** five sizes only: .75rem (labels, captions), .875rem (secondary), 1rem (body), 1.125rem (emphasis), 1.375rem (section titles, key numbers), plus large display numbers (scores, win chance). Secondary text uses `--ink-soft` / `--ink-faint` rather than smaller sizes.
- **Spacing:** a 4px scale (`--sp1`..`--sp6` = 4, 8, 12, 16, 24, 32px). Group with space and background shades, not borders.
- **Buttons, three levels:** `.btn.primary` / `.tcGo` (one main action per view), `.btn` / `.dbtn` tonal (secondary), `.btn.ghost` (low-stakes). Destructive confirms use `.btn.primary.danger`.
- **Chips are filters and toggles only** (`.seg` outside the nav, `.tcF`, `.chBtn`): separate pills, tonal accent when on. A selected filter never looks like a main button.
- **Page switchers are underline tabs:** the group's sub-tabs (`#navSub`) and the Trades tabs (`.tcTabs`), one row that scrolls sideways. The bottom bar (phones) / top bar keeps the 5 destinations.
- **More is a list page** (`renderMore`): rarely used pages live there; each sub-page shows a "More" back link instead of tabs.
- **Icons:** the outline SVG set in `ICONS`; use `ic('name')`. Old text glyphs (↓ ↑ ◀ ▶ ⓘ ×) in buttons are swapped for icons automatically (`hydrateIcons`), but new code should call `ic()`. No emoji as icons.
- **Header:** the data time plus a small refresh icon button (Refresh everything).
- **Confirms:** `await hubConfirm(title, body, okLabel, danger)`. Never `alert`, `confirm` or `prompt`.
- **Touch:** controls are at least 44px tall on phones (chips 40-44px).
- **Copy:** no explainer or method text on pages (`.legend` is hidden unless it is `.legend.keep`, used for live status or advice); the Glossary covers what numbers mean.
