# Archive

Superseded one-off scripts, kept for reference rather than deleted. None of these are imported or run by anything live (`refresh_all.py`, the GitHub Actions workflows, or any other script) as of 2026-09-29.

- `injury_risk_flag.py`, `injury_risk_flag_v2.py`, `injury_risk_flag_v3.py` — early single-team injury-risk flagging; superseded by the generalized, all-players version now built directly in `build_hub_data.py`.
- `build_dataset.py`, `build_dataset_v2.py` — early dataset-building experiments, fully superseded, no other script ever referenced them.

This is a separate, much smaller list than the ~100 files in `ingest/research/` that aren't referenced elsewhere — most of those are legitimate standalone research scripts (fit-and-report one-off studies) or form a genuine, still-interconnected chain of rookie-model iteration experiments (`fit_rookie_model_*.py`, `fit_ensemble_*.py`, `seed_check_*.py`, etc.) that reference each other even though nothing in production calls them. Those were deliberately left alone rather than archived, since moving pieces of an interconnected chain risks breaking the ability to re-run any of it later. Only files with zero inbound references from anywhere (including comments) landed here.
