"""Freeze the preseason breakout predictions into an immutable ledger.

Run this at/after the 2026-27 season opener (Tue 2026-10-20, first tip 3pm ET). It snapshots
data/breakout_candidates.csv + breakout_validation.json (plus a hash of the model code) into
data/breakout_ledger/<season>.csv|.json. Once a ledger exists it is NEVER overwritten -- that's
the point: predictions can't be tweaked in hindsight, and the dashboard shows the frozen values.

  python freeze_breakout_ledger.py            # refuses before the opener
  python freeze_breakout_ledger.py --force    # override the date check (records that it was forced)

Grading (comparing frozen predictions to what actually happened) is grade_breakout_ledger.py.
"""
import hashlib
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
OPENER = date(2026, 10, 20)
SEASON = "2026-27"
LEDGER = D / "breakout_ledger"
LEDGER.mkdir(exist_ok=True)
csv_path, meta_path = LEDGER / f"{SEASON}.csv", LEDGER / f"{SEASON}.json"

if csv_path.exists():
    print(f"Ledger for {SEASON} is already frozen ({json.loads(meta_path.read_text())['frozen_at']}). Not overwriting.")
    sys.exit(0)
force = "--force" in sys.argv
if date.today() < OPENER and not force:
    print(f"Season opens {OPENER}; not freezing yet (today {date.today()}). Use --force to override.")
    sys.exit(1)

cand = pd.read_csv(D / "breakout_candidates.csv")
val = json.loads((D / "breakout_validation.json").read_text(encoding="utf-8"))
code_hash = hashlib.sha256((ROOT / "breakout_model.py").read_bytes()).hexdigest()[:16]
adp_written = datetime.fromtimestamp((D / "espn_adp.csv").stat().st_mtime).isoformat(timespec="seconds")
cand.to_csv(csv_path, index=False)
meta = {"season": SEASON, "frozen_at": datetime.now().isoformat(timespec="seconds"), "opener": str(OPENER),
        "forced_before_opener": bool(force and date.today() < OPENER), "rows": int(len(cand)),
        "model_code_sha256_16": code_hash, "adp_data_pulled": adp_written, "validation_at_freeze": val}
meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
shutil.copy(D / "breakout_candidates.csv", LEDGER / f"{SEASON}.candidates_at_freeze.csv")
print(f"FROZEN {len(cand)} predictions for {SEASON} at {meta['frozen_at']} (code {code_hash}, ADP pulled {adp_written})")
