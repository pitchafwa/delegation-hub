"""Central config for the ingest pipeline.

League settings (team count, roster slots, scoring) are read from the ESPN
API at runtime — only identity and paths live here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # fantasy-basketball-hub/
load_dotenv(ROOT / ".env")

LEAGUE_ID = int(os.getenv("LEAGUE_ID", "600271905"))
SEASON = int(os.getenv("SEASON", "2026"))

ESPN_S2 = os.getenv("ESPN_S2") or None
SWID = os.getenv("SWID") or None

# League rule, not an ESPN setting — ESPN's own draftSettings.keeperCount says
# 6, which is stale/wrong. Tommy confirmed the real rule directly: 3 keepers
# for the 2026 season (locked in), moving to 5 in future seasons (current
# plan, but subject to change). Always a live override, never hardcoded logic.
KEEPER_COUNT = int(os.getenv("KEEPER_COUNT", "3"))

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DATA_DIR = ROOT / "web" / "public" / "data"

# Be polite to ESPN.
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
