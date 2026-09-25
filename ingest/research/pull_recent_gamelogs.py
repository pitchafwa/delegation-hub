"""Pull NBA game logs (stats.nba.com via nba_api) for the CURRENT season (default) or the seasons given on the command line,
e.g.  python pull_recent_gamelogs.py 2024-25 2025-26.  Past seasons are static, so the daily refresh only re-pulls the current one.
Writes data/game_logs/nba_api_<season>.csv (build_game_logs.py picks up every such file)."""
import sys
from datetime import date
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog

ROOT = Path(__file__).resolve().parent


def current_season(today=None):
    today = today or date.today()
    y = today.year if today.month >= 10 else today.year - 1
    return f"{y}-{str(y + 1)[2:]}"


seasons = sys.argv[1:] or [current_season()]
for season in seasons:
    resp = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="P", timeout=120)
    df = resp.get_data_frames()[0]
    if len(df) == 0:
        print(season, "no games yet; nothing written")
        continue
    df.to_csv(ROOT / "data" / "game_logs" / f"nba_api_{season}.csv", index=False)
    print(season, df.shape)
