from pathlib import Path
from nba_api.stats.endpoints import leaguegamelog

ROOT = Path(__file__).resolve().parent
for season in ["2024-25", "2025-26"]:
    resp = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="P")
    df = resp.get_data_frames()[0]
    df.to_csv(ROOT / "data" / "game_logs" / f"nba_api_{season}.csv", index=False)
    print(season, df.shape)
