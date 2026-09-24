from espn_api.basketball import League
import config

for year in range(2021, 2027):
    try:
        league = League(
            league_id=config.LEAGUE_ID,
            year=year,
            espn_s2=config.ESPN_S2,
            swid=config.SWID,
        )
        n_teams = len(league.teams)
        try:
            draft = league.draft
            n_picks = len(draft) if draft else 0
        except Exception as e:
            n_picks = f"ERROR: {e}"
        try:
            activity = league.recent_activity(size=5)
            n_activity = len(activity)
        except Exception as e:
            n_activity = f"ERROR: {e}"
        print(f"year={year}: teams={n_teams} draft_picks={n_picks} recent_activity_sample={n_activity}")
    except Exception as e:
        print(f"year={year}: FAILED to load — {e}")
