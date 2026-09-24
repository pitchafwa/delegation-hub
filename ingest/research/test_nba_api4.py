from nba_api.stats.endpoints import playerindex
resp = playerindex.PlayerIndex(season="2023-24", historical_nullable="1")
df = resp.get_data_frames()[0]
print(df.shape)
print(df[["PERSON_ID","PLAYER_FIRST_NAME","PLAYER_LAST_NAME","POSITION","HEIGHT","WEIGHT","DRAFT_YEAR","DRAFT_ROUND","DRAFT_NUMBER","FROM_YEAR","TO_YEAR"]].head(5))
