import pandas as pd

def build_features(df_teams):
    # ensure dates are datetime
    df_teams['date'] = pd.to_datetime(df_teams['date'])
    
    # -----FEATURE ENGINEERING-----
    # sort chronologically
    df_teams = df_teams.sort_values(['teamname', 'date'])
    
    # -----ELO CALCULATION-----
    START_ELO = 1500
    K = 32
    
    elo_ratings = {} 
    pre_game_elo = pd.Series(index=df_teams.index, dtype=float)
    df_teams = df_teams.sort_values('date')

    # process strictly in chronological order, one GAME (2 rows) at a time
    for gameid, game_rows in df_teams.sort_values('date').groupby('gameid', sort=False):
        if len(game_rows) != 2:
            continue  # skip malformed games missing a side
    
        row_a = game_rows.iloc[0]
        row_b = game_rows.iloc[1]
    
        elo_a = elo_ratings.get(row_a['teamname'], START_ELO)
        elo_b = elo_ratings.get(row_b['teamname'], START_ELO)
    
        # record PRE-GAME elo for both rows — this is what becomes the feature
        pre_game_elo[row_a.name] = elo_a
        pre_game_elo[row_b.name] = elo_b
    
        # standard Elo expected score
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        result_a = row_a['result']  # 1 if team A won, 0 otherwise
    
        # update AFTER recording pre-game values
        rating_change = K * (result_a - expected_a)
        elo_ratings[row_a['teamname']] = elo_a + rating_change
        elo_ratings[row_b['teamname']] = elo_b - rating_change
    
    df_teams['elo'] = pre_game_elo.round(2)
    
    # opponent Elo
    self_and_opp = df_teams[['gameid', 'teamname', 'elo']].merge(
        df_teams[['gameid', 'teamname', 'elo']],
        on='gameid', suffixes=('', '_opp'))
    self_and_opp = self_and_opp[self_and_opp['teamname'] != self_and_opp['teamname_opp']]
    opp_elo_map = self_and_opp.set_index(['gameid', 'teamname'])['elo_opp']
 
    df_teams['opp_elo'] = df_teams.set_index(['gameid', 'teamname']).index.map(opp_elo_map).values
    
    # -----DATA CLEANING-----
    df_teams = df_teams.sort_values(['teamname', 'date'])

    df_teams['gold_per_min'] = df_teams['totalgold'] / df_teams['gamelength']
    # rolling features (.shift(1) makes sure current game's outcome is not included)
    ROLLING_COLS = [
        'gold_per_min',
        'dragons',
        'barons',
        'towers',
        'kills',
        'deaths',
        'assists',
        'ckpm',
        'opp_elo']
 
    df_teams['last_5_winrate'] = (
        df_teams.groupby('teamname')['result'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
 
    for col in ROLLING_COLS:
        df_teams[f'last_5_{col}'] = (
            df_teams.groupby('teamname')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
 
    # split by side
    df_blue = df_teams[df_teams['side'] == 'Blue'].copy()
    df_red = df_teams[df_teams['side'] == 'Red'].copy()
    
    rolling_feature_names = ['last_5_winrate'] + [f'last_5_{col}' for col in ROLLING_COLS]

    useful_features = [
        'teamname',
        'result',
        'elo'] + rolling_feature_names
    
    match_metadata = [
        'gameid',
        'date',
        'league']
    
    # merge blue and red teams into one match row
    df_merged = pd.merge(
        df_blue[match_metadata + useful_features],
        df_red[['gameid'] + useful_features],
        on='gameid',
        how='inner',
        suffixes=('_blue','_red'))
    
    # rename target variable and drop redundant column
    df_merged = df_merged.rename(columns={'result_blue': 'blue_win'})
    df_merged = df_merged.drop(columns=['result_red'])
    
    # drop rows with missing critical values
    dropna_cols = (
        [f'{name}_blue' for name in rolling_feature_names] +
        [f'{name}_red' for name in rolling_feature_names] +
        ['elo_blue', 'elo_red'])
    df_merged = df_merged.dropna(subset=dropna_cols)
    
    # delta features (modified so that +ve means Blue advantage, -ve means Red advantage)
    # the more the better
    df_merged['winrate_delta'] = df_merged['last_5_winrate_blue'] - df_merged['last_5_winrate_red']
    df_merged['gold_per_min_delta'] = df_merged['last_5_gold_per_min_blue'] - df_merged['last_5_gold_per_min_red']
    df_merged['dragons_delta'] = df_merged['last_5_dragons_blue'] - df_merged['last_5_dragons_red']
    df_merged['barons_delta'] = df_merged['last_5_barons_blue'] - df_merged['last_5_barons_red']
    df_merged['towers_delta'] = df_merged['last_5_towers_blue'] - df_merged['last_5_towers_red']
    df_merged['kills_delta'] = df_merged['last_5_kills_blue'] - df_merged['last_5_kills_red']
    df_merged['assists_delta'] = df_merged['last_5_assists_blue'] - df_merged['last_5_assists_red']
    df_merged['elo_delta'] = df_merged['elo_blue'] - df_merged['elo_red']
    df_merged['opp_elo_delta'] = df_merged['last_5_opp_elo_blue'] - df_merged['last_5_opp_elo_red']
    df_merged['ckpm_delta'] = df_merged['last_5_ckpm_blue'] - df_merged['last_5_ckpm_red']
    # the less the better
    df_merged['deaths_delta'] = df_merged['last_5_deaths_red'] - df_merged['last_5_deaths_blue']

    # safety net
    df_merged = df_merged.sort_values('date').reset_index(drop=True)
    
    return df_merged
