import pandas as pd
def get_latest_stats(teamname, df_processed):
    # only rows of that team
    df_team = df_processed[(df_processed['teamname_blue'] == teamname) | (df_processed['teamname_red'] == teamname)].copy()
    
    if df_team.empty:
        raise ValueError(f"Team '{teamname}' not found in the dataset.")

    # absolute last row (most up-to-date stats)
    df_team = df_team.sort_values(by='date')
    latest_match = df_team.iloc[-1]

    if latest_match['teamname_blue'] == teamname:
        stats = {
            'teamname': teamname,
            'elo': latest_match['elo_blue'],
            'winrate': latest_match['last_5_winrate_blue'],
            'gold_per_min': latest_match['last_5_gold_per_min_blue'],
            'dragons': latest_match['last_5_dragons_blue'],
            'barons': latest_match['last_5_barons_blue'],
            'kills': latest_match['last_5_kills_blue'],
            'deaths': latest_match['last_5_deaths_blue'],
            'ckpm': latest_match['last_5_ckpm_blue'],
            'opp_elo': latest_match['last_5_opp_elo_blue']}
    else:
        stats = {
            'teamname': teamname,
            'elo': latest_match['elo_red'],
            'winrate': latest_match['last_5_winrate_red'],
            'gold_per_min': latest_match['last_5_gold_per_min_red'],
            'dragons': latest_match['last_5_dragons_red'],
            'barons': latest_match['last_5_barons_red'],
            'kills': latest_match['last_5_kills_red'],
            'deaths': latest_match['last_5_deaths_red'],
            'ckpm': latest_match['last_5_ckpm_red'],
            'opp_elo': latest_match['last_5_opp_elo_red']}

    return stats


def predict_any_matchup(blue_team_stats, red_team_stats, model, X_train, scaler, threshold=0.50):
    if blue_team_stats['teamname'] == red_team_stats['teamname']:
        raise ValueError("Both teams cannot be the same. Please provide two different teams for prediction.")

    # blue minus red
    match_features = {
        'winrate_delta': blue_team_stats['winrate'] - red_team_stats['winrate'],
        'gold_per_min_delta': blue_team_stats['gold_per_min'] - red_team_stats['gold_per_min'],
        'dragons_delta': blue_team_stats['dragons'] - red_team_stats['dragons'],
        'barons_delta': blue_team_stats['barons'] - red_team_stats['barons'],
        'kills_delta': blue_team_stats['kills'] - red_team_stats['kills'],
        'deaths_delta': red_team_stats['deaths'] - blue_team_stats['deaths'],
        'ckpm_delta': blue_team_stats['ckpm'] - red_team_stats['ckpm'],
        'elo_delta': blue_team_stats['elo'] - red_team_stats['elo'],
        'opp_elo_delta': blue_team_stats['opp_elo'] - red_team_stats['opp_elo']}
    
    X_match = pd.DataFrame([match_features])[X_train.columns]  # ensure same column order as training data
    X_match = scaler.transform(X_match)
    X_match = pd.DataFrame(X_match, columns=X_train.columns)
    # get probability of blue winning
    prob_blue_wins = model.predict_proba(X_match)[0, 1] 
    
    # apply threshold
    if prob_blue_wins >= threshold:
        result = "Blue Side Wins"
        confidence = prob_blue_wins
    else:
        result = "Red Side Wins"
        confidence = 1 - prob_blue_wins # probability of red winning
        
    print(f"----- MATCH PREDICTION -----")
    print(f"Blue Side Elo: {blue_team_stats['elo']:<7.0f} | Red Side Elo: {red_team_stats['elo']:<7.0f}")
    print(f"Last 5 WR: {blue_team_stats['winrate']*100:.1f}%       | Last 5 WR: {red_team_stats['winrate']*100:.1f}%")
    print(f"─" * 48)
    print(f"Probability of {result}: {confidence * 100:.1f}%")
    
    return prob_blue_wins
