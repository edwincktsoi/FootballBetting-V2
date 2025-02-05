import os
import json
import pandas as pd
from understatapi import UnderstatClient
from understatapi.exceptions import InvalidMatch
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
import pickle 
import statsmodels.api as sm

#%%
print('Load Data')

match_info = pd.read_csv("./Football_data/understats/match_info/combined_match_info.csv")
match_info = match_info.sort_values(by='date',ascending=True)

fixtures = pd.read_csv('./Football_data/fixtures/fixtures.csv')
fixtures = fixtures[fixtures['Div'] == 'E0']  # Filter English Premier League fixtures

# Map team names in final_ml_data
fixrure_to_understats = {
    'Newcastle': 'Newcastle United',
    'Leicester': 'Leicester',
    'Everton': 'Everton',
    'Chelsea': 'Chelsea',
    'West Ham': 'West Ham',
    'Brentford': 'Brentford',
    'Arsenal': 'Arsenal',
    'Ipswich': 'Ipswich',
    'Man United': 'Manchester United',
    "Nott'm Forest": 'Nottingham Forest',
    'Tottenham': 'Tottenham',
    'Aston Villa': 'Aston Villa',
    'Bournemouth': 'Bournemouth',
    'Crystal Palace': 'Crystal Palace',
    'Man City': 'Manchester City',
    'Southampton': 'Southampton',
    'Brighton': 'Brighton',
    'Fulham': 'Fulham',
    'Liverpool': 'Liverpool',
    'Wolves': 'Wolverhampton Wanderers'
}

fixtures['HomeTeam'] = fixtures['HomeTeam'].map(fixrure_to_understats)
fixtures['AwayTeam'] = fixtures['AwayTeam'].map(fixrure_to_understats)
fixtures = fixtures[['Date','HomeTeam', 'AwayTeam']]
#%%
print('Create-clean-sheets-feature')
# Create a helper DataFrame to track clean sheets for all teams
def calculate_clean_sheets(df, team_type):
    # For home teams: clean sheet = a_goals == 0
    # For away teams: clean sheet = h_goals == 0
    if team_type == "home":
        team_col = 'team_h'
        goals_conceded_col = 'a_goals'
    else:
        team_col = 'team_a'
        goals_conceded_col = 'h_goals'
    
    # Create a team-specific DataFrame
    team_df = df[[team_col, 'date', goals_conceded_col]].copy()
    team_df.columns = ['team', 'date', 'goals_conceded']
    
    # Calculate clean sheets (0 goals conceded)
    team_df['clean_sheet'] = (team_df['goals_conceded'] == 0).astype(int)
    
    # Sort by team and date
    team_df = team_df.sort_values(['team', 'date'])
    
    # Calculate rolling clean sheets (last 5 matches)
    team_df['clean_sheets_last_5'] = (
        team_df.groupby('team')['clean_sheet']
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .shift(1)  # Exclude current match
    )
    
    return team_df[['team', 'date', 'clean_sheets_last_5']]

# Calculate for home and away teams
home_clean = calculate_clean_sheets(match_info, "home")
away_clean = calculate_clean_sheets(match_info, "away")

# Merge back to original data
match_info = match_info.merge(
    home_clean,
    left_on=['team_h', 'date'],
    right_on=['team', 'date'],
    how='left'
).rename(columns={'clean_sheets_last_5': 'h_clean_sheets_last_5'})

match_info = match_info.merge(
    away_clean,
    left_on=['team_a', 'date'],
    right_on=['team', 'date'],
    how='left'
).rename(columns={'clean_sheets_last_5': 'a_clean_sheets_last_5'})

# Clean up and fill NaN values (first matches)
match_info = match_info.drop(columns=['team_x', 'team_y'])
match_info[['h_clean_sheets_last_5', 'a_clean_sheets_last_5']] = match_info[
    ['h_clean_sheets_last_5', 'a_clean_sheets_last_5']
].fillna(0)

#%%
print('Create-head-to-head-points-difference')

import pandas as pd

# Sort data by date to ensure chronological order
match_info = match_info.sort_values(by='date', ascending=True)

# Initialize the new column
match_info['h_h2h_wins_last_5'] = 0

# Iterate over each row to calculate H2H wins
for idx, row in match_info.iterrows():
    current_date = row['date']
    team_h = row['team_h']
    team_a = row['team_a']
    
    # Filter past H2H matches between these teams (before the current match)
    past_h2h = match_info[
        (
            ((match_info['team_h'] == team_h) & (match_info['team_a'] == team_a)) |  # Team_h as home
            ((match_info['team_h'] == team_a) & (match_info['team_a'] == team_h))    # Team_h as away
        ) & 
        (match_info['date'] < current_date)  # Only matches before the current one
    ]
    
    # Get the last 3 H2H matches (most recent)
    last_3_h2h = past_h2h.sort_values('date', ascending=True).tail(5)
    
    # Count wins for the home team (team_h) in these matches
    wins = 0
    for _, match in last_3_h2h.iterrows():
        if (match['team_h'] == team_h) & (match['h_goals'] > match['a_goals']):
            wins += 1  # Team_h won as home team
        elif (match['team_a'] == team_h) & (match['a_goals'] > match['h_goals']):
            wins += 1  # Team_h won as away team
    
    match_info.at[idx, 'h_h2h_wins_last_5'] = wins

# Display the result
print(match_info[['date', 'team_h', 'team_a', 'h_h2h_wins_last_5']])

#%%
print('Create-League-Position-Feature')
import pandas as pd
def calculate_and_append_future_league_positions(df):
    """
    Calculate future league positions for each team after each match in a season.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing match data with columns:
        - date: match date
        - season: season identifier
        - team_h: home team
        - team_a: away team
        - h_goals: home team goals
        - a_goals: away team goals
    
    Returns:
    pandas.DataFrame: Original dataframe with additional columns showing future stats
    """
    # Sort matches chronologically
    df_sorted = df.sort_values(['season', 'date']).reset_index(drop=True)
    
    # Create a copy to avoid modifying the original
    result_df = df_sorted.copy()
    
    # Initialize new columns
    result_df['h_position'] = None
    result_df['a_position'] = None
    result_df['h_points'] = None
    result_df['a_points'] = None
    result_df['h_played'] = None
    result_df['a_played'] = None
    result_df['h_goal_diff'] = None
    result_df['a_goal_diff'] = None
    result_df['h_goals_scored'] = None
    result_df['a_goals_scored'] = None
    
    # Process each season separately
    for season in df_sorted['season'].unique():
        season_mask = df_sorted['season'] == season
        season_matches = df_sorted[season_mask].copy()
        
        # Track team standings throughout the season
        team_standings = {}
        
        # Initialize standings for all teams in the season
        all_teams = set(season_matches['team_h'].unique()).union(set(season_matches['team_a'].unique()))
        for team in all_teams:
            team_standings[team] = {
                'points': 0,
                'played': 0,
                'goal_diff': 0,
                'goals_scored': 0,
                'goals_conceded': 0
            }
        
        # Process matches in chronological order
        for idx, match in season_matches.iterrows():
            h_team = match['team_h']
            a_team = match['team_a']
            h_goals = match['h_goals']
            a_goals = match['a_goals']
            
            # Update matches played
            team_standings[h_team]['played'] += 1
            team_standings[a_team]['played'] += 1
            
            # Update goals scored and conceded
            team_standings[h_team]['goals_scored'] += h_goals
            team_standings[h_team]['goals_conceded'] += a_goals
            team_standings[a_team]['goals_scored'] += a_goals
            team_standings[a_team]['goals_conceded'] += h_goals
            
            # Update goal difference
            team_standings[h_team]['goal_diff'] = team_standings[h_team]['goals_scored'] - team_standings[h_team]['goals_conceded']
            team_standings[a_team]['goal_diff'] = team_standings[a_team]['goals_scored'] - team_standings[a_team]['goals_conceded']
            
            # Update points based on match result
            if h_goals > a_goals:  # Home win
                team_standings[h_team]['points'] += 3
            elif h_goals < a_goals:  # Away win
                team_standings[a_team]['points'] += 3
            else:  # Draw
                team_standings[h_team]['points'] += 1
                team_standings[a_team]['points'] += 1
            
            # Calculate league positions after the match
            standings_list = [
                {
                    'team': team,
                    'points': stats['points'],
                    'goal_diff': stats['goal_diff'],
                    'goals_scored': stats['goals_scored']
                }
                for team, stats in team_standings.items()
            ]
            
            # Sort by Points -> Goal Difference -> Goals Scored
            standings_list.sort(key=lambda x: (-x['points'], -x['goal_diff'], -x['goals_scored']))
            
            # Assign positions
            for pos, team_stats in enumerate(standings_list, 1):
                team_standings[team_stats['team']]['position'] = pos
            
            # Store future position and stats
            result_df.loc[idx, 'h_position'] = team_standings[h_team]['position']
            result_df.loc[idx, 'h_points'] = team_standings[h_team]['points']
            result_df.loc[idx, 'h_played'] = team_standings[h_team]['played']
            result_df.loc[idx, 'h_goal_diff'] = team_standings[h_team]['goal_diff']
            result_df.loc[idx, 'h_goals_scored'] = team_standings[h_team]['goals_scored']
            
            result_df.loc[idx, 'a_position'] = team_standings[a_team]['position']
            result_df.loc[idx, 'a_points'] = team_standings[a_team]['points']
            result_df.loc[idx, 'a_played'] = team_standings[a_team]['played']
            result_df.loc[idx, 'a_goal_diff'] = team_standings[a_team]['goal_diff']
            result_df.loc[idx, 'a_goals_scored'] = team_standings[a_team]['goals_scored']
    
    # Ensure all numeric columns are of type float
    numeric_columns = ['h_position', 'a_position', 'h_points', 'a_points', 'h_played', 'a_played', 'h_goal_diff', 'a_goal_diff', 'h_goals_scored', 'a_goals_scored']
    result_df[numeric_columns] = result_df[numeric_columns].astype(float)
    
    return result_df

# Example usage:
# Assuming `match_info` is your DataFrame containing match data
match_info = calculate_and_append_future_league_positions(match_info)


#%%
print('create-rolling-mean-columns')
# List to store all home and away DataFrames
all_team_data = []
# Get unique teams
teams = pd.concat([match_info['team_h'], match_info['team_a']]).unique()
# Loop over each team
for team_name in teams:
    # Create home_df for the team
    home_df = match_info[match_info['team_h'] == team_name].copy()
    home_df['venue'] = 'home'
    home_df['team'] = home_df['team_h']
    home_df = home_df.rename(columns={
        'h_goals': 'goals', 
        'h_xg': 'xg', 
        'h_shot': 'shot', 
        'h_shotOnTarget': 'shotOnTarget', 
        'h_deep': 'deep', 
        'h_ppda': 'ppda',
    })

    # Create away_df for the team
    away_df = match_info[match_info['team_a'] == team_name].copy()
    away_df['venue'] = 'away'
    away_df['team'] = away_df['team_a']
    away_df = away_df.rename(columns={
        'a_goals': 'goals', 
        'a_xg': 'xg', 
        'a_shot': 'shot', 
        'a_shotOnTarget': 'shotOnTarget', 
        'a_deep': 'deep', 
        'a_ppda': 'ppda',
    })

    # Select relevant columns for both home and away data
    home_df = home_df[['date', 'season', 'team', 'goals', 'xg', 'shot', 'shotOnTarget', 'deep', 'ppda', 'venue']]
    away_df = away_df[['date', 'season', 'team', 'goals', 'xg', 'shot', 'shotOnTarget', 'deep', 'ppda', 'venue']]

    # Append home and away DataFrames for this team to the list
    all_team_data.append(home_df)
    all_team_data.append(away_df)

# Concatenate all home and away DataFrames into a single DataFrame
final_team_data = pd.concat(all_team_data, ignore_index=True)
final_team_data = final_team_data.sort_values(by=['team', 'date']).reset_index(drop=True)


# Group by team and venue, then shift the data to exclude the current match for rolling mean calculation
final_team_data[['xg_rolling',  'deep_rolling']] = \
    final_team_data.groupby(['team', 'venue'])[['xg','deep' ]] \
    .rolling(window=5, min_periods=5) \
    .mean().reset_index(drop=True)  # Resetting index with drop=True to avoid issues with multi-level index


home_df = final_team_data[final_team_data['venue'] == 'home']
away_df = final_team_data[final_team_data['venue'] == 'away']

#%%
print('Data Pre-parperation for ML')
#fixture
match_index = match_info[['date', 'season', 'team_h', 'team_a','h_position', 'a_position','h_clean_sheets_last_5','h_h2h_wins_last_5',
       'a_clean_sheets_last_5']]

# Merge home stats
home_stats = pd.merge(match_index, home_df, left_on=['date', 'season', 'team_h'], right_on=['date', 'season', 'team'], how='left', suffixes=('_home', ''))
# Rename the home stats columns by adding '_home'
home_stats = home_stats.rename(columns={col: f'{col}_home' for col in home_stats.columns if col not in ['date', 'season', 'team_h']})

# Merge away stats
away_stats = pd.merge(match_index, away_df, left_on=['date', 'season', 'team_a'], right_on=['date', 'season', 'team'], how='left', suffixes=('_away', ''))
# Rename the away stats columns by adding '_away'
away_stats = away_stats.rename(columns={col: f'{col}_away' for col in away_stats.columns if col not in ['date', 'season', 'team_a']})


# Now merge the home and away stats together
final_ml_data = pd.merge(home_stats, away_stats, left_on=['date', 'season', 'team_h', 'team_a_home'], right_on=['date', 'season', 'team_h_away', 'team_a'], how='left')



#drop missing values
final_ml_data = final_ml_data.dropna()

final_ml_data = final_ml_data[['date', 'season', 'team_h', 'team_a','xg_rolling_home',
                               'deep_rolling_home', 'xg_rolling_away','h_clean_sheets_last_5_home',
                              'a_clean_sheets_last_5_away','h_h2h_wins_last_5_home',
                              'deep_rolling_away','h_position_home','a_position_away']]

                                                                                                                                                                                   


#%%
print('Construct team profile')
# Combine home and away data, then extract the latest record for each team
home_team_df = pd.DataFrame()
away_team_df = pd.DataFrame()

# Iterate over unique teams
teams = pd.concat([final_ml_data['team_h'], final_ml_data['team_a']]).unique()

# Rename columns for merging clarity
home_team_df_all = final_ml_data[['date', 'season', 'team_h', 'xg_rolling_home','h_h2h_wins_last_5_home',
       'deep_rolling_home','h_clean_sheets_last_5_home','h_position_home']].rename(columns={'team_h':'team'})

# Rename columns for merging clarity
away_team_df_all = final_ml_data[['date', 'team_a',  'xg_rolling_away', 'deep_rolling_away','a_clean_sheets_last_5_away',
       'a_position_away']].rename(columns={'team_a':'team'})

for team in teams:
    # Filter rows where the team is either home or away
    home_team = home_team_df_all[(home_team_df_all['team'] == team)] 
    away_team = away_team_df_all[(away_team_df_all['team'] == team)]

    # Get the latest row based on date
    last_home_team = home_team[home_team['date']==home_team['date'].max()]
    # Get the latest row based on date
    last_away_team = away_team[away_team['date']==away_team['date'].max()]
    
    
    
    # If you have an existing DataFrame to append to, use pd.concat()
    home_team_df = pd.concat([home_team_df, last_home_team], ignore_index=True) 
    away_team_df = pd.concat([away_team_df, last_away_team], ignore_index=True) 


#%%
final_ml_data = pd.merge(fixtures,home_team_df,  left_on=['HomeTeam'], right_on=['team'], how='left')
final_ml_data = pd.merge(final_ml_data,away_team_df,  left_on=['AwayTeam'], right_on=['team'], how='left')

final_ml_data['xg_rolling_diff'] = final_ml_data['xg_rolling_home'] - final_ml_data['xg_rolling_away']
final_ml_data['deep_rolling_diff'] = final_ml_data['deep_rolling_home'] - final_ml_data['deep_rolling_away']
final_ml_data['league_position_diff'] = final_ml_data['h_position_home'] - final_ml_data['a_position_away']
final_ml_data['defensive_strength_diff'] = abs(final_ml_data['h_clean_sheets_last_5_home'] - final_ml_data['a_clean_sheets_last_5_away'])


final_ml_data = final_ml_data[['Date', 'HomeTeam', 'AwayTeam','defensive_strength_diff','h_h2h_wins_last_5_home', 'xg_rolling_diff', 'deep_rolling_diff', 'league_position_diff']]

final_ml_data['deep_rolling_diff'] = np.where(final_ml_data['deep_rolling_diff']>=8.8,8.8,final_ml_data['deep_rolling_diff'])
final_ml_data['xg_rolling_diff'] = np.where(final_ml_data['xg_rolling_diff']>=2,2,final_ml_data['xg_rolling_diff'])

final_ml_data = final_ml_data.rename(columns={"h_h2h_wins_last_5_home":"h_h2h_wins_last_5", "xg_rolling_diff":"xg_rolling_diff", "deep_rolling_diff":"deep_rolling_diff", "league_position_diff":"league_position_diff", "defensive_strength_diff":"defensive_strength_diff"})
#%%
# Load the model and scaler
with open('logistic_regression_model_pl.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_pl.pkl', 'rb') as f:
    scaler = pickle.load(f)



# Ensure the features are selected properly
features = ['xg_rolling_diff', 'deep_rolling_diff', 'league_position_diff', 
                     'h_h2h_wins_last_5', 'defensive_strength_diff']
X = final_ml_data[features]

# Example: Prepare new data for prediction
new_data_scaled = scaler.transform(X)


# Predict the match outcomes
print('Save Test result')

# Get predicted probabilities for each class (home win, draw, away win)
probabilities = model.predict_proba(new_data_scaled)

# Add the probabilities for each class (home win, draw, away win)
final_ml_data['home_win_prob'] = probabilities[:, 0]
final_ml_data['draw_prob'] = probabilities[:, 1]
final_ml_data['away_win_prob'] = probabilities[:, 2]


# Define the folder name
folder_name = "./Football_data/output/"

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"'{folder_name}' folder created successfully.")
else:
    print(f"'{folder_name}' folder already exists.")

final_ml_data.to_csv('./Football_data/output/predictions.csv', index=False)
# %%
