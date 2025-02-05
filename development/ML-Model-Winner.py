#%% 
import pandas as pd
import numpy as np

print('Data loading and pre-processing')
# Load dataset
directory = "./Football_data/understats"
data = pd.read_csv(f"{directory}/match_info/combined_match_info.csv")

# Select relevant columns
data = data[['date', 'season', 'team_h', 'team_a', 'h_goals', 'a_goals', 
             'h_xg', 'a_xg', 'h_shot', 'a_shot', 'h_shotOnTarget', 'a_shotOnTarget',
             'h_deep', 'a_deep', 'h_ppda', 'a_ppda']]

data = data.sort_values(by='date',ascending=True)
#%%
print('dixon_coles_likelihood....')
import numpy as np
from scipy.optimize import minimize

# Define the Dixon-Coles likelihood function
def dixon_coles_likelihood(params, data, teams):
    # Extract parameters
    alpha = params[:len(teams)]  # Attack strengths
    beta = params[len(teams):2*len(teams)]  # Defense strengths
    gamma = params[-2]  # Home advantage
    rho = params[-1]  # Dependence parameter
    
    log_likelihood = 0
    for idx, row in data.iterrows():
        home_team = row['team_h']
        away_team = row['team_a']
        home_goals = row['h_goals']
        away_goals = row['a_goals']
        
        # Get team indices
        home_idx = teams.index(home_team)
        away_idx = teams.index(away_team)
        
        # Expected goals
        mu_h = np.exp(alpha[home_idx] - beta[away_idx] + gamma)
        mu_a = np.exp(alpha[away_idx] - beta[home_idx])
        
        # Dixon-Coles adjustment for low-scoring games
        if home_goals == 0 and away_goals == 0:
            tau = 1 - (mu_h * mu_a * rho)
        elif home_goals == 0 and away_goals == 1:
            tau = 1 + (mu_h * rho)
        elif home_goals == 1 and away_goals == 0:
            tau = 1 + (mu_a * rho)
        elif home_goals == 1 and away_goals == 1:
            tau = 1 - rho
        else:
            tau = 1
        
        # Poisson likelihood
        poisson_likelihood = (np.exp(-mu_h) * mu_h**home_goals / np.math.factorial(home_goals)) * \
                             (np.exp(-mu_a) * mu_a**away_goals / np.math.factorial(away_goals))
        
        # Add to log-likelihood
        log_likelihood += np.log(tau * poisson_likelihood)
    
    return -log_likelihood  # Negative for minimization

# Get unique teams
teams = sorted(list(set(data['team_h'].unique()).union(set(data['team_a'].unique()))))

# Initial parameters: alpha (attack), beta (defense), gamma (home advantage), rho (dependence)
initial_params = np.random.normal(0, 1, size=2*len(teams) + 2)

# Minimize the negative log-likelihood
result = minimize(dixon_coles_likelihood, initial_params, args=(data, teams), method='L-BFGS-B')

# Extract parameters
alpha = result.x[:len(teams)]
beta = result.x[len(teams):2*len(teams)]
gamma = result.x[-2]
rho = result.x[-1]

# Print results
print("Attack strengths (alpha):", dict(zip(teams, alpha)))
print("Defense strengths (beta):", dict(zip(teams, beta)))
print("Home advantage (gamma):", gamma)
print("Dependence parameter (rho):", rho)

# Function to compute expected goals using Dixon-Coles parameters
def compute_expected_goals(data, alpha, beta, gamma, teams):
    data['h_xg_dc'] = 0.0
    data['a_xg_dc'] = 0.0
    
    for idx, row in data.iterrows():
        home_team = row['team_h']
        away_team = row['team_a']
        
        home_idx = teams.index(home_team)
        away_idx = teams.index(away_team)
        
        # Expected goals for home and away teams
        h_xg = np.exp(alpha[home_idx] - beta[away_idx] + gamma)
        a_xg = np.exp(alpha[away_idx] - beta[home_idx])
        
        # Add to dataset
        data.at[idx, 'h_xg_dc'] = h_xg
        data.at[idx, 'a_xg_dc'] = a_xg
    
    return data

# Add expected goals to the dataset
data = compute_expected_goals(data, alpha, beta, gamma, teams)

# Add team strengths and home advantage as features
for team in teams:
    data[f'{team}_attack'] = alpha[teams.index(team)]
    data[f'{team}_defense'] = beta[teams.index(team)]

data['home_advantage'] = gamma
data['dependence_rho'] = rho

# Display the updated dataset
print(data.head())
#%%
print('number of matches played....')

# Melt the DataFrame to stack home/away teams into a single column
melted = data.melt(
    id_vars=['date'],
    value_vars=['team_h', 'team_a'],
    var_name='home_or_away',
    value_name='team'
)

# Sort by team and date to prepare for cumulative count
melted = melted.sort_values(['team', 'date'])

# Calculate cumulative matches played for each team
melted['matches_played'] = melted.groupby('team').cumcount()

# Split back into home and away DataFrames
home_matches = melted[melted['home_or_away'] == 'team_h'][['date', 'team', 'matches_played']]
away_matches = melted[melted['home_or_away'] == 'team_a'][['date', 'team', 'matches_played']]

# Merge home team's matches_played into original data
data = data.merge(
    home_matches,
    left_on=['date', 'team_h'],
    right_on=['date', 'team'],
    how='left'
).rename(columns={'matches_played': 'h_matches_played'}).drop(columns='team')

# Merge away team's matches_played into original data
data = data.merge(
    away_matches,
    left_on=['date', 'team_a'],
    right_on=['date', 'team'],
    how='left'
).rename(columns={'matches_played': 'a_matches_played'}).drop(columns='team')

# Fill NaN values (for teams playing their first match)
data[['h_matches_played', 'a_matches_played']] = data[
    ['h_matches_played', 'a_matches_played']
].fillna(0)

# Convert to integer (optional)
data['h_matches_played'] = data['h_matches_played'].astype(int)
data['a_matches_played'] = data['a_matches_played'].astype(int)
#%%
# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Sort by date
data = data.sort_values(by='date', ascending=True)

# Initialize new columns
data['days_since_last_play_h'] = None
data['days_since_last_play_a'] = None

# Calculate days since last play for home and away teams
for team in set(data['team_h']).union(set(data['team_a'])):
    # Filter matches involving the team (home or away)
    team_matches = data[(data['team_h'] == team) | (data['team_a'] == team)]
    
    # Calculate the difference in days between consecutive matches
    team_matches = team_matches.assign(
        days_since_last_play=team_matches['date'].diff().dt.days
    )
    
    # Update the main DataFrame with the calculated values
    for idx, row in team_matches.iterrows():
        if row['team_h'] == team:
            data.at[idx, 'days_since_last_play_h'] = row['days_since_last_play']
        elif row['team_a'] == team:
            data.at[idx, 'days_since_last_play_a'] = row['days_since_last_play']

# Fill NaN values (first match for each team) with a default value (e.g., 0 or a large number)
data['days_since_last_play_h'] = data['days_since_last_play_h'].fillna(0)
data['days_since_last_play_a'] = data['days_since_last_play_a'].fillna(0)


#%%
import pandas as pd

# Assuming 'data' is your DataFrame
# Sort the data by date to ensure chronological order
data = data.sort_values(by='date')

# Initialize dictionaries to store the last 5 match results for each team
team_last_5_results = {}

# Function to determine the result of a match for a team
def get_match_result(team, row):
    if team == row['team_h']:
        if row['h_goals'] > row['a_goals']:
            return 'win'
        elif row['h_goals'] == row['a_goals']:
            return 'draw'
        else:
            return 'loss'
    elif team == row['team_a']:
        if row['a_goals'] > row['h_goals']:
            return 'win'
        elif row['a_goals'] == row['h_goals']:
            return 'draw'
        else:
            return 'loss'
    return None

# Function to calculate the win rate in the last 5 matches
def calculate_win_rate(results):
    if len(results) == 0:
        return 0.0
    wins = results.count('win')
    return (wins / len(results)) * 100  # Return as a percentage

# Iterate through the data and update the last 5 results for each team
for index, row in data.iterrows():
    home_team = row['team_h']
    away_team = row['team_a']
    
    # Calculate win rates for home and away teams **before** updating the current match results
    if home_team in team_last_5_results:
        data.at[index, 'h_recent_win_rate'] = calculate_win_rate(team_last_5_results[home_team])
    else:
        data.at[index, 'h_recent_win_rate'] = 0.0  # If no previous matches, win rate is 0
    
    if away_team in team_last_5_results:
        data.at[index, 'a_recent_win_rate'] = calculate_win_rate(team_last_5_results[away_team])
    else:
        data.at[index, 'a_recent_win_rate'] = 0.0  # If no previous matches, win rate is 0
    
    # Update last 5 results for the home team **after** calculating the win rate
    if home_team not in team_last_5_results:
        team_last_5_results[home_team] = []
    team_last_5_results[home_team].append(get_match_result(home_team, row))
    if len(team_last_5_results[home_team]) > 5:
        team_last_5_results[home_team].pop(0)
    
    # Update last 5 results for the away team **after** calculating the win rate
    if away_team not in team_last_5_results:
        team_last_5_results[away_team] = []
    team_last_5_results[away_team].append(get_match_result(away_team, row))
    if len(team_last_5_results[away_team]) > 5:
        team_last_5_results[away_team].pop(0)

# Display the updated DataFrame
print(data[['date', 'season', 'team_h', 'team_a', 'h_recent_win_rate', 'a_recent_win_rate']])

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
data = calculate_and_append_future_league_positions(data)


#%%
# Print statement
print('Create-Win-Streak-Feature')
import pandas as pd

import pandas as pd
import numpy as np



# Create a column for home team result and away team result
data['h_result'] = data.apply(lambda row: 1 if row['h_goals'] > row['a_goals'] else (0 if row['h_goals'] < row['a_goals'] else None), axis=1)
data['a_result'] = data.apply(lambda row: 1 if row['a_goals'] > row['h_goals'] else (0 if row['a_goals'] < row['h_goals'] else None), axis=1)

# Initialize streak columns
data['team_h_streak'] = 0
data['team_a_streak'] = 0

# Iterate through each row to calculate streaks
team_h_streak = 0
team_a_streak = 0

# Create a dictionary to store streaks for each team
team_h_streaks = {}
team_a_streaks = {}

for index, row in data.iterrows():
    # For Home team (exclude current match)
    team_h_streak_prev = team_h_streaks.get(row['team_h'], 0)
    if row['h_result'] == 1:  # Home win
        team_h_streak = team_h_streak_prev + 1
    elif row['h_result'] == 0:  # Home loss
        team_h_streak = team_h_streak_prev - 1
    else:
        team_h_streak = 0  # Reset streak for a draw
    
    # For Away team (exclude current match)
    team_a_streak_prev = team_a_streaks.get(row['team_a'], 0)
    if row['a_result'] == 1:  # Away win
        team_a_streak = team_a_streak_prev + 1
    elif row['a_result'] == 0:  # Away loss
        team_a_streak = team_a_streak_prev - 1
    else:
        team_a_streak = 0  # Reset streak for a draw

    # Store the streak values for this match
    data.at[index, 'team_h_streak'] = team_h_streak
    data.at[index, 'team_a_streak'] = team_a_streak

    # Update the dictionary with the latest streaks
    team_h_streaks[row['team_h']] = team_h_streak
    team_a_streaks[row['team_a']] = team_a_streak

# Adjust the streaks: convert results into cumulative win/loss streaks
data['team_h_streak'] = np.where(data['team_h_streak'] != 0, data['team_h_streak'], 0)
data['team_a_streak'] = np.where(data['team_a_streak'] != 0, data['team_a_streak'], 0)

# Adjust the streak sign to make losses negative (both for home and away)
data['team_h_streak'] = data.apply(lambda row: row['team_h_streak'] if row['h_result'] == 1 else -abs(row['team_h_streak']), axis=1)
data['team_a_streak'] = data.apply(lambda row: row['team_a_streak'] if row['a_result'] == 1 else -abs(row['team_a_streak']), axis=1)

data['team_h_streak'] = data['team_h_streak'].shift(1)
data['team_a_streak'] = data['team_a_streak'].shift(1)
# Display the updated dataframe with combined streak columns
print(data[['date', 'team_h', 'team_a', 'team_h_streak', 'team_a_streak']])



#%%
import pandas as pd
# Print statement
print('Create-league-point-diff')

# Step 1: Calculate points for each match
def calculate_points(row):
    if row['h_goals'] > row['a_goals']:
        return pd.Series([3, 0])  # Home wins, Away loses
    elif row['h_goals'] < row['a_goals']:
        return pd.Series([0, 3])  # Away wins, Home loses
    else:
        return pd.Series([1, 1])  # Draw

# Apply the function to calculate points for each team (home and away)
data[['h_points', 'a_points']] = data.apply(calculate_points, axis=1)

# Step 2: Create a DataFrame to store total points for each team by season and date
team_points = pd.DataFrame({
    'date': pd.concat([data['date'], data['date']]),  # Concatenate dates for both teams
    'season': pd.concat([data['season'], data['season']]),  # Concatenate seasons for both teams
    'team': pd.concat([data['team_h'], data['team_a']]),     # Concatenate home and away teams
    'points': pd.concat([data['h_points'], data['a_points']])  # Concatenate home and away points
})

# Step 3: Sort data by season, team, and date to calculate cumulative sum correctly
team_points = team_points.sort_values(by=['season', 'team', 'date'])

# Step 4: Calculate cumulative sum of points for each team up to each date, then shift(1)
team_points['cumsum'] = team_points.groupby(['season', 'team'])['points'].cumsum()
team_points['prev_points'] = team_points.groupby(['season', 'team'])['cumsum'].shift(1).fillna(0)

# Step 5: Split shifted cumulative points into home and away teams
home_cumsum = team_points.rename(columns={'team': 'team_h', 'prev_points': 'h_league_point'})
away_cumsum = team_points.rename(columns={'team': 'team_a', 'prev_points': 'a_league_point'})

# Step 6: Merge shifted cumulative points back into the original data
# Merge home cumulative points
data = data.merge(home_cumsum[['date', 'season', 'team_h', 'h_league_point']],
                  on=['date', 'season', 'team_h'], how='left')

# Merge away cumulative points
data = data.merge(away_cumsum[['date', 'season', 'team_a', 'a_league_point']],
                  on=['date', 'season', 'team_a'], how='left')

# Step 7: Select relevant columns
data = data[['date', 'season', 'team_h', 'team_a', 'h_goals', 'a_goals', 'h_matches_played','a_matches_played',
             'h_xg', 'a_xg', 'h_shot', 'a_shot', 'h_shotOnTarget', 'a_shotOnTarget','h_recent_win_rate', 'a_recent_win_rate',
             'h_deep', 'a_deep', 'h_ppda', 'a_ppda', 'h_league_point', 'a_league_point','team_h_streak', 'team_a_streak','h_position', 'a_position',
             'days_since_last_play_h','days_since_last_play_a']]






#%%
print('Create-head-to-head-points-difference')

import pandas as pd

# Sort data by date to ensure chronological order
data = data.sort_values(by='date', ascending=True)

# Initialize the new column
data['h_h2h_wins_last_5'] = 0

# Iterate over each row to calculate H2H wins
for idx, row in data.iterrows():
    current_date = row['date']
    team_h = row['team_h']
    team_a = row['team_a']
    
    # Filter past H2H matches between these teams (before the current match)
    past_h2h = data[
        (
            ((data['team_h'] == team_h) & (data['team_a'] == team_a)) |  # Team_h as home
            ((data['team_h'] == team_a) & (data['team_a'] == team_h))    # Team_h as away
        ) & 
        (data['date'] < current_date)  # Only matches before the current one
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
    
    data.at[idx, 'h_h2h_wins_last_5'] = wins

# Display the result
print(data[['date', 'team_h', 'team_a', 'h_h2h_wins_last_5']])
#%%

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
home_clean = calculate_clean_sheets(data, "home")
away_clean = calculate_clean_sheets(data, "away")

# Merge back to original data
data = data.merge(
    home_clean,
    left_on=['team_h', 'date'],
    right_on=['team', 'date'],
    how='left'
).rename(columns={'clean_sheets_last_5': 'h_clean_sheets_last_5'})

data = data.merge(
    away_clean,
    left_on=['team_a', 'date'],
    right_on=['team', 'date'],
    how='left'
).rename(columns={'clean_sheets_last_5': 'a_clean_sheets_last_5'})

# Clean up and fill NaN values (first matches)
data = data.drop(columns=['team_x', 'team_y'])
data[['h_clean_sheets_last_5', 'a_clean_sheets_last_5']] = data[
    ['h_clean_sheets_last_5', 'a_clean_sheets_last_5']
].fillna(0)



#%%
import math

print('Applying Elo Rating Algorithm with Reset for Non-PL Seasons')

# Initialize team ratings with a default value of 1600
initial_rating = 1600
reset_rating = 1580
team_ratings = {}

# Elo k-factor, controls how much ratings change per match
k_factor = 32

# Initialize columns for Elo ratings in the data
data['home_elo'] = 0
data['away_elo'] = 0

# Get unique seasons and teams
unique_seasons = data['season'].unique()
teams_in_pl = {season: set(data[data['season'] == season]['team_h'].unique()) | 
                       set(data[data['season'] == season]['team_a'].unique()) 
               for season in unique_seasons}

# Create a DataFrame to track Elo ratings by team, season, and date
elo_history = []

# Iterate through the dataset row by row
for index, row in data.iterrows():
    # Get home and away team names and the season
    home_team = row['team_h']
    away_team = row['team_a']
    current_season = row['season']

    # Reset ratings to 1550 for teams not in PL this season
    for team in list(team_ratings.keys()):
        if team not in teams_in_pl[current_season]:
            team_ratings[team] = reset_rating

    # Initialize ratings for teams if not already present
    if home_team not in team_ratings:
        team_ratings[home_team] = initial_rating
    if away_team not in team_ratings:
        team_ratings[away_team] = initial_rating

    # Fetch current ratings (these represent ratings *before* the match)
    home_rating = team_ratings[home_team]
    away_rating = team_ratings[away_team]

    # Save current ratings to the Elo history
    elo_history.append({'date': row['date'], 'team': home_team, 'season': current_season, 'elo': home_rating})
    elo_history.append({'date': row['date'], 'team': away_team, 'season': current_season, 'elo': away_rating})

    # Calculate expected probabilities
    expected_home_win = 1 / (1 + math.pow(10, (away_rating - home_rating) / 400))
    expected_away_win = 1 - expected_home_win

    # Determine match outcome
    if row['h_goals'] > row['a_goals']:
        home_result = 1  # Home win
        away_result = 0
    elif row['h_goals'] < row['a_goals']:
        home_result = 0  # Away win
        away_result = 1
    else:
        home_result = 0.5  # Draw
        away_result = 0.5

    # Update ratings based on match result
    new_home_rating = home_rating + k_factor * (home_result - expected_home_win)
    new_away_rating = away_rating + k_factor * (away_result - expected_away_win)

    # Save updated ratings
    team_ratings[home_team] = new_home_rating
    team_ratings[away_team] = new_away_rating

# Convert Elo history to a DataFrame
elo_history_df = pd.DataFrame(elo_history)

# Shift Elo ratings for each team to represent "previous rating"
elo_history_df['prev_elo'] = elo_history_df.groupby(['season', 'team'])['elo'].shift(1).fillna(initial_rating)

# Merge shifted Elo ratings back into the main dataset for both home and away teams
home_elo = elo_history_df.rename(columns={'team': 'team_h', 'prev_elo': 'home_elo'})
away_elo = elo_history_df.rename(columns={'team': 'team_a', 'prev_elo': 'away_elo'})

data = data.merge(home_elo[['date', 'season', 'team_h', 'home_elo']], on=['date', 'season', 'team_h'], how='left')
data = data.merge(away_elo[['date', 'season', 'team_a', 'away_elo']], on=['date', 'season', 'team_a'], how='left')

data = data.drop(columns=['home_elo_x', 'away_elo_x'],axis=1).rename({'home_elo_y':'home_elo','away_elo_y':'away_elo'},axis=1)

# Calculate Elo difference
data['elo_diff'] = data['home_elo'] - data['away_elo']

data = data[data['home_elo'] != 0]



#%%
print('create-rolling-mean-columns')
# List to store all home and away DataFrames
all_team_data = []
# Get unique teams
teams = pd.concat([data['team_h'], data['team_a']]).unique()
# Loop over each team
for team_name in teams:
    # Create home_df for the team
    home_df = data[data['team_h'] == team_name].copy()
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
    away_df = data[data['team_a'] == team_name].copy()
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
final_team_data[['goals_rolling', 'xg_rolling', 'shot_rolling', 'shotOnTarget_rolling', 'deep_rolling', 'ppda_rolling']] = \
    final_team_data.groupby(['team', 'venue'])[['goals', 'xg', 'shot', 'shotOnTarget', 'deep', 'ppda']] \
    .shift(1) \
    .rolling(window=5, min_periods=5) \
    .mean().reset_index(drop=True)  # Resetting index with drop=True to avoid issues with multi-level index




## Compute rolling sum of points for the last 5 matches
#final_team_data['last_5_points_rolling'] = (
#    final_team_data.groupby('team')['points']
#    .shift(1)  # Exclude the current match from the rolling calculation
#    .rolling(window=5, min_periods=5)
#    .sum()
#    .reset_index(drop=True)  # Resetting index with drop=True to avoid issues with multi-level index
#)
#
## Compute rolling sum of points for the last  matches
#final_team_data['last_match_points'] = final_team_data.groupby('team')['points'].shift(1).reset_index(drop=True)  # Resetting index with drop=True to avoid issues with multi-level index
#
## Calculate cumulative points for each team within each season
#final_team_data['total_league_points'] = final_team_data.groupby(['season', 'team'])['points'].cumsum()

home_df = final_team_data[final_team_data['venue'] == 'home']
away_df = final_team_data[final_team_data['venue'] == 'away']


#%%
print('create-target-column')
#dont model draw
#data = data[data['h_goals'] != data['a_goals']]

# Create a target column 'target' based on the goals scored by home and away teams
data['target'] = data.apply(lambda row: 0 if row['h_goals'] > row['a_goals'] else (1 if row['h_goals'] == row['a_goals'] else 2), axis=1)
                                   



# %%
print('Data Pre-parperation for ML')
#fixture
match_index = data[['date', 'season', 'team_h', 'team_a','elo_diff','team_h_streak', 'team_a_streak','h_league_point', 
                    'a_league_point','h_position','a_position','h_recent_win_rate', 'a_recent_win_rate','h_matches_played','a_matches_played',
                    'days_since_last_play_h','days_since_last_play_a','h_h2h_wins_last_5','h_clean_sheets_last_5', 'a_clean_sheets_last_5',
                    'home_advantage','dependence_rho',
                    'target']]

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

final_ml_data = final_ml_data[['date', 'season', 'team_h', 'team_a', 
                              'xg_rolling_home',
       #'shot_rolling_home', 'shotOnTarget_rolling_home', 
       'deep_rolling_home',
       'ppda_rolling_home',  'goals_rolling_away', 'xg_rolling_away','goals_rolling_home',
       #'shot_rolling_away', 'shotOnTarget_rolling_away', 
       'deep_rolling_away','h_league_point_home', 'a_league_point_home','h_matches_played_home','a_matches_played_home',
       'ppda_rolling_away','elo_diff_home',
       'team_h_streak_home', 'team_a_streak_home','h_position_home','a_position_home','h_recent_win_rate_home', 'a_recent_win_rate_home',
       'days_since_last_play_h_home','days_since_last_play_a_home','h_h2h_wins_last_5_home','h_clean_sheets_last_5_home',  'a_clean_sheets_last_5_home',
       'target_home',
       'home_advantage_home','dependence_rho_home',
       ]]
final_ml_data = final_ml_data.rename(columns={'target_home': 'target',
                                       'head_to_head_points_diff_home':'head_to_head_points_diff',
                                       'team_h_streak_home':'team_h_streak', 
                                       'team_a_streak_home':'team_a_streak',
                                        'elo_diff_home':'elo_diff',
                                        'h_league_point_home':'h_league_point', 
                                        'a_league_point_home':'a_league_point',
                                        'h_position_home':'h_position',
                                        'a_position_home':'a_position',
                                        'h_recent_win_rate_home':'h_recent_win_rate', 
                                        'a_recent_win_rate_home':'a_recent_win_rate',
                                        'days_since_last_play_h_home':'days_since_last_play_h',
                                        'days_since_last_play_a_home':'days_since_last_play_a',
                                        'h_h2h_wins_last_5_home':'h_h2h_wins_last_5',
'h_clean_sheets_last_5_home':'h_clean_sheets_last_5',
'a_clean_sheets_last_5_home':'a_clean_sheets_last_5',
'h_matches_played_home':'h_matches_played',
'a_matches_played_home':'a_matches_played',
'home_advantage_home':'home_advantage',

'dependence_rho_home':'dependence_rho',
                                        })
                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
# Preview the final dataset
final_ml_data.head()

# %%

import statsmodels.api as sm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


# Sort the data by date to ensure chronological order
final_ml_data = final_ml_data.sort_values(by='date')

# Split the data based on time: e.g., using the date 2022-01-01 for splitting
split_date = '2024-01-01'

# Create columns for differences between home and away stats
final_ml_data['goals_rolling_diff'] = final_ml_data['goals_rolling_home'] - final_ml_data['goals_rolling_away']
final_ml_data['xg_rolling_diff'] = final_ml_data['xg_rolling_home'] - final_ml_data['xg_rolling_away']
final_ml_data['deep_rolling_diff'] = final_ml_data['deep_rolling_home'] - final_ml_data['deep_rolling_away']
final_ml_data['ppda_rolling_diff'] = final_ml_data['ppda_rolling_home'] - final_ml_data['ppda_rolling_away']
#final_ml_data['last_5_point_diff'] =final_ml_data['last_5_points_rolling_home'] - final_ml_data['last_5_points_rolling_away']
#final_ml_data['point_diff'] =final_ml_data['last_match_points_home'] - final_ml_data['last_match_points_away']
final_ml_data['league_points_diff'] =final_ml_data['h_league_point'] - final_ml_data['a_league_point']

final_ml_data['league_position_diff'] =final_ml_data['h_position'] - final_ml_data['a_position']

final_ml_data['team_streak_diff'] = final_ml_data['team_h_streak'] - final_ml_data['team_a_streak']
final_ml_data['recent_win_rate_diff'] = final_ml_data['h_recent_win_rate'] - final_ml_data['a_recent_win_rate']
final_ml_data['days_since_last_play_diff'] = final_ml_data['days_since_last_play_h'] - final_ml_data['days_since_last_play_a']
final_ml_data['defensive_strength_diff'] = abs(final_ml_data['h_clean_sheets_last_5'] - final_ml_data['a_clean_sheets_last_5'])
final_ml_data['defensive_interaction'] = final_ml_data['h_clean_sheets_last_5'] * data['a_clean_sheets_last_5']
final_ml_data['defensive_interaction'] = final_ml_data['h_clean_sheets_last_5'] * data['a_clean_sheets_last_5']
# Drop the original home and away columns if needed (optional)
final_ml_data = final_ml_data[['date', 'season', 'team_h', 'team_a','team_streak_diff','goals_rolling_diff','league_position_diff','league_points_diff',
                               'team_streak_diff','goals_rolling_diff', 'xg_rolling_diff', 'deep_rolling_diff', 'ppda_rolling_diff' 
                               ,  'recent_win_rate_diff','days_since_last_play_diff','h_h2h_wins_last_5','h_matches_played',
                               'elo_diff','defensive_interaction','defensive_strength_diff',
                               'dependence_rho','home_advantage'
                               
                               'target']]



final_ml_data['month'] = pd.to_datetime(final_ml_data['date']).dt.month
final_ml_data['deep_rolling_diff'] = np.where(final_ml_data['deep_rolling_diff']>=8.8,8.8,final_ml_data['deep_rolling_diff'])
final_ml_data['xg_rolling_diff'] = np.where(final_ml_data['xg_rolling_diff']>=2,2,final_ml_data['xg_rolling_diff'])
#final_ml_data['league_position_diff'] = np.where(final_ml_data['league_position_diff']>=15,15,final_ml_data['league_position_diff'])
#final_ml_data['elo_diff'] = np.where(final_ml_data['elo_diff']>=200,200,final_ml_data['elo_diff'])
#final_ml_data['league_points_diff'] = np.where(final_ml_data['league_points_diff']>=47,47,final_ml_data['league_points_diff'])

#final_ml_data= final_ml_data[final_ml_data['target']!=1]
#final_ml_data['target']= final_ml_data['target'].replace(2,1)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 1. Split data into train/test FIRST (to avoid data leakage)
train_data = final_ml_data[final_ml_data['date'] < split_date]
test_data = final_ml_data[final_ml_data['date'] >= split_date]

# 2. Define features to scale
features_to_scale = ['xg_rolling_diff', 'deep_rolling_diff', 'league_position_diff', 'home_advantage',
                     'h_h2h_wins_last_5', 'defensive_strength_diff']

# 3. Initialize scaler and fit ONLY on training data
scaler = StandardScaler()
scaler.fit(train_data[features_to_scale])  # Fit only on training data

# 4. Transform both train and test data
X_train_scaled = scaler.transform(train_data[features_to_scale])
X_test_scaled = scaler.transform(test_data[features_to_scale])

# 5. Prepare target variables
y_train = train_data['target']
y_test = test_data['target']


y_train = y_train.astype(int)  # Ensure the target is in integer format
y_test = y_test.astype(int)  # Ensure the target is in integer format



#%%
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import statsmodels.api as sm

print('Logistic Regression Model')
# Convert 'result' to categorical values (e.g., 0: 'win', 1: 'draw', 2: 'lose')

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority',random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)


# Initialize the Logistic Regression model
model = LogisticRegression(solver='liblinear', max_iter=1000)

# Initialize OneVsRestClassifier with Logistic Regression
#model = OneVsRestClassifier(logreg)

# Fit the model
model.fit(X_train_scaled, y_train)

# Predict the outcomes (home win, draw, away win)
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  # For multiclass, use weighted average
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score (Weighted): {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Get the coefficients for each class
print("Coefficients for each class:")
print(model.coef_)

# Get the intercepts for each class
print("Intercepts for each class:")
print(model.intercept_)
# %%

print('Save Test result')

# Merge the predictions with the original test set data
# Merge the team and date columns from the original test data
test_data_with_preds = test_data.copy()

# Assuming test_data contains the original team and date information (you should load or pass the test data)
test_data_with_preds['team_h'] = test_data['team_h'].values
test_data_with_preds['team_a'] = test_data['team_a'].values
test_data_with_preds['date'] = test_data['date'].values

# Get predicted probabilities for each class (home win, draw, away win)
probabilities = model.predict_proba(X_test_scaled)

# Add the actual, predicted, and probability columns
test_data_with_preds['actual'] = y_test
test_data_with_preds['predicted'] = y_pred

# Add the probabilities for each class (home win, draw, away win)
test_data_with_preds['home_win_prob'] = probabilities[:, 0]
test_data_with_preds['draw_prob'] = probabilities[:, 1]
test_data_with_preds['away_win_prob'] = probabilities[:, 2]

test_data_with_preds.to_csv('data.csv')
# %%
print('save pickle files...')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle


# Initialize scaler and fit on the entire dataset
scaler = StandardScaler()
scaler.fit(final_ml_data[features_to_scale])  # Fit on the entire dataset

# Transform the entire dataset
X_scaled = scaler.transform(final_ml_data[features_to_scale])

# Prepare target variable
y = final_ml_data['target'].astype(int)  # Ensure the target is in integer format

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='minority', random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Initialize and fit the Logistic Regression model
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_scaled, y)

# Save the model and scaler as pickle files
with open('logistic_regression_model_pl.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler_pl.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as pickle files.")