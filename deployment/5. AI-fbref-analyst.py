from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import os
import pandas as pd
from datetime import datetime

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDzk8hcv-niskBel3kNOpbknzMX46nwaRQ'

# Initialize the language model
llm = GoogleGenerativeAI(model="gemini-1.5-flash")

# Load and preprocess the data
directory = "./Football_data/FBRef/raw"
data = pd.read_csv(f"{directory}/Premier_League_2.csv")
data1 = pd.read_csv(f"{directory}/Premier_League.csv")

prediction = pd.read_csv('./Football_data/output/predictions.csv')
prediction = prediction[['Date', 'HomeTeam', 'AwayTeam', 'home_win_prob', 'draw_prob', 'away_win_prob']]

data = pd.merge(data, data1, on=['Squad', 'Rk'])

# Create team name mapping dictionary
team_name_mapping = {
    'Liverpool': 'Liverpool',
    'Arsenal': 'Arsenal',
    'Nottingham Forest': "Nott'ham Forest",
    'Chelsea': 'Chelsea',
    'Newcastle United': 'Newcastle Utd',
    'Manchester City': 'Manchester City',
    'Bournemouth': 'Bournemouth',
    'Aston Villa': 'Aston Villa',
    'Fulham': 'Fulham',
    'Brighton': 'Brighton',
    'Brentford': 'Brentford',
    'Tottenham': 'Tottenham',
    'Manchester United': 'Manchester Utd',
    'West Ham': 'West Ham',
    'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton',
    'Wolverhampton Wanderers': 'Wolves',
    'Ipswich': 'Ipswich Town',
    'Leicester': 'Leicester City',
    'Southampton': 'Southampton'
}

# Function to standardize team names without converting to lowercase
def standardize_team_name(team_name, mapping):
    return mapping.get(team_name, team_name)

# Process each row in the CSV file
for _, row in prediction.iterrows():
    # Extract match details
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    match_date = row['Date']

    # Standardize the team names without converting to lowercase
    team_1 = standardize_team_name(home_team, team_name_mapping)
    team_2 = standardize_team_name(away_team, team_name_mapping)

    # Get probabilities
    home_probability = row['home_win_prob']
    draw_probability = row['draw_prob']
    away_probability = row['away_win_prob']

    # Filter data for the two teams
    team_1_data = data[data['Squad'] == team_1]
    team_2_data = data[data['Squad'] == team_2]

    # Summarize home and away statistics for each team
    #team_1_summary = team_1_data[['Home xG', 'Home xGA', 'Home xGD', 'Home xGD/90', 'Home Pts', 'Last 5']]
    #team_2_summary = team_2_data[['Away xG', 'Away xGA', 'Away xGD', 'Away xGD/90', 'Away Pts', 'Last 5']]
    team_1_summary = team_1_data[['Home xGA', 'Home Pts', 'Last 5']]
    team_2_summary = team_2_data[['Away xGA', 'Away Pts', 'Last 5']]

    # Convert summaries to strings for the prompt
    team_1_summary_str = team_1_summary.to_string(index=False)
    team_2_summary_str = team_2_summary.to_string(index=False)

    # Define LangChain prompt to analyze and report for the match
    prompt_template = f"""
        You are an expert football analyst, specializing in match performance evaluation and generating comprehensive reports.

        Below is the summarized performance data for the two teams involved in the match:

        - **Match Date:** {match_date}

        - **{team_1} (Home and Away Stats):**
        {team_1_summary_str}

        - **{team_2} (Home and Away Stats):**
        {team_2_summary_str}

        Additionally, the pre-match probabilities are as follows:

        - **Home Team Win Probability:** {home_probability * 100}%  
        - **Away Team Win Probability:** {away_probability * 100}%  
        - **Draw Probability:** {(away_probability) * 100}%

        Based on the provided data and probabilities, please:

        1. **Compare Team Performance:** Analyze the key performance metrics for both teams and provide a comparative analysis, including a side-by-side table highlighting key stats for each team.

        2. **Home and Away Advantage:** Evaluate the home and away performance factors for both teams, identifying any advantages or challenges based on their historical data.

        3. **Adjust Winning Probability:** Adjust the initial winning probabilities for both teams based on their current form, historical data, and performance metrics.

        4. **Probability Breakdown:** Present the adjusted probabilities of each team winning sightly not too much, along with the likelihood of a draw. Include a table with these probabilities for clarity.

        5. **Detailed Professional Analysis:** Provide an in-depth analysis of the match with clear headings and explanations, covering all aspects of team performance, strategy, and prediction.

        Ensure the report is clear, concise, and uses professional football terminology to describe the teams' current form, strengths, weaknesses, and overall chances in this match.
    """

    # Initialize LangChain prompt and LLM chain
    prompt = PromptTemplate(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate the report
    report = chain.run(input={})

    # Print the report to console
    print(f"\nAnalysis for {home_team} vs {away_team} on {match_date}:")
    print(report)

    # Save the result as a text file
    match_date = match_date.replace('/', '-')
    dated_directory = f"./Football_data/output/{match_date}/{home_team}_vs_{away_team}"
    os.makedirs(dated_directory, exist_ok=True)

    output_filename = f"{dated_directory}/data_report.txt"
    with open(output_filename, 'w') as f:
        f.write(report)

    print(f"Report saved as {output_filename}")
