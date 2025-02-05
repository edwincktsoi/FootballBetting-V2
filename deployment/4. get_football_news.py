from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper
import os 
from bs4 import BeautifulSoup
import requests
import os
import pandas as pd
import logging

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDzk8hcv-niskBel3kNOpbknzMX46nwaRQ'
os.environ['SERPAPI_API_KEY'] = '3142ef9693996bf4c15230449c0f185443b2ac44bd44d443f00f7d9617ff5138'  # Replace with your SerpAPI key
os.environ["GOOGLE_CSE_ID"] = "62388394eef9240b5"

# Example for reading fixtures data
prediction = pd.read_csv('./Football_data/output/predictions.csv')
prediction = prediction[['Date', 'HomeTeam', 'AwayTeam', 'home_win_prob', 'draw_prob', 'away_win_prob']]


# Initialize LLM and SerpAPI
llm = GoogleGenerativeAI(model="gemini-1.5-flash")
serp_api = SerpAPIWrapper()



def fetch_urls(query, num_results=10):
    try:
        search_results = serp_api.run(query)
        if isinstance(search_results, list):
            urls = [result.get('link') for result in search_results[:num_results] if 'link' in result]
            return urls
        else:
            logging.warning(f"Unexpected response from SerpAPI: {search_results}")
            return []
    except Exception as e:
        logging.error(f"Error fetching URLs: {e}")
        return []

def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text() for p in paragraphs])
        return content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Error parsing {url}: {e}")
        return ""

def generate_consolidated_summary(query, num_websites=10):
    urls = fetch_urls(query, num_websites)
    if not urls:
        logging.warning(f"No URLs found for query: {query}")
        return "No URLs found", {}

    combined_content = ""
    for url in urls:
        content = scrape_website(url)
        if content:
            combined_content += f"Content from {url}:\n{content}\n\n"

    if not combined_content:
        logging.warning(f"No content fetched for query: {query}")
        return "No content fetched", {}

    chain = LLMChain(llm=llm, prompt=performance_analysis_prompt)
    try:
        summary = chain.run(combined_content=combined_content)
        adjusted_probs = parse_adjusted_probabilities(summary)
        return summary, adjusted_probs
    except Exception as e:
        logging.error(f"Error running LLM chain: {e}")
        return "Error in analysis", {}

def parse_adjusted_probabilities(summary):
    adjustments = {"availability": 0, "tactics": 0, "morale": 0}
    impact_words = {"High": 0.05, "Medium": 0.02, "Low": 0.01}

    for factor in adjustments:
        for impact, value in impact_words.items():
            if f"{factor.capitalize()}: {impact}" in summary:
                adjustments[factor] = value
                break  # Stop checking impacts once one is found

    return adjustments


# Adjusted probabilities storage
adjusted_probabilities = []

# Main execution
if __name__ == "__main__":
    prediction = pd.read_csv('./Football_data/output/predictions.csv')
    prediction = prediction[['Date', 'HomeTeam', 'AwayTeam', 'home_win_prob', 'draw_prob', 'away_win_prob']]

    for index, match in prediction.iterrows():
        # ... (rest of the code for fetching data, generating query, etc.)
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date'].replace('/', '-')
        home_prob = match['home_win_prob']
        away_prob = match['away_win_prob']
        draw_prob = match['draw_prob']
        
        # Prompt templates
        performance_analysis_prompt = PromptTemplate(
        input_variables=["combined_content",'HomeTeam', "home_prob", 'AwayTeam',"away_prob", "draw_prob"],
        template=(
        f"You are a football performance analyst. "
        "Based on the following content collected from multiple websites, "
        "summarize the news into a concise report. Evaluate the potential impact of the news on the team's performance "
        "using factors like player availability, tactical changes, and morale. Provide an impact score for each factor "
        "(Low, Medium, High), and calculate an overall probability of how this affects the team's chances of winning in a table.\n\n"

        "Additionally, the pre-match probabilities are as follows:"
        f"- **Home Team Win Probability:** {home_team}={home_prob * 100}%  "
        f"- **Draw Probability:** {(draw_prob) * 100}%"
        f"- **Away Team Win Probability:**{away_team} = {away_prob * 100}%  "
        
        "{combined_content}\n\n"
        "Your response should include:\n"
        "- Summary of key points.\n"
        "- Impact evaluation on performance (availability, tactics, morale).\n"
        "- Estimated probability change of winning, with an adjusted table that includes the probabilities for each team (including draw).\n"
        "- Adjust Winning Probability:**Based on the provided probabilities, please: Adjust the initial winning probabilities for both teams based on their current form, player availability, tactical changes, and morale. and create a table for each team's adjusted probability of winning in this fashion | Initial Win Probability | Adjusted Win Probability | Change in Probability ."
    )
)

        query = f"{home_team} vs {away_team} latest team news"
        report, adjustments = generate_consolidated_summary(query)
        print(f"Fetching URLs for: {query}")
        urls = fetch_urls(query)

        if urls:
            print(f"Fetched URLs for {home_team} vs {away_team}:")
            for url in urls:
                print(url)
        else:
            print(f"No URLs found for {home_team} vs {away_team}")

        home_adjustment = adjustments.get("availability", 0) + adjustments.get("tactics", 0)
        away_adjustment = adjustments.get("morale", 0) + adjustments.get("tactics",0)

        adjusted_home_prob = max(0, min(1, home_prob + home_adjustment))  # Clamp between 0 and 1
        adjusted_away_prob = max(0, min(1, away_prob + away_adjustment))
        adjusted_draw_prob = max(0, min(1, 1 - adjusted_home_prob - adjusted_away_prob))

        # ... (rest of the code for saving the report and printing probabilities)
        dated_directory = f"./Football_data/output/{match_date}/{home_team}_vs_{away_team}"
        os.makedirs(dated_directory, exist_ok=True)

        output_filename = f"{dated_directory}/news_report.txt"
        with open(output_filename, 'w') as f:
            f.write(report)
    
        adjusted_home_prob = max(0, min(1, home_prob + home_adjustment))
        adjusted_away_prob = max(0, min(1, away_prob + away_adjustment))
        adjusted_draw_prob = max(0, min(1, 1 - adjusted_home_prob - adjusted_away_prob))
        
        adjusted_probabilities.append({
            'Date': match_date,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Initial Home Win Probability': round(home_prob, 4)*100,
            'Initial Draw Probability': round(draw_prob, 4)*100,
            'Initial Away Win Probability': round(away_prob, 4)*100,
            'Adjusted Home Win Probability': round(adjusted_home_prob, 4)*100,
            'Adjusted Draw Probability': round(adjusted_draw_prob, 4)*100,
            'Adjusted Away Win Probability': round(adjusted_away_prob, 4)*100
        })

        # Save to CSV
        adjusted_probabilities_df = pd.DataFrame(adjusted_probabilities)
        output_csv_path = f'./Football_data/output/{match_date}/adjusted_probabilities.csv'
        adjusted_probabilities_df.to_csv(output_csv_path, index=False)

        print(f"Adjusted probabilities saved to {output_csv_path}")