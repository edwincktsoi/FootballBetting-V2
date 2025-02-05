import os
import json
import pandas as pd
from understatapi import UnderstatClient
from understatapi.exceptions import InvalidMatch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the Understat client
understat = UnderstatClient()

# Base folder to organize data
base_folder = "./Football_data/understats"
os.makedirs(base_folder, exist_ok=True)

# Subfolders for different data types
folders = {
    "match_info": os.path.join(base_folder, "match_info"),
    "match_shot_data": os.path.join(base_folder, "match_shot_data"),
    "match_roster_data": os.path.join(base_folder, "match_roster_data"),
}

# Create subfolders
for folder in folders.values():
    os.makedirs(folder, exist_ok=True)


def process_match_data(match_id):
    """Fetch and save data for a single match."""
    try:
        print(f"Processing match ID: {match_id}")

        # Fetch data for the current match
        match_info = understat.match(match=match_id).get_match_info()
        match_shot_data = understat.match(match=match_id).get_shot_data()
        match_roster_data = understat.match(match=match_id).get_roster_data()

        # Save match_info as a dictionary
        match_info_file = f"{folders['match_info']}/match_info_{match_id}.json"
        with open(match_info_file, "w") as f:
            json.dump(match_info, f, indent=4)

        # Save match_shot_data and match_roster_data as JSON files
        with open(f"{folders['match_shot_data']}/match_shot_data_{match_id}.json", "w") as f:
            json.dump(match_shot_data, f, indent=4)
        with open(f"{folders['match_roster_data']}/match_roster_data_{match_id}.json", "w") as f:
            json.dump(match_roster_data, f, indent=4)

        print(f"Saved data for match ID: {match_id}")
        return match_info

    except InvalidMatch as e:
        print(f"Skipping invalid match ID {match_id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing match ID {match_id}: {e}")
        return None

all_match_info = []
for season in np.arange(2017, 2025):
    print(f"Processing season: {season}")

    # Fetch match data for the current season
    team_match_data = understat.league(league="EPL").get_match_data(season=str(season))

    # Extract match IDs
    match_ids = [match['id'] for match in team_match_data]

    # List to store match_info data for combining into a single CSV
    

    # Use ThreadPoolExecutor to process matches in parallel
    with ThreadPoolExecutor(max_workers=12) as executor:  # Adjust max_workers based on your system
        future_to_match = {executor.submit(process_match_data, match_id): match_id for match_id in match_ids}

        for future in as_completed(future_to_match):
            match_id = future_to_match[future]
            try:
                result = future.result()
                if result:
                    all_match_info.append(result)
            except Exception as e:
                print(f"Error fetching data for match ID {match_id}: {e}")

# Combine all match_info into a single CSV file
if all_match_info:
    combined_df = pd.DataFrame(all_match_info)
    combined_df.to_csv(f"{folders['match_info']}/combined_match_info.csv", index=False)
    print(f"Combined match_info saved for season {season} as {folders['match_info']}/combined_match_info.csv")
print(f"Completed processing for season: {season}")

print(f"All data saved in the base folder: {base_folder}")
