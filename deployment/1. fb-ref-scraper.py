import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import glob

print("Scraping football data from FBRef...")
# URL of the website to scrape
url = "https://fbref.com/en/comps/9/Premier-League-Stats"

# Send an HTTP GET request to the URL
response = requests.get(url)
response.raise_for_status()  # Check for request errors

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all table elements on the page
tables = soup.find_all("table")

# Check if tables were found
if not tables:
    print("No tables found on the page.")
else:
    print(f"Found {len(tables)} tables on the page.")

    # Ensure the directory exists
    directory = "./Football_data/FBRef/raw"
    os.makedirs(directory, exist_ok=True)
    # Delete all existing files in the directory
    files = glob.glob(f"{directory}/*")
    for f in files:
        os.remove(f)
    print(f"Deleted all files in {directory}.")

    # Loop through each table and save it as a CSV file
    for i, table in enumerate(tables):
        # Get the table title if available
        title_tag = table.find_previous("h2") or table.find_previous("h3") or table.find_previous("h4")
        title = title_tag.get_text(strip=True) if title_tag else f"table_{i+1}"
        # Clean the title to use it as a filename
        safe_title = "_".join(title.split()).replace("/", "-")

        # Generate a unique filename with a counter if needed
        csv_filename = os.path.join(directory, f"{safe_title}.csv")
        counter = 2
        while os.path.exists(csv_filename):
            csv_filename = os.path.join(directory, f"{safe_title}_{counter}.csv")
            counter += 1

        # Use pandas to parse the HTML table
        try:
            df = pd.read_html(str(table))[0]

            # Check if the table has two levels of headers
            if isinstance(df.columns, pd.MultiIndex):
                # Merge multi-level headers, ignoring "Unnamed" columns
                df.columns = [
                    ' '.join(col).strip() if "Unnamed" not in col[0] else col[1].strip()
                    for col in df.columns.values
                ]

            # Save the DataFrame as a CSV file
            df.to_csv(csv_filename, index=False)
            print(f"Table saved as {csv_filename}")
        except ValueError as e:
            print(f"Skipping table {i+1}: {e}")

print("Scraping completed.")
#%%

import pandas as pd
import os
import json

# Path to your CSV files
directory = "./Football_data/FBRef/raw"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize an empty dictionary to hold the data
combined_data = {}

# Loop through each CSV file, read it, and store it in the dictionary
for file in csv_files:
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(os.path.join(directory, file))
    
    # Create a key for each table, using the filename (without extension)
    table_name = os.path.splitext(file)[0]
    
    # Convert the DataFrame to a dictionary (e.g., list of rows)
    combined_data[table_name] = df.to_dict(orient='records')

# Save the combined data as a JSON file
with open(os.path.join("./Football_data/FBRef/", 'combined_football_data.json'), 'w') as json_file:
    json.dump(combined_data, json_file, indent=4)

print("Combined data saved as 'combined_football_data.json'")

# %%
print("Downloading future fixtures...")
import requests
import os

def download_csv(url, save_path):
    """
    Downloads a CSV file from the given URL and saves it to the specified path.

    :param url: The URL of the CSV file to download.
    :param save_path: The local path to save the downloaded file.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for unsuccessful status codes

        with open(save_path, 'wb') as file:
            file.write(response.content)

        print(f"File downloaded successfully and saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")

if __name__ == "__main__":
    # URL of the CSV file to download
    csv_url = "https://www.football-data.co.uk/fixtures.csv"

    # Path to save the downloaded CSV file
    base_directory = os.path.expanduser("./Football_data/fixtures")  # Change this as needed
    os.makedirs(base_directory, exist_ok=True)  # Create the directory if it doesn't exist

    save_file_name = "fixtures.csv"
    save_path = os.path.join(base_directory, save_file_name)

    # Download the CSV file
    download_csv(csv_url, save_path)
# %%
print("Download complete.")