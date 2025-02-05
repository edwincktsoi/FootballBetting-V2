import subprocess


#def install_requirements(requirements_file="requirements.txt"):
#    try:
#        # Run pip install -r requirements.txt
#        result = subprocess.run(
#            ["pip", "install", "-r", requirements_file],
#            check=True,
#            text=True,
#            capture_output=True
#        )
#        print("Requirements installed successfully:")
#        print(result.stdout)
#    except subprocess.CalledProcessError as e:
#        print("An error occurred while installing requirements:")
#        print(e.stderr)

# Call the function
#install_requirements()

# List of scripts with numbers
scripts = [
    "1. fb-ref-scraper.py",
    "2. get_under_stats_data.py",
    "3. ML Prediction.py",
    "4. get_football_news.py",
  #  "5. AI-fbref-analyst.py",
  #  "6. modified_Kelly_Cretirion.py",
]

for script in scripts:
    print(f"Running {script}...")
    try:
        subprocess.run(["python", script], check=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}\n")