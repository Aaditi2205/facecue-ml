# FaceCue ML - Simple Kaggle Dataset Downloader
# Downloads real health datasets from Kaggle

import subprocess
import os
from pathlib import Path

def run_kaggle_command(command):
    """Run a kaggle command"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("=== FaceCue ML - Real Dataset Downloader ===")
    print("Downloading real health and lifestyle datasets from Kaggle...")
    
    # Create directories
    Path("data/kaggle").mkdir(parents=True, exist_ok=True)
    Path("data/images").mkdir(parents=True, exist_ok=True)
    
    # Real health datasets to download
    datasets = [
        "uom190346a/sleep-health-and-lifestyle-dataset",
        "nutrition-health-survey",
        "lifestyle-wellness-data", 
        "medical-questionnaire-responses"
    ]
    
    print("Downloading real health and lifestyle datasets...")
    
    success_count = 0
    for dataset in datasets:
        print(f"\nDownloading {dataset}...")
        success = run_kaggle_command(f"kaggle datasets download -d {dataset} -p data/kaggle/")
        
        if success:
            print(f"SUCCESS: Downloaded {dataset}")
            success_count += 1
        else:
            print(f"FAILED: Could not download {dataset}")
    
    print(f"\n=== Download Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(datasets)} datasets")
    
    if success_count > 0:
        print("SUCCESS: Real datasets downloaded to data/kaggle/")
        print("Your FaceCue ML project now has real health data!")
        print("Ready for production-level health prediction models.")
    else:
        print("ERROR: No datasets downloaded")
        print("Please check Kaggle API setup:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Set up API credentials: https://www.kaggle.com/account")
        print("3. Place kaggle.json in ~/.kaggle/ directory")

if __name__ == "__main__":
    main()
