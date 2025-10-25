# Kaggle API Setup Script for FaceCue ML
# This script helps set up Kaggle API for downloading datasets

import os
import json
from pathlib import Path

def setup_kaggle_api():
    """Guide user through Kaggle API setup"""
    print("=== Kaggle API Setup for FaceCue ML ===")
    print("\nStep 1: Get Kaggle API Credentials")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    
    print("\nStep 2: Install Kaggle Package")
    print("Run: pip install kaggle")
    
    print("\nStep 3: Place API Credentials")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    print(f"Place kaggle.json in: {kaggle_dir}")
    print("Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    print("\nStep 4: Test API Connection")
    print("Run: kaggle datasets list")
    
    return kaggle_dir

def get_recommended_datasets():
    """Return list of recommended datasets for FaceCue ML"""
    datasets = {
        "lifestyle_health": [
            "nutrition-health-survey",
            "lifestyle-wellness-data", 
            "medical-questionnaire-responses",
            "sleep-health-lifestyle",
            "diet-nutrition-analysis"
        ],
        "facial_images": [
            "utkface-dataset",
            "ffhq-dataset", 
            "fitzpatrick17k-skin-analysis",
            "celeba-dataset"
        ],
        "medical_surveys": [
            "nhanes-health-survey",
            "cdc-health-statistics",
            "who-global-health-data"
        ]
    }
    return datasets

def download_datasets():
    """Download recommended datasets"""
    datasets = get_recommended_datasets()
    
    print("\n=== Recommended Dataset Downloads ===")
    print("\n1. Lifestyle & Health Datasets:")
    for dataset in datasets["lifestyle_health"]:
        print(f"   kaggle datasets download -d {dataset}")
    
    print("\n2. Facial Image Datasets:")
    for dataset in datasets["facial_images"]:
        print(f"   kaggle datasets download -d {dataset}")
    
    print("\n3. Medical Survey Datasets:")
    for dataset in datasets["medical_surveys"]:
        print(f"   kaggle datasets download -d {dataset}")

def create_download_script():
    """Create automated download script"""
    script_content = '''#!/bin/bash
# Automated dataset download script for FaceCue ML

echo "Downloading Lifestyle & Health Datasets..."

# Lifestyle datasets
kaggle datasets download -d nutrition-health-survey -p data/kaggle/
kaggle datasets download -d lifestyle-wellness-data -p data/kaggle/
kaggle datasets download -d medical-questionnaire-responses -p data/kaggle/
kaggle datasets download -d sleep-health-lifestyle -p data/kaggle/
kaggle datasets download -d diet-nutrition-analysis -p data/kaggle/

echo "Downloading Facial Image Datasets..."

# Facial image datasets  
kaggle datasets download -d utkface-dataset -p data/images/
kaggle datasets download -d ffhq-dataset -p data/images/
kaggle datasets download -d fitzpatrick17k-skin-analysis -p data/images/

echo "Extracting downloaded files..."
find data/ -name "*.zip" -exec unzip -o {} -d data/ \\;

echo "Dataset download complete!"
'''
    
    with open('scripts/download_datasets.sh', 'w') as f:
        f.write(script_content)
    
    print("✓ Created download script: scripts/download_datasets.sh")

if __name__ == "__main__":
    setup_kaggle_api()
    download_datasets()
    create_download_script()
    
    print("\n=== Quick Start Commands ===")
    print("1. pip install kaggle")
    print("2. Set up kaggle.json credentials")
    print("3. python scripts/kaggle_setup.py")
    print("4. bash scripts/download_datasets.sh")
