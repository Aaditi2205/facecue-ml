# FaceCue ML - Real Dataset Collector
# Downloads real health and lifestyle datasets from various sources

import pandas as pd
import numpy as np
import requests
import os
import zipfile
import urllib.request
from pathlib import Path
import json
import time

class RealDatasetCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "kaggle").mkdir(exist_ok=True)
        (self.data_dir / "government").mkdir(exist_ok=True)
        (self.data_dir / "academic").mkdir(exist_ok=True)
        (self.data_dir / "images").mkdir(exist_ok=True)
        
    def download_kaggle_datasets(self):
        """Download specific Kaggle datasets for health/lifestyle analysis"""
        kaggle_datasets = {
            "sleep-health-lifestyle": {
                "url": "https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset",
                "description": "Sleep patterns, lifestyle habits, and health indicators"
            },
            "nutrition-health-survey": {
                "url": "https://www.kaggle.com/datasets/nutrition-health-survey",
                "description": "Comprehensive nutrition and health survey data"
            },
            "lifestyle-wellness": {
                "url": "https://www.kaggle.com/datasets/lifestyle-wellness-data",
                "description": "Daily lifestyle and wellness metrics"
            },
            "medical-questionnaire": {
                "url": "https://www.kaggle.com/datasets/medical-questionnaire-responses",
                "description": "Medical questionnaire responses and health status"
            },
            "diet-nutrition-analysis": {
                "url": "https://www.kaggle.com/datasets/diet-nutrition-analysis",
                "description": "Dietary patterns and nutritional analysis"
            }
        }
        
        print("=== Kaggle Datasets to Download ===")
        for name, info in kaggle_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")
            print(f"  Command: kaggle datasets download -d {name} -p data/kaggle/")
        
        return kaggle_datasets
    
    def download_government_datasets(self):
        """Download government health datasets"""
        gov_datasets = {
            "cdc_diabetes": {
                "url": "https://archive.ics.uci.edu/ml/datasets/Diabetes+Health+Indicators+Dataset",
                "description": "CDC diabetes health indicators with lifestyle factors",
                "direct_download": "https://archive.ics.uci.edu/ml/machine-learning-databases/00536/diabetes_012_health_indicators_BRFSS2015.csv"
            },
            "nhanes_nutrition": {
                "url": "https://www.cdc.gov/nchs/nhanes/",
                "description": "National Health and Nutrition Examination Survey",
                "note": "Requires registration and approval"
            },
            "brfss_lifestyle": {
                "url": "https://www.cdc.gov/brfss/annual_data/annual_data.htm",
                "description": "Behavioral Risk Factor Surveillance System",
                "note": "Annual lifestyle and health survey data"
            }
        }
        
        print("\n=== Government Health Datasets ===")
        for name, info in gov_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")
            if 'direct_download' in info:
                print(f"  Direct Download: {info['direct_download']}")
            if 'note' in info:
                print(f"  Note: {info['note']}")
        
        return gov_datasets
    
    def download_academic_datasets(self):
        """Download academic/research datasets"""
        academic_datasets = {
            "uci_heart_disease": {
                "url": "https://archive.ics.uci.edu/ml/datasets/Heart+Disease",
                "description": "Heart disease prediction with lifestyle factors",
                "direct_download": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
            },
            "uci_student_performance": {
                "url": "https://archive.ics.uci.edu/ml/datasets/Student+Performance",
                "description": "Student performance with health/lifestyle attributes",
                "direct_download": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/"
            },
            "maternal_health_risk": {
                "url": "https://archive.ics.uci.edu/ml/datasets/Maternal+Health+Risk",
                "description": "Maternal health risks and lifestyle factors",
                "direct_download": "https://archive.ics.uci.edu/ml/machine-learning-databases/00639/"
            }
        }
        
        print("\n=== Academic Research Datasets ===")
        for name, info in academic_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")
            if 'direct_download' in info:
                print(f"  Direct Download: {info['direct_download']}")
        
        return academic_datasets
    
    def download_facial_image_datasets(self):
        """Download facial image datasets for health analysis"""
        image_datasets = {
            "utkface": {
                "url": "https://susanqq.github.io/UTKFace/",
                "description": "20,000+ face images with age/gender/ethnicity labels",
                "size": "~2GB",
                "features": "Age, gender, ethnicity annotations"
            },
            "ffhq": {
                "url": "https://github.com/NVlabs/ffhq-dataset",
                "description": "High-quality diverse face images",
                "size": "~7GB",
                "features": "High resolution, diverse demographics"
            },
            "fitzpatrick17k": {
                "url": "https://github.com/sfu-mial/awesome-skin-image-analysis-datasets",
                "description": "Clinical images with skin type labels",
                "size": "~500MB",
                "features": "Skin type classification, clinical annotations"
            },
            "celeba": {
                "url": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
                "description": "Celebrity faces with attribute annotations",
                "size": "~1.3GB",
                "features": "40+ facial attributes"
            }
        }
        
        print("\n=== Facial Image Datasets ===")
        for name, info in image_datasets.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  URL: {info['url']}")
            print(f"  Size: {info['size']}")
            print(f"  Features: {info['features']}")
        
        return image_datasets
    
    def download_direct_datasets(self):
        """Download datasets that have direct download links"""
        print("\n=== Downloading Direct Datasets ===")
        
        # CDC Diabetes Health Indicators Dataset
        try:
            print("Downloading CDC Diabetes Health Indicators...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00536/diabetes_012_health_indicators_BRFSS2015.csv"
            df = pd.read_csv(url)
            df.to_csv(self.data_dir / "government" / "cdc_diabetes_indicators.csv", index=False)
            print(f"✓ Downloaded CDC dataset: {df.shape[0]} samples, {df.shape[1]} features")
            print(f"  Features: {list(df.columns)[:10]}...")  # Show first 10 features
        except Exception as e:
            print(f"❌ Failed to download CDC dataset: {e}")
        
        # UCI Heart Disease Dataset
        try:
            print("\nDownloading UCI Heart Disease dataset...")
            # Download the processed Cleveland dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            df = pd.read_csv(url, header=None)
            
            # Add column names
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df.columns = columns
            
            df.to_csv(self.data_dir / "academic" / "uci_heart_disease.csv", index=False)
            print(f"✓ Downloaded Heart Disease dataset: {df.shape[0]} samples, {df.shape[1]} features")
            print(f"  Features: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Failed to download Heart Disease dataset: {e}")
    
    def create_download_scripts(self):
        """Create shell scripts for downloading datasets"""
        
        # Kaggle download script
        kaggle_script = '''#!/bin/bash
# Kaggle Dataset Download Script for FaceCue ML

echo "Setting up Kaggle API..."
pip install kaggle

echo "Downloading Kaggle datasets..."

# Health and Lifestyle datasets
kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset -p data/kaggle/
kaggle datasets download -d nutrition-health-survey -p data/kaggle/
kaggle datasets download -d lifestyle-wellness-data -p data/kaggle/
kaggle datasets download -d medical-questionnaire-responses -p data/kaggle/
kaggle datasets download -d diet-nutrition-analysis -p data/kaggle/

# Facial image datasets
kaggle datasets download -d utkface-dataset -p data/images/
kaggle datasets download -d ffhq-dataset -p data/images/

echo "Extracting downloaded files..."
find data/ -name "*.zip" -exec unzip -o {} -d data/ \\;

echo "Kaggle dataset download complete!"
'''
        
        with open('scripts/download_kaggle_datasets.sh', 'w') as f:
            f.write(kaggle_script)
        
        # Python download script
        python_script = '''#!/usr/bin/env python3
# Python Dataset Download Script for FaceCue ML

import subprocess
import os

def run_command(command):
    """Run a shell command"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("=== FaceCue ML Dataset Downloader ===")
    
    # Install required packages
    print("Installing required packages...")
    run_command("pip install kaggle pandas requests")
    
    # Download Kaggle datasets
    print("\\nDownloading Kaggle datasets...")
    kaggle_datasets = [
        "uom190346a/sleep-health-and-lifestyle-dataset",
        "nutrition-health-survey", 
        "lifestyle-wellness-data",
        "medical-questionnaire-responses",
        "diet-nutrition-analysis"
    ]
    
    for dataset in kaggle_datasets:
        run_command(f"kaggle datasets download -d {dataset} -p data/kaggle/")
    
    print("\\nDataset download complete!")

if __name__ == "__main__":
    main()
'''
        
        with open('scripts/download_datasets.py', 'w') as f:
            f.write(python_script)
        
        print("✓ Created download scripts:")
        print("  - scripts/download_kaggle_datasets.sh")
        print("  - scripts/download_datasets.py")
    
    def create_dataset_summary(self):
        """Create a summary of all available datasets"""
        summary = {
            "kaggle_datasets": self.download_kaggle_datasets(),
            "government_datasets": self.download_government_datasets(),
            "academic_datasets": self.download_academic_datasets(),
            "image_datasets": self.download_facial_image_datasets()
        }
        
        with open('data/dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("✓ Created dataset summary: data/dataset_summary.json")
        return summary

def main():
    print("=== FaceCue ML Real Dataset Collector ===")
    print("Collecting real-world health and lifestyle datasets...")
    
    collector = RealDatasetCollector()
    
    # Show available datasets
    collector.download_kaggle_datasets()
    collector.download_government_datasets()
    collector.download_academic_datasets()
    collector.download_facial_image_datasets()
    
    # Download direct datasets
    collector.download_direct_datasets()
    
    # Create download scripts
    collector.create_download_scripts()
    
    # Create summary
    collector.create_dataset_summary()
    
    print("\n=== Next Steps ===")
    print("1. Set up Kaggle API credentials")
    print("2. Run: python scripts/download_datasets.py")
    print("3. Or run: bash scripts/download_kaggle_datasets.sh")
    print("4. Process downloaded datasets with data_exploration.py")
    print("5. Train models with real data")
    
    print("\n=== Kaggle API Setup ===")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Create API token (download kaggle.json)")
    print("3. Place kaggle.json in ~/.kaggle/ directory")
    print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    print("5. Test: kaggle datasets list")

if __name__ == "__main__":
    main()
