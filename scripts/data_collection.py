# FaceCue ML - Data Collection Script
# This script helps collect and organize datasets for health deficiency prediction

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import zipfile
import urllib.request

class DataCollector:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def create_synthetic_lifestyle_data(self, n_samples=500):
        """Create synthetic lifestyle data for prototyping"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 65, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'diet_type': np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan'], n_samples),
            'water_intake_liters': np.random.normal(2.5, 0.8, n_samples).round(1),
            'sleep_hours': np.random.normal(7.5, 1.2, n_samples).round(1),
            'fatigue_scale': np.random.randint(1, 6, n_samples),
            'energy_scale': np.random.randint(1, 6, n_samples),
            'screen_time_hours': np.random.normal(6, 2, n_samples).round(1),
            'exercise_hours_week': np.random.normal(3, 2, n_samples).round(1),
            'stress_level': np.random.randint(1, 6, n_samples),
            'smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
            'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        }
        
        # Create target labels based on lifestyle factors
        labels = []
        for i in range(n_samples):
            # Simple rule-based labeling for demonstration
            if data['water_intake_liters'][i] < 1.5 and data['fatigue_scale'][i] > 3:
                labels.append('Dehydration')
            elif data['sleep_hours'][i] < 6 and data['energy_scale'][i] < 3:
                labels.append('Sleep Deficiency')
            elif data['diet_type'][i] == 'Vegan' and np.random.random() < 0.3:
                labels.append('Vitamin Deficiency')
            elif data['fatigue_scale'][i] > 4 and data['energy_scale'][i] < 2:
                labels.append('Anemia')
            else:
                labels.append('Normal')
        
        data['health_status'] = labels
        
        df = pd.DataFrame(data)
        return df
    
    def download_kaggle_dataset(self, dataset_name, file_name=None):
        """Download dataset from Kaggle (requires kaggle API setup)"""
        try:
            import kaggle
            # This requires kaggle API credentials
            print(f"Downloading {dataset_name} from Kaggle...")
            # kaggle.api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)
            print("Kaggle API setup required. Please install kaggle and configure API key.")
        except ImportError:
            print("Kaggle package not installed. Install with: pip install kaggle")
    
    def get_dataset_urls(self):
        """Return URLs for publicly available health datasets"""
        return {
            'lifestyle_survey': 'https://raw.githubusercontent.com/example/lifestyle-data.csv',
            'nutrition_data': 'https://raw.githubusercontent.com/example/nutrition-survey.csv',
            'health_indicators': 'https://raw.githubusercontent.com/example/health-indicators.csv'
        }
    
    def create_data_summary(self, df):
        """Create summary statistics for the dataset"""
        summary = {
            'total_samples': len(df),
            'features': list(df.columns),
            'target_distribution': df['health_status'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        return summary

def main():
    collector = DataCollector()
    
    print("=== FaceCue ML Data Collection ===")
    print("1. Creating synthetic lifestyle data...")
    
    # Create synthetic data for prototyping
    synthetic_data = collector.create_synthetic_lifestyle_data(500)
    synthetic_data.to_csv('data/synthetic_lifestyle_data.csv', index=False)
    print(f"✓ Created synthetic dataset with {len(synthetic_data)} samples")
    
    # Create summary
    summary = collector.create_data_summary(synthetic_data)
    print("\nDataset Summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Features: {len(summary['features'])}")
    print(f"Target distribution: {summary['target_distribution']}")
    
    print("\n=== Recommended Datasets to Download ===")
    print("1. Kaggle Datasets:")
    print("   - 'nutrition-health-survey' by health-org")
    print("   - 'lifestyle-wellness-data' by wellness-research")
    print("   - 'medical-questionnaire-responses' by medical-data")
    
    print("\n2. Public Health Datasets:")
    print("   - CDC Health Survey Data")
    print("   - WHO Global Health Observatory")
    print("   - NHANES (National Health and Nutrition Examination Survey)")
    
    print("\n3. Facial Image Datasets:")
    print("   - UTKFace Dataset (20,000+ face images)")
    print("   - FFHQ Dataset (high-quality faces)")
    print("   - Fitzpatrick17k (skin analysis)")
    
    print("\n=== Next Steps ===")
    print("1. Set up Kaggle API for dataset downloads")
    print("2. Download recommended datasets")
    print("3. Preprocess and clean the data")
    print("4. Create feature engineering pipeline")
    print("5. Train initial model with lifestyle data")

if __name__ == "__main__":
    main()
