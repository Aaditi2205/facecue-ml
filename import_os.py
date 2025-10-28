import os
import json
import requests
import pandas as pd
import urllib.request
from pathlib import Path

class RealDatasetDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_direct_datasets(self):
        """Download datasets with direct URLs"""
        print("=== Downloading Direct Available Datasets ===")

        datasets_to_download = [
            {
                "name": "CDC Diabetes Health Indicators",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00536/diabetes_012_health_indicators_BRFSS2015.csv",
                "filename": "cdc_diabetes_indicators.csv",
                "category": "government",
                "description": "CDC diabetes health indicators with lifestyle factors"
            },
            {
                "name": "UCI Heart Disease (Cleveland processed)",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
                "filename": "processed.cleveland.data",
                "category": "academic",
                "description": "UCI Heart Disease processed Cleveland data (no header)"
            },
            {
                "name": "UCI Student Performance (zip)",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
                "filename": "student_performance.zip",
                "category": "academic",
                "description": "Student Performance datasets (math/por) - zip file"
            },
            {
                "name": "LFW (faces)",
                "url": "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
                "filename": "lfw.tgz",
                "category": "images",
                "description": "LFW face images tarball"
            },
            {
                "name": "UTKFace (info page)",
                "url": "https://susanqq.github.io/UTKFace/",
                "filename": "utkface_info.txt",
                "category": "images",
                "description": "UTKFace information page (manual download recommended)"
            }
        ]

        for dataset in datasets_to_download:
            try:
                print(f"\nDownloading {dataset['name']}...")
                if dataset['url'].endswith('.csv') or dataset['url'].endswith('.data'):
                    df = pd.read_csv(dataset['url'], header=None if 'processed.cleveland' in dataset['filename'] else 'infer')
                    output_path = self.data_dir / dataset['category'] / dataset['filename'].replace('.data', '.csv')
                    df.to_csv(output_path, index=False)
                    print(f"✓ Downloaded: {df.shape[0]} samples, {df.shape[1]} features")
                    print(f"  Saved to: {output_path}")
                elif dataset['url'].endswith('.zip') or dataset['url'].endswith('.tgz') or dataset['url'].endswith('.tgz'):
                    output_path = self.data_dir / dataset['category'] / dataset['filename']
                    urllib.request.urlretrieve(dataset['url'], output_path)
                    print(f"✓ Downloaded archive: {output_path}")
                else:
                    # For info pages or non-direct downloads, save a small fetch of the HTML/info as reference
                    try:
                        resp = requests.get(dataset['url'], timeout=20)
                        out = self.data_dir / dataset['category'] / dataset['filename']
                        out.write_text(resp.text[:10000], encoding='utf-8')
                        print(f"✓ Saved info page for {dataset['name']} -> {out}")
                    except Exception:
                        print(f"⚠️ Could not fetch content for {dataset['name']}, saved metadata only.")
            except Exception as e:
                print(f"❌ Failed to download {dataset['name']}: {e}")

    # New helper: scan data folder and create a simple inventory (samples/features)
    def create_dataset_inventory(self, output_file="data/dataset_inventory.json"):
        """Scan data/ subfolders and create a JSON inventory for CSV files"""
        inventory = {}
        for sub in ["academic", "government", "kaggle", "images"]:
            dirpath = self.data_dir / sub
            if not dirpath.exists():
                continue
            for file in dirpath.glob("**/*"):
                if file.is_file() and file.suffix.lower() in ['.csv', '.tsv', '.txt']:
                    try:
                        if file.suffix.lower() == '.csv':
                            df = pd.read_csv(file, nrows=5)
                            df_full = pd.read_csv(file)
                            inventory[file.as_posix()] = {
                                "samples": len(df_full),
                                "features": len(df_full.columns),
                                "path": file.as_posix()
                            }
                            print(f"Inventory: {file.name} -> {inventory[file.as_posix()]['samples']} samples, {inventory[file.as_posix()]['features']} features")
                        else:
                            inventory[file.as_posix()] = {"path": file.as_posix(), "note": "non-csv/text file"}
                    except Exception as e:
                        inventory[file.as_posix()] = {"path": file.as_posix(), "error": str(e)}
        # Write JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(inventory, f, indent=2)
        print(f"✓ Created dataset inventory: {output_file}")
        return output_file

    # ...existing code...