#!/bin/bash
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
find data/ -name "*.zip" -exec unzip -o {} -d data/ \;

echo "Kaggle dataset download complete!"
