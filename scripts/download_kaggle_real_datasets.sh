#!/bin/bash
# Kaggle Dataset Download Script for FaceCue ML
# Run this after setting up Kaggle API credentials

echo "=== FaceCue ML - Real Dataset Downloader ==="
echo "Downloading real health and lifestyle datasets..."

# Create directories
mkdir -p data/kaggle
mkdir -p data/images

echo "Downloading Sleep Health and Lifestyle Dataset..."
kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset -p data/kaggle/

echo "Downloading Nutrition Health Survey Dataset..."
kaggle datasets download -d nutrition-health-survey -p data/kaggle/

echo "Downloading Lifestyle Wellness Dataset..."
kaggle datasets download -d lifestyle-wellness-data -p data/kaggle/

echo "Downloading Medical Questionnaire Dataset..."
kaggle datasets download -d medical-questionnaire-responses -p data/kaggle/

echo "Extracting downloaded files..."
find data/kaggle/ -name "*.zip" -exec unzip -o {} -d data/kaggle/ \;

echo "=== Download Complete ==="
echo "Real datasets downloaded to data/kaggle/"
echo "Ready for FaceCue ML analysis!"
