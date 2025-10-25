# FaceCue ML - Dataset Collection Guide

## 🎯 Project Scope: Health Deficiency Prediction
**Target Deficiencies:** Anemia, Vitamin D deficiency, Dehydration, General malnutrition
**Input Types:** Lifestyle data + Optional facial images

## 📊 Data Collection Strategy

### Phase 1: Lifestyle Data (Start Here)
**Priority:** HIGH - Build foundation model first

#### 1. Kaggle Datasets (Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Set up API credentials (download from Kaggle account settings)
# Place kaggle.json in ~/.kaggle/ directory
```

**Specific Kaggle Datasets to Download:**
- `nutrition-health-survey` - Comprehensive nutrition data
- `lifestyle-wellness-data` - Daily habits and wellness metrics  
- `medical-questionnaire-responses` - Health survey responses
- `sleep-health-lifestyle` - Sleep patterns and health correlation
- `diet-nutrition-analysis` - Dietary patterns and deficiencies

#### 2. Public Health Datasets
- **CDC Health Survey Data** - National health statistics
- **WHO Global Health Observatory** - International health data
- **NHANES Dataset** - Nutrition and health examination survey
- **BRFSS (Behavioral Risk Factor Surveillance System)** - Lifestyle factors

#### 3. Academic Datasets
- **MIMIC-III Clinical Database** - ICU patient data (requires approval)
- **PhysioNet** - Physiological signal databases
- **OpenNeuro** - Neuroimaging and behavioral data

### Phase 2: Facial Image Data (Optional Enhancement)
**Priority:** MEDIUM - Add after lifestyle model works

#### 1. General Facial Datasets
- **UTKFace Dataset** - 20,000+ face images with age/gender/ethnicity
- **FFHQ Dataset** - High-quality diverse face images
- **CelebA Dataset** - Celebrity faces with attributes

#### 2. Health-Specific Facial Datasets
- **Fitzpatrick17k** - Clinical images with skin type labels
- **Derm7pt** - Skin lesion analysis dataset
- **Facial Parts Segmentation** - Detailed facial feature annotations

## 🛠️ Implementation Steps

### Step 1: Set Up Data Collection Environment
```bash
# Create virtual environment
python -m venv facecue_env
source facecue_env/bin/activate  # On Windows: facecue_env\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install kaggle requests beautifulsoup4
pip install opencv-python pillow  # For image processing
```

### Step 2: Download Datasets
```python
# Run the data collection script
python scripts/data_collection.py
```

### Step 3: Data Preprocessing Pipeline
```python
# Create preprocessing scripts for:
# - Data cleaning and validation
# - Feature engineering
# - Data splitting (train/validation/test)
# - Data augmentation (for images)
```

## 📋 Dataset Requirements

### Lifestyle Data Features (Minimum Required)
- **Demographics:** Age, Gender, Location
- **Diet:** Diet type, Meal frequency, Food preferences
- **Hydration:** Water intake (L/day), Fluid sources
- **Sleep:** Sleep hours, Sleep quality, Sleep schedule
- **Activity:** Exercise frequency, Physical activity level
- **Lifestyle:** Screen time, Stress level, Smoking/Alcohol
- **Symptoms:** Fatigue scale, Energy level, Mood

### Target Labels
- Normal
- Anemia (Iron deficiency)
- Vitamin D Deficiency  
- Dehydration
- General Malnutrition
- Sleep Deficiency

### Facial Image Features (Optional)
- **Skin Analysis:** Tone, texture, pallor
- **Eye Analysis:** Brightness, dark circles, redness
- **Lip Analysis:** Color, dryness, cracks
- **Overall:** Facial symmetry, color distribution

## 🔗 Direct Dataset Links

### Kaggle Datasets
1. **Nutrition & Health Survey:** `https://www.kaggle.com/datasets/nutrition-health-survey`
2. **Lifestyle Wellness Data:** `https://www.kaggle.com/datasets/lifestyle-wellness-data`
3. **Medical Questionnaire:** `https://www.kaggle.com/datasets/medical-questionnaire-responses`

### Public Datasets
1. **CDC Data:** `https://www.cdc.gov/nchs/data_access/ftp_data.htm`
2. **WHO Data:** `https://www.who.int/data/gho`
3. **NHANES:** `https://www.cdc.gov/nchs/nhanes/`

### Facial Image Datasets
1. **UTKFace:** `https://susanqq.github.io/UTKFace/`
2. **FFHQ:** `https://github.com/NVlabs/ffhq-dataset`
3. **Fitzpatrick17k:** `https://github.com/sfu-mial/awesome-skin-image-analysis-datasets`

## ⚠️ Important Considerations

### Ethical & Legal
- Use only publicly available, anonymized datasets
- Respect data usage terms and conditions
- Avoid datasets with personal health information
- Consider bias in datasets (demographic representation)

### Data Quality
- Verify data completeness and accuracy
- Check for missing values and outliers
- Ensure consistent data formats
- Validate target label distributions

### Technical Limitations
- Start with tabular data (easier to process)
- Add image data gradually
- Focus on feature engineering for better results
- Use cross-validation for robust evaluation

## 🚀 Quick Start Commands

```bash
# 1. Run data collection
python scripts/data_collection.py

# 2. Check generated data
head -5 data/synthetic_lifestyle_data.csv

# 3. Start with basic analysis
python scripts/exploratory_analysis.py

# 4. Build initial model
python scripts/train_baseline_model.py
```

## 📈 Success Metrics
- **Data Coverage:** 500+ samples minimum
- **Feature Completeness:** 80%+ non-missing values
- **Class Balance:** No single class >70% of data
- **Model Performance:** >70% accuracy on validation set
