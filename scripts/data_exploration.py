# FaceCue ML - Data Exploration & Preprocessing
# This script explores the collected data and prepares it for modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DataExplorer:
    def __init__(self, data_path="data/synthetic_lifestyle_data.csv"):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded dataset with {len(self.df)} samples and {len(self.df.columns)} features")
        except FileNotFoundError:
            print(f"❌ Dataset not found at {self.data_path}")
            return None
    
    def basic_info(self):
        """Display basic dataset information"""
        print("\n=== Dataset Overview ===")
        print(f"Shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        print(f"Target variable: health_status")
        
        print("\n=== Data Types ===")
        print(self.df.dtypes)
        
        print("\n=== Missing Values ===")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
        
        print("\n=== Target Distribution ===")
        print(self.df['health_status'].value_counts())
        
    def numerical_analysis(self):
        """Analyze numerical features"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"\n=== Numerical Features Analysis ===")
        print(f"Numerical columns: {list(numerical_cols)}")
        
        print("\nDescriptive Statistics:")
        print(self.df[numerical_cols].describe())
        
        # Correlation analysis
        print("\n=== Correlation with Target ===")
        target_encoded = LabelEncoder().fit_transform(self.df['health_status'])
        correlations = {}
        for col in numerical_cols:
            correlations[col] = np.corrcoef(self.df[col], target_encoded)[0,1]
        
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        print(corr_df)
        
    def categorical_analysis(self):
        """Analyze categorical features"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'health_status']
        
        print(f"\n=== Categorical Features Analysis ===")
        print(f"Categorical columns: {list(categorical_cols)}")
        
        for col in categorical_cols:
            print(f"\n{col} distribution:")
            print(self.df[col].value_counts())
            
    def visualize_data(self):
        """Create visualizations"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        self.df['health_status'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Health Status Distribution')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Age distribution by health status
        sns.boxplot(data=self.df, x='health_status', y='age', ax=axes[0,1])
        axes[0,1].set_title('Age Distribution by Health Status')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Water intake vs fatigue
        sns.scatterplot(data=self.df, x='water_intake_liters', y='fatigue_scale', 
                       hue='health_status', ax=axes[1,0])
        axes[1,0].set_title('Water Intake vs Fatigue Scale')
        
        # Sleep hours vs energy
        sns.scatterplot(data=self.df, x='sleep_hours', y='energy_scale', 
                       hue='health_status', ax=axes[1,1])
        axes[1,1].set_title('Sleep Hours vs Energy Scale')
        
        plt.tight_layout()
        plt.savefig('data/data_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: data/data_visualization.png")
        
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        print("\n=== Data Preprocessing ===")
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'diet_type', 'smoking', 'alcohol_consumption']
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
            
        # Encode target variable
        target_encoder = LabelEncoder()
        df_processed['health_status_encoded'] = target_encoder.fit_transform(df_processed['health_status'])
        
        # Separate features and target
        feature_cols = [col for col in df_processed.columns if col not in ['health_status', 'health_status_encoded']]
        X = df_processed[feature_cols]
        y = df_processed['health_status_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✓ Preprocessed data: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        print(f"✓ Features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, target_encoder, scaler, label_encoders
        
    def train_baseline_model(self, X_train, X_test, y_train, y_test, target_encoder):
        """Train a baseline Random Forest model"""
        print("\n=== Training Baseline Model ===")
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Evaluate model
        print("\nClassification Report:")
        target_names = target_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Feature importance
        feature_names = [col for col in self.df.columns if col != 'health_status']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return rf_model, feature_importance

def main():
    print("=== FaceCue ML Data Exploration ===")
    
    # Initialize data explorer
    explorer = DataExplorer()
    
    if explorer.df is not None:
        # Basic analysis
        explorer.basic_info()
        explorer.numerical_analysis()
        explorer.categorical_analysis()
        
        # Visualizations
        explorer.visualize_data()
        
        # Preprocessing
        X_train, X_test, y_train, y_test, target_encoder, scaler, label_encoders = explorer.preprocess_data()
        
        # Train baseline model
        model, feature_importance = explorer.train_baseline_model(X_train, X_test, y_train, y_test, target_encoder)
        
        print("\n=== Next Steps ===")
        print("1. ✓ Data exploration completed")
        print("2. ✓ Baseline model trained")
        print("3. → Download real datasets from Kaggle")
        print("4. → Improve feature engineering")
        print("5. → Try different algorithms (SVM, XGBoost, Neural Networks)")
        print("6. → Add facial image analysis")
        
        # Save processed data
        processed_data = {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [col for col in explorer.df.columns if col != 'health_status'],
            'target_names': target_encoder.classes_
        }
        
        np.savez('data/processed_data.npz', **processed_data)
        print("✓ Saved processed data: data/processed_data.npz")

if __name__ == "__main__":
    main()
