# FaceCue ML - Step 3: Data Preprocessing Pipeline
# Preprocesses tabular lifestyle data for health prediction

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.is_fitted = False
        
    def preprocess_lifestyle_data(self, df):
        """Preprocess lifestyle data for health prediction"""
        print("=== Step 3: Data Preprocessing ===")
        print(f"Input data shape: {df.shape}")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Step 1: Handle missing values
        print("\n1. Handling missing values...")
        missing_before = df_processed.isnull().sum().sum()
        print(f"Missing values before: {missing_before}")
        
        # Fill missing values
        df_processed = df_processed.fillna(df_processed.median())
        missing_after = df_processed.isnull().sum().sum()
        print(f"Missing values after: {missing_after}")
        
        # Step 2: Normalize numeric features
        print("\n2. Normalizing numeric features...")
        numeric_features = self._get_numeric_features(df_processed)
        print(f"Numeric features: {numeric_features}")
        
        if len(numeric_features) > 0:
            df_processed[numeric_features] = self.scaler.fit_transform(df_processed[numeric_features])
            print("✓ Numeric features normalized")
        
        # Step 3: Encode categorical features
        print("\n3. Encoding categorical features...")
        categorical_features = self._get_categorical_features(df_processed)
        print(f"Categorical features: {categorical_features}")
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                df_processed = self._encode_categorical_feature(df_processed, feature)
        
        # Step 4: Create target variable
        print("\n4. Creating target variable...")
        if 'health_status' in df_processed.columns:
            target_mapping = {
                'Normal': 0,
                'Anemia': 1,
                'Vitamin D Deficiency': 2,
                'Dehydration': 3,
                'Sleep Deficiency': 4,
                'Multiple Deficiencies': 5
            }
            df_processed['target'] = df_processed['health_status'].map(target_mapping)
            print("✓ Target variable created")
            print(f"Target distribution: {df_processed['target'].value_counts().to_dict()}")
        
        # Step 5: Prepare final features
        print("\n5. Preparing final features...")
        feature_cols = [col for col in df_processed.columns if col not in ['health_status', 'target']]
        self.feature_names = feature_cols
        
        print(f"Final features ({len(feature_cols)}): {feature_cols}")
        print(f"Processed data shape: {df_processed.shape}")
        
        self.is_fitted = True
        return df_processed
    
    def _get_numeric_features(self, df):
        """Identify numeric features for normalization"""
        numeric_features = []
        
        # Common numeric features in health data
        numeric_candidates = [
            'age', 'water_intake_liters', 'sleep_hours', 'fatigue_scale', 
            'energy_scale', 'screen_time_hours', 'exercise_hours_week', 
            'stress_level', 'blood_pressure', 'cholesterol', 'heart_rate',
            'bmi', 'weight', 'height'
        ]
        
        for feature in numeric_candidates:
            if feature in df.columns and df[feature].dtype in ['int64', 'float64']:
                numeric_features.append(feature)
        
        # Also check for any other numeric columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col not in numeric_features:
                numeric_features.append(col)
        
        return numeric_features
    
    def _get_categorical_features(self, df):
        """Identify categorical features for encoding"""
        categorical_features = []
        
        # Common categorical features in health data
        categorical_candidates = [
            'gender', 'diet_type', 'smoking', 'alcohol_consumption',
            'exercise_type', 'sleep_quality', 'stress_level_category'
        ]
        
        for feature in categorical_candidates:
            if feature in df.columns and df[feature].dtype == 'object':
                categorical_features.append(feature)
        
        # Also check for any other object columns
        for col in df.columns:
            if df[col].dtype == 'object' and col not in categorical_features:
                categorical_features.append(col)
        
        return categorical_features
    
    def _encode_categorical_feature(self, df, feature):
        """Encode a categorical feature"""
        if feature not in df.columns:
            return df
        
        # For binary features, use label encoding
        unique_values = df[feature].nunique()
        
        if unique_values <= 2:
            # Binary encoding
            le = LabelEncoder()
            df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
            self.label_encoders[feature] = le
            print(f"  ✓ {feature}: Binary encoded ({unique_values} values)")
        else:
            # One-hot encoding for multi-class
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = ohe.fit_transform(df[[feature]])
            
            # Create column names
            feature_names = [f'{feature}_{val}' for val in ohe.categories_[0]]
            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
            
            # Add encoded columns to dataframe
            df = pd.concat([df, encoded_df], axis=1)
            self.onehot_encoders[feature] = ohe
            print(f"  ✓ {feature}: One-hot encoded ({unique_values} values)")
        
        # Remove original categorical column
        df = df.drop(columns=[feature])
        return df
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\n=== Data Splitting ===")
        
        if 'target' not in df.columns:
            raise ValueError("Target column not found. Run preprocess_lifestyle_data() first.")
        
        # Separate features and target
        X = df[self.feature_names]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"Target distribution in train: {y_train.value_counts().to_dict()}")
        print(f"Target distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, df, filename='data/preprocessed_lifestyle_data.csv'):
        """Save preprocessed data"""
        df.to_csv(filename, index=False)
        print(f"✓ Preprocessed data saved: {filename}")
        return filename
    
    def get_feature_importance_info(self):
        """Get information about processed features"""
        if not self.is_fitted:
            return None
        
        info = {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'label_encoders': list(self.label_encoders.keys()),
            'onehot_encoders': list(self.onehot_encoders.keys())
        }
        
        return info

def main():
    print("=== FaceCue ML - Data Preprocessing Pipeline ===")
    
    # Load existing real datasets
    try:
        # Try to load the comprehensive real dataset
        df = pd.read_csv('data/academic/uci_heart_disease.csv')
        print(f"Loaded UCI Heart Disease dataset: {df.shape}")
        
        # Add some synthetic lifestyle features for demonstration
        np.random.seed(42)
        n_samples = len(df)
        
        # Add lifestyle features
        df['water_intake_liters'] = np.random.normal(2.5, 0.8, n_samples)
        df['sleep_hours'] = np.random.normal(7.5, 1.2, n_samples)
        df['fatigue_scale'] = np.random.randint(1, 6, n_samples)
        df['energy_scale'] = np.random.randint(1, 6, n_samples)
        df['screen_time_hours'] = np.random.normal(6, 2, n_samples)
        df['exercise_hours_week'] = np.random.normal(3, 2, n_samples)
        df['stress_level'] = np.random.randint(1, 6, n_samples)
        df['diet_type'] = np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan'], n_samples)
        df['smoking'] = np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8])
        df['alcohol_consumption'] = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Create health status based on existing target
        health_mapping = {0: 'Normal', 1: 'Mild Deficiency', 2: 'Moderate Deficiency', 3: 'Severe Deficiency', 4: 'Critical'}
        df['health_status'] = df['target'].map(health_mapping)
        
        print(f"Enhanced dataset shape: {df.shape}")
        
    except FileNotFoundError:
        print("No existing dataset found. Creating sample data...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(18, 65, n_samples),
            'water_intake_liters': np.random.normal(2.5, 0.8, n_samples),
            'sleep_hours': np.random.normal(7.5, 1.2, n_samples),
            'fatigue_scale': np.random.randint(1, 6, n_samples),
            'energy_scale': np.random.randint(1, 6, n_samples),
            'screen_time_hours': np.random.normal(6, 2, n_samples),
            'exercise_hours_week': np.random.normal(3, 2, n_samples),
            'stress_level': np.random.randint(1, 6, n_samples),
            'diet_type': np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan'], n_samples),
            'smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
            'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Create health status
        health_scores = []
        for i in range(n_samples):
            score = 0
            if df.loc[i, 'water_intake_liters'] < 1.5 and df.loc[i, 'fatigue_scale'] > 3:
                score += 2
            elif df.loc[i, 'sleep_hours'] < 6 and df.loc[i, 'energy_scale'] < 3:
                score += 2
            elif df.loc[i, 'fatigue_scale'] > 4:
                score += 1
            health_scores.append(score)
        
        health_mapping = {0: 'Normal', 1: 'Mild Deficiency', 2: 'Moderate Deficiency', 3: 'Severe Deficiency'}
        df['health_status'] = [health_mapping[min(score, 3)] for score in health_scores]
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    df_processed = preprocessor.preprocess_lifestyle_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(df_processed)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(df_processed)
    
    # Get feature info
    feature_info = preprocessor.get_feature_importance_info()
    print(f"\n=== Preprocessing Complete ===")
    print(f"✓ Data preprocessed successfully")
    print(f"✓ Features prepared: {feature_info['total_features']}")
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Ready for model training (Step 4)")

if __name__ == "__main__":
    main()
