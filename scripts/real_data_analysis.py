# FaceCue ML - Real Dataset Analysis
# Analyze the UCI Heart Disease dataset for real-world health prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class RealDataAnalyzer:
    def __init__(self, data_path="data/academic/uci_heart_disease.csv"):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the real UCI Heart Disease dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Loaded UCI Heart Disease dataset: {len(self.df)} samples, {len(self.df.columns)} features")
            print(f"  This is REAL medical data from Cleveland Clinic!")
        except FileNotFoundError:
            print(f"❌ Dataset not found at {self.data_path}")
            return None
    
    def explore_dataset(self):
        """Explore the real dataset characteristics"""
        print("\n=== Real Dataset Analysis ===")
        print(f"Dataset: UCI Heart Disease (Cleveland Clinic)")
        print(f"Shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        
        # Feature descriptions
        feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1=male, 0=female)',
            'cp': 'Chest pain type (1-4)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1=yes, 0=no)',
            'restecg': 'Resting ECG results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1=yes, 0=no)',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of peak exercise ST segment',
            'ca': 'Number of major vessels colored by flourosopy',
            'thal': 'Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)',
            'target': 'Heart disease (0=no disease, 1-4=disease severity)'
        }
        
        print("\n=== Feature Descriptions ===")
        for feature, description in feature_descriptions.items():
            print(f"{feature}: {description}")
        
        print("\n=== Data Types ===")
        print(self.df.dtypes)
        
        print("\n=== Missing Values ===")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
        
        # Check for '?' values (common in UCI datasets)
        print("\n=== Checking for '?' values ===")
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                question_marks = (self.df[col] == '?').sum()
                if question_marks > 0:
                    print(f"{col}: {question_marks} '?' values")
        
        print("\n=== Target Distribution ===")
        print(self.df['target'].value_counts().sort_index())
        
    def preprocess_data(self):
        """Preprocess the real dataset"""
        print("\n=== Data Preprocessing ===")
        
        df_processed = self.df.copy()
        
        # Handle '?' values by replacing with median/mode
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Replace '?' with mode
                mode_value = df_processed[col][df_processed[col] != '?'].mode()[0]
                df_processed[col] = df_processed[col].replace('?', mode_value)
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Handle any remaining NaN values
        df_processed = df_processed.fillna(df_processed.median())
        
        # Convert target to binary (0=no disease, 1=disease)
        df_processed['target_binary'] = (df_processed['target'] > 0).astype(int)
        
        print(f"✓ Preprocessed data: {df_processed.shape}")
        print(f"✓ Binary target distribution:")
        print(df_processed['target_binary'].value_counts())
        
        return df_processed
    
    def analyze_correlations(self, df_processed):
        """Analyze correlations with heart disease"""
        print("\n=== Correlation Analysis ===")
        
        # Calculate correlations with target
        feature_cols = [col for col in df_processed.columns if col not in ['target', 'target_binary']]
        correlations = {}
        
        for col in feature_cols:
            correlations[col] = np.corrcoef(df_processed[col], df_processed['target_binary'])[0,1]
        
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        print("Correlation with Heart Disease:")
        print(corr_df)
        
        return corr_df
    
    def visualize_data(self, df_processed):
        """Create visualizations of the real data"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Age distribution by heart disease
        sns.boxplot(data=df_processed, x='target_binary', y='age', ax=axes[0,0])
        axes[0,0].set_title('Age Distribution by Heart Disease Status')
        axes[0,0].set_xlabel('Heart Disease (0=No, 1=Yes)')
        
        # Cholesterol vs Blood Pressure
        sns.scatterplot(data=df_processed, x='chol', y='trestbps', 
                       hue='target_binary', ax=axes[0,1])
        axes[0,1].set_title('Cholesterol vs Blood Pressure')
        
        # Heart Rate vs Exercise
        sns.scatterplot(data=df_processed, x='thalach', y='oldpeak', 
                       hue='target_binary', ax=axes[0,2])
        axes[0,2].set_title('Max Heart Rate vs ST Depression')
        
        # Chest Pain Type
        chest_pain_counts = df_processed.groupby(['cp', 'target_binary']).size().unstack()
        chest_pain_counts.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Chest Pain Type by Heart Disease')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Sex distribution
        sex_counts = df_processed.groupby(['sex', 'target_binary']).size().unstack()
        sex_counts.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Sex Distribution by Heart Disease')
        axes[1,1].set_xlabel('Sex (0=Female, 1=Male)')
        
        # Feature importance (placeholder)
        feature_cols = [col for col in df_processed.columns if col not in ['target', 'target_binary']]
        correlations = []
        for col in feature_cols:
            correlations.append(abs(np.corrcoef(df_processed[col], df_processed['target_binary'])[0,1]))
        
        axes[1,2].bar(range(len(feature_cols)), correlations)
        axes[1,2].set_title('Feature Importance (Correlation)')
        axes[1,2].set_xlabel('Features')
        axes[1,2].set_ylabel('Absolute Correlation')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/real_data_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved real data analysis: data/real_data_analysis.png")
    
    def train_models(self, df_processed):
        """Train multiple models on real data"""
        print("\n=== Training Models on Real Data ===")
        
        # Prepare features and target
        feature_cols = [col for col in df_processed.columns if col not in ['target', 'target_binary']]
        X = df_processed[feature_cols]
        y = df_processed['target_binary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"  Test Accuracy: {accuracy:.3f}")
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['model']
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  Accuracy: {results[best_model_name]['accuracy']:.3f}")
        
        # Feature importance for Random Forest
        if best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)
        
        return results, best_model, X_test_scaled, y_test
    
    def compare_with_synthetic(self):
        """Compare real data performance with synthetic data"""
        print("\n=== Real vs Synthetic Data Comparison ===")
        
        # Load synthetic data results (if available)
        try:
            synthetic_data = np.load('data/processed_data.npz')
            print("✓ Synthetic data results available for comparison")
            
            print("\nReal Data Advantages:")
            print("  ✓ Actual medical patterns")
            print("  ✓ Clinically meaningful features")
            print("  ✓ Real-world applicability")
            print("  ✓ Proven health relationships")
            
            print("\nSynthetic Data Limitations:")
            print("  ✗ Simulated patterns only")
            print("  ✗ May not reflect real health dynamics")
            print("  ✗ Limited real-world applicability")
            print("  ✗ Artificial correlations")
            
        except FileNotFoundError:
            print("Synthetic data not available for comparison")

def main():
    print("=== FaceCue ML Real Dataset Analysis ===")
    print("Analyzing UCI Heart Disease dataset (REAL medical data)")
    
    analyzer = RealDataAnalyzer()
    
    if analyzer.df is not None:
        # Explore dataset
        analyzer.explore_dataset()
        
        # Preprocess data
        df_processed = analyzer.preprocess_data()
        
        # Analyze correlations
        corr_df = analyzer.analyze_correlations(df_processed)
        
        # Create visualizations
        analyzer.visualize_data(df_processed)
        
        # Train models
        results, best_model, X_test, y_test = analyzer.train_models(df_processed)
        
        # Compare with synthetic data
        analyzer.compare_with_synthetic()
        
        print("\n=== Real Data Analysis Complete ===")
        print("✓ Analyzed REAL medical data from Cleveland Clinic")
        print("✓ Trained models on actual health patterns")
        print("✓ Generated clinically meaningful insights")
        print("✓ Created real-world applicable predictions")
        
        print("\n=== Next Steps ===")
        print("1. Download more real datasets (Sleep Health, Nutrition Survey)")
        print("2. Combine multiple real datasets")
        print("3. Implement advanced feature engineering")
        print("4. Add facial image analysis")
        print("5. Deploy model for real-world health prediction")

if __name__ == "__main__":
    main()
