# FaceCue ML - Step 4: Model Building Pipeline
# Builds and compares multiple models for health prediction

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, install if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

class ModelBuilder:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def build_tabular_models(self):
        """Build multiple models for tabular data"""
        print("=== Step 4: Building Models ===")
        
        # Define models to try
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            ),
            'SVM': SVC(
                random_state=42,
                kernel='rbf',
                probability=True
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100
            )
        
        self.models = models
        print(f"✓ Initialized {len(models)} models: {list(models.keys())}")
        return models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        print("\n=== Training and Evaluating Models ===")
        
        if not self.models:
            self.build_tabular_models()
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                f1_weighted = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✓ Accuracy: {accuracy:.3f}")
                print(f"  ✓ F1-Score (macro): {f1_macro:.3f}")
                print(f"  ✓ CV Accuracy: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model_name = max(valid_results.keys(), key=lambda x: valid_results[x]['accuracy'])
            self.best_model = valid_results[best_model_name]['model']
            self.best_model_name = best_model_name
            
            print(f"\n🏆 Best Model: {best_model_name}")
            print(f"   Accuracy: {valid_results[best_model_name]['accuracy']:.3f}")
            print(f"   F1-Score: {valid_results[best_model_name]['f1_macro']:.3f}")
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='Random Forest'):
        """Perform hyperparameter tuning for the best model"""
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            param_grid = {
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def create_model_comparison_visualization(self, y_test):
        """Create visualization comparing all models"""
        print("\n=== Creating Model Comparison Visualization ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model accuracy comparison
        model_names = []
        accuracies = []
        f1_scores = []
        
        for name, result in self.results.items():
            if 'error' not in result:
                model_names.append(name)
                accuracies.append(result['accuracy'])
                f1_scores.append(result['f1_macro'])
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        axes[0, 1].bar(model_names, f1_scores)
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score (Macro)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confusion matrix for best model
        if self.best_model_name and self.best_model_name in self.results:
            best_predictions = self.results[self.best_model_name]['predictions']
            cm = confusion_matrix(y_test, best_predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
            axes[1, 0].set_title(f'Confusion Matrix - {self.best_model_name}')
            axes[1, 0].set_xlabel('Predicted')
            axes[1, 0].set_ylabel('Actual')
        
        # Cross-validation scores
        cv_means = []
        cv_stds = []
        for name in model_names:
            if name in self.results:
                cv_means.append(self.results[name]['cv_mean'])
                cv_stds.append(self.results[name]['cv_std'])
        
        axes[1, 1].bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        axes[1, 1].set_title('Cross-Validation Scores')
        axes[1, 1].set_ylabel('CV Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison visualization saved: data/model_comparison.png")
    
    def generate_detailed_report(self, y_test):
        """Generate detailed classification report"""
        print("\n=== Detailed Model Reports ===")
        
        for name, result in self.results.items():
            if 'error' not in result:
                print(f"\n{name} - Classification Report:")
                print(classification_report(y_test, result['predictions']))
    
    def save_best_model(self, filename='data/best_model.pkl'):
        """Save the best model"""
        if self.best_model is None:
            print("No best model to save")
            return None
        
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"✓ Best model saved: {filename}")
        return filename
    
    def get_model_summary(self):
        """Get summary of all models"""
        if not self.results:
            return None
        
        summary = []
        for name, result in self.results.items():
            if 'error' not in result:
                summary.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.3f}",
                    'F1-Score': f"{result['f1_macro']:.3f}",
                    'CV Accuracy': f"{result['cv_mean']:.3f}",
                    'CV Std': f"{result['cv_std']:.3f}"
                })
        
        return pd.DataFrame(summary)

def main():
    print("=== FaceCue ML - Model Building Pipeline ===")
    
    # Load preprocessed data
    try:
        df = pd.read_csv('data/preprocessed_lifestyle_data.csv')
        print(f"Loaded preprocessed data: {df.shape}")
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing first...")
        import subprocess
        subprocess.run(['python', 'scripts/step3_preprocess_data.py'])
        df = pd.read_csv('data/preprocessed_lifestyle_data.csv')
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['health_status', 'target']]
    X = df[feature_cols]
    y = df['target']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize model builder
    builder = ModelBuilder()
    
    # Build models
    builder.build_tabular_models()
    
    # Train and evaluate models
    results = builder.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    builder.create_model_comparison_visualization(y_test)
    
    # Generate detailed reports
    builder.generate_detailed_report(y_test)
    
    # Save best model
    builder.save_best_model()
    
    # Print summary
    summary = builder.get_model_summary()
    if summary is not None:
        print("\n=== Model Performance Summary ===")
        print(summary.to_string(index=False))
    
    print(f"\n=== Model Building Complete ===")
    print(f"✓ {len(results)} models trained and evaluated")
    print(f"✓ Best model: {builder.best_model_name}")
    print(f"✓ Ready for recommendations (Step 6)")

if __name__ == "__main__":
    main()
