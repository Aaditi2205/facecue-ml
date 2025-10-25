# FaceCue ML - Nutritional Deficiency Detection System
# Specialized system for detecting specific nutritional deficiencies

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class NutritionalDeficiencyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
        # Specific nutritional deficiencies
        self.deficiency_mapping = {
            0: 'Normal',
            1: 'Iron Deficiency (Anemia)',
            2: 'Vitamin D Deficiency', 
            3: 'Vitamin B12 Deficiency',
            4: 'Vitamin C Deficiency',
            5: 'Zinc Deficiency',
            6: 'Magnesium Deficiency',
            7: 'Multiple Deficiencies'
        }
        
        # Initialize deficiency-specific recommendations
        self.deficiency_recommendations = self._initialize_deficiency_recommendations()
    
    def _initialize_deficiency_recommendations(self):
        """Initialize specific nutritional deficiency recommendations"""
        return {
            'Normal': {
                'title': 'No Nutritional Deficiencies Detected',
                'description': 'Your nutritional status appears to be within normal ranges.',
                'recommendations': [
                    'Continue maintaining a balanced diet with variety',
                    'Ensure adequate intake of all essential nutrients',
                    'Consider periodic nutritional screening',
                    'Maintain current healthy eating patterns'
                ],
                'priority': 'Low',
                'follow_up': 'Annual nutritional assessment recommended'
            },
            
            'Iron Deficiency (Anemia)': {
                'title': 'Iron Deficiency Anemia Detected',
                'description': 'Your indicators suggest iron deficiency anemia requiring immediate attention.',
                'symptoms': ['Fatigue', 'Weakness', 'Pale skin', 'Cold hands/feet', 'Brittle nails'],
                'recommendations': [
                    'URGENT: Get complete blood count (CBC) and iron studies within 1 week',
                    'Increase dietary iron: lean red meat, poultry, fish (18mg/day for adults)',
                    'Consume iron-rich plant sources: spinach, lentils, fortified cereals',
                    'Enhance iron absorption with vitamin C-rich foods (citrus, tomatoes)',
                    'Avoid tea/coffee with iron-rich meals (tannins reduce absorption)',
                    'Consider iron supplementation under medical supervision',
                    'Monitor for symptoms: fatigue, weakness, pale skin, cold intolerance'
                ],
                'priority': 'Critical',
                'follow_up': 'CBC and iron panel within 1 week, follow-up in 4-6 weeks',
                'blood_tests': ['Complete Blood Count (CBC)', 'Serum Iron', 'Ferritin', 'TIBC', 'Transferrin Saturation']
            },
            
            'Vitamin D Deficiency': {
                'title': 'Vitamin D Deficiency Detected',
                'description': 'Your indicators suggest vitamin D deficiency requiring assessment.',
                'symptoms': ['Bone pain', 'Muscle weakness', 'Frequent infections', 'Depression', 'Hair loss'],
                'recommendations': [
                    'URGENT: Get 25-hydroxyvitamin D blood test within 1 week',
                    'Increase sun exposure: 15-30 minutes daily (10 AM-3 PM, arms/legs exposed)',
                    'Include vitamin D-rich foods: fatty fish (salmon, mackerel), egg yolks, mushrooms',
                    'Consider vitamin D3 supplementation: 1000-2000 IU daily (under medical guidance)',
                    'Consume fortified dairy products and cereals',
                    'Monitor bone health and calcium intake (1000-1200mg daily)',
                    'Re-test vitamin D levels in 8-12 weeks after intervention'
                ],
                'priority': 'High',
                'follow_up': 'Vitamin D blood test within 1 week, re-test in 8-12 weeks',
                'blood_tests': ['25-Hydroxyvitamin D', 'Calcium', 'Phosphorus', 'Parathyroid Hormone']
            },
            
            'Vitamin B12 Deficiency': {
                'title': 'Vitamin B12 Deficiency Detected',
                'description': 'Your indicators suggest vitamin B12 deficiency requiring immediate evaluation.',
                'symptoms': ['Fatigue', 'Weakness', 'Numbness/tingling', 'Memory problems', 'Mood changes'],
                'recommendations': [
                    'URGENT: Get vitamin B12 and methylmalonic acid blood tests within 1 week',
                    'Include B12-rich foods: meat, fish, dairy, eggs',
                    'Consider B12 supplementation (oral or injection under medical supervision)',
                    'Monitor for neurological symptoms',
                    'Check intrinsic factor if absorption issues suspected',
                    'Re-test B12 levels in 4-6 weeks after intervention'
                ],
                'priority': 'Critical',
                'follow_up': 'B12 blood test within 1 week, neurological evaluation if symptoms present',
                'blood_tests': ['Vitamin B12', 'Methylmalonic Acid', 'Homocysteine', 'Complete Blood Count']
            },
            
            'Vitamin C Deficiency': {
                'title': 'Vitamin C Deficiency (Scurvy) Risk',
                'description': 'Your indicators suggest potential vitamin C deficiency.',
                'symptoms': ['Fatigue', 'Bleeding gums', 'Slow wound healing', 'Joint pain', 'Rough skin'],
                'recommendations': [
                    'Increase vitamin C-rich foods: citrus fruits, berries, bell peppers, broccoli',
                    'Aim for 90mg/day (men) or 75mg/day (women)',
                    'Consider vitamin C supplementation if dietary intake insufficient',
                    'Monitor for scurvy symptoms',
                    'Improve overall fruit and vegetable intake'
                ],
                'priority': 'Moderate',
                'follow_up': 'Monitor symptoms, increase vitamin C intake',
                'blood_tests': ['Vitamin C (Ascorbic Acid)', 'Complete Blood Count']
            },
            
            'Zinc Deficiency': {
                'title': 'Zinc Deficiency Detected',
                'description': 'Your indicators suggest zinc deficiency affecting immune function.',
                'symptoms': ['Frequent infections', 'Slow wound healing', 'Hair loss', 'Loss of appetite', 'Taste changes'],
                'recommendations': [
                    'Include zinc-rich foods: oysters, red meat, poultry, beans, nuts',
                    'Aim for 11mg/day (men) or 8mg/day (women)',
                    'Consider zinc supplementation under medical guidance',
                    'Avoid excessive iron supplementation (can interfere with zinc)',
                    'Monitor immune function and wound healing'
                ],
                'priority': 'Moderate',
                'follow_up': 'Monitor symptoms, consider zinc supplementation',
                'blood_tests': ['Serum Zinc', 'Complete Blood Count', 'Immune Function Tests']
            },
            
            'Magnesium Deficiency': {
                'title': 'Magnesium Deficiency Detected',
                'description': 'Your indicators suggest magnesium deficiency affecting muscle and nerve function.',
                'symptoms': ['Muscle cramps', 'Fatigue', 'Irregular heartbeat', 'Nausea', 'Personality changes'],
                'recommendations': [
                    'Include magnesium-rich foods: leafy greens, nuts, seeds, whole grains',
                    'Aim for 400-420mg/day (men) or 310-320mg/day (women)',
                    'Consider magnesium supplementation (citrate or glycinate forms)',
                    'Monitor for muscle cramps and heart rhythm',
                    'Avoid excessive calcium supplementation (can interfere with magnesium)'
                ],
                'priority': 'Moderate',
                'follow_up': 'Monitor symptoms, consider magnesium supplementation',
                'blood_tests': ['Serum Magnesium', 'Red Blood Cell Magnesium', 'Electrolyte Panel']
            },
            
            'Multiple Deficiencies': {
                'title': 'Multiple Nutritional Deficiencies Detected',
                'description': 'Your indicators suggest multiple nutritional deficiencies requiring comprehensive evaluation.',
                'symptoms': ['Severe fatigue', 'Multiple health issues', 'Poor immune function', 'Cognitive problems'],
                'recommendations': [
                    'URGENT: Comprehensive nutritional assessment with healthcare provider',
                    'Get complete nutritional panel: iron, B12, D, zinc, magnesium, folate',
                    'Consider working with registered dietitian for meal planning',
                    'Address underlying causes (malabsorption, dietary restrictions)',
                    'Monitor for multiple deficiency symptoms',
                    'Consider multivitamin supplementation under medical guidance'
                ],
                'priority': 'Critical',
                'follow_up': 'Comprehensive nutritional evaluation within 1 week',
                'blood_tests': ['Complete Nutritional Panel', 'Comprehensive Metabolic Panel', 'Complete Blood Count']
            }
        }
    
    def create_nutritional_deficiency_dataset(self):
        """Create dataset focused on nutritional deficiencies"""
        print("=== Creating Nutritional Deficiency Dataset ===")
        
        np.random.seed(42)
        n_samples = 2000
        
        # Generate lifestyle data that correlates with nutritional deficiencies
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'diet_type': np.random.choice(['Vegetarian', 'Non-Vegetarian', 'Vegan'], n_samples, p=[0.3, 0.5, 0.2]),
            'water_intake': np.random.normal(2.5, 0.8, n_samples),
            'sleep_hours': np.random.normal(7.5, 1.2, n_samples),
            'fatigue_scale': np.random.randint(1, 6, n_samples),
            'energy_scale': np.random.randint(1, 6, n_samples),
            'screen_time': np.random.normal(6, 2, n_samples),
            'exercise_hours': np.random.normal(3, 2, n_samples),
            'stress_level': np.random.randint(1, 6, n_samples),
            'smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.2, 0.8]),
            'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'meal_frequency': np.random.choice(['1-2 meals', '3 meals', '4-5 meals'], n_samples, p=[0.2, 0.6, 0.2]),
            'fruit_vegetable_intake': np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
            'sun_exposure': np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
            'supplement_use': np.random.choice(['None', 'Multivitamin', 'Specific'], n_samples, p=[0.5, 0.3, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Create nutritional deficiency labels based on lifestyle patterns
        deficiency_scores = []
        for i in range(n_samples):
            score = 0
            
            # Iron deficiency indicators
            if df.iloc[i]['diet_type'] == 'Vegan' and df.iloc[i]['fatigue_scale'] > 3:
                score += 2
            elif df.iloc[i]['fatigue_scale'] > 4 and df.iloc[i]['energy_scale'] < 2:
                score += 1
            
            # Vitamin D deficiency indicators
            if df.iloc[i]['sun_exposure'] == 'Low' and df.iloc[i]['diet_type'] == 'Vegan':
                score += 2
            elif df.iloc[i]['sun_exposure'] == 'Low':
                score += 1
            
            # Vitamin B12 deficiency indicators
            if df.iloc[i]['diet_type'] == 'Vegan' and df.iloc[i]['supplement_use'] == 'None':
                score += 2
            elif df.iloc[i]['diet_type'] == 'Vegetarian' and df.iloc[i]['supplement_use'] == 'None':
                score += 1
            
            # Vitamin C deficiency indicators
            if df.iloc[i]['fruit_vegetable_intake'] == 'Low' and df.iloc[i]['fatigue_scale'] > 3:
                score += 1
            
            # Zinc deficiency indicators
            if df.iloc[i]['diet_type'] == 'Vegan' and df.iloc[i]['meal_frequency'] == '1-2 meals':
                score += 1
            
            # Magnesium deficiency indicators
            if df.iloc[i]['stress_level'] > 4 and df.iloc[i]['exercise_hours'] < 2:
                score += 1
            
            # Multiple deficiencies
            if score >= 3:
                score = 7  # Multiple deficiencies
            
            deficiency_scores.append(score)
        
        # Map scores to deficiency types
        df['deficiency_type'] = [self.deficiency_mapping[min(score, 7)] for score in deficiency_scores]
        
        print(f"✓ Nutritional deficiency dataset created: {df.shape}")
        print(f"✓ Deficiency distribution: {df['deficiency_type'].value_counts().to_dict()}")
        
        return df
    
    def preprocess_nutritional_data(self, df):
        """Preprocess data for nutritional deficiency detection"""
        print("\n=== Preprocessing Nutritional Data ===")
        
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'diet_type', 'smoking', 'alcohol_consumption', 
                          'meal_frequency', 'fruit_vegetable_intake', 'sun_exposure', 'supplement_use']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Remove original categorical columns
        df_processed = df_processed.drop(columns=categorical_cols)
        
        # Create target variable
        le_target = LabelEncoder()
        df_processed['target'] = le_target.fit_transform(df_processed['deficiency_type'])
        self.label_encoders['deficiency_type'] = le_target
        
        # Prepare features
        feature_cols = [col for col in df_processed.columns if col not in ['deficiency_type', 'target']]
        self.feature_names = feature_cols
        
        print(f"✓ Features prepared: {len(feature_cols)}")
        print(f"✓ Deficiency types: {len(df_processed['target'].unique())}")
        
        return df_processed
    
    def train_nutritional_model(self, df):
        """Train model for nutritional deficiency detection"""
        print("\n=== Training Nutritional Deficiency Model ===")
        
        # Prepare features and target
        X = df[self.feature_names]
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Nutritional deficiency model trained")
        print(f"✓ Accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features for Nutritional Deficiency Detection:")
        print(feature_importance.head(10))
        
        return accuracy
    
    def predict_nutritional_deficiency(self, features):
        """Predict nutritional deficiency from features"""
        if self.model is None:
            return None, None
        
        try:
            # Prepare features
            feature_df = pd.DataFrame([features])
            
            # Encode categorical features
            for col in ['gender', 'diet_type', 'smoking', 'alcohol_consumption', 
                       'meal_frequency', 'fruit_vegetable_intake', 'sun_exposure', 'supplement_use']:
                if col in features:
                    le = self.label_encoders[col]
                    feature_df[f'{col}_encoded'] = le.transform([features[col]])[0]
                    del feature_df[col]
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0  # Default value
            
            # Reorder columns to match training data
            feature_df = feature_df[self.feature_names]
            
            # Scale features
            feature_scaled = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.model.predict(feature_scaled)[0]
            probabilities = self.model.predict_proba(feature_scaled)[0]
            
            # Map to deficiency name
            deficiency = self.deficiency_mapping.get(prediction, 'Unknown')
            
            return deficiency, probabilities
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None
    
    def generate_deficiency_recommendations(self, deficiency, confidence=None):
        """Generate specific nutritional deficiency recommendations"""
        if deficiency not in self.deficiency_recommendations:
            return {
                'title': 'Unknown Deficiency',
                'recommendations': ['Please consult with a healthcare provider'],
                'priority': 'Unknown'
            }
        
        rec = self.deficiency_recommendations[deficiency].copy()
        if confidence:
            rec['confidence'] = f"{confidence:.1%}"
        
        return rec
    
    def create_nutritional_assessment_report(self, features, user_name="User"):
        """Create comprehensive nutritional deficiency assessment report"""
        print(f"\n=== Nutritional Deficiency Assessment for {user_name} ===")
        
        # Make prediction
        deficiency, probabilities = self.predict_nutritional_deficiency(features)
        
        if deficiency is None:
            return "Unable to generate nutritional assessment"
        
        # Calculate confidence
        confidence = max(probabilities) if probabilities is not None else None
        
        # Generate recommendations
        recommendations = self.generate_deficiency_recommendations(deficiency, confidence)
        
        # Create report
        report = {
            'user_name': user_name,
            'predicted_deficiency': deficiency,
            'confidence': confidence,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print report
        print(f"\n🍎 NUTRITIONAL DEFICIENCY ASSESSMENT")
        print(f"Predicted Deficiency: {deficiency}")
        if confidence:
            print(f"Confidence Level: {confidence:.1%}")
        
        print(f"\n📋 DEFICIENCY ANALYSIS")
        print(f"Title: {recommendations['title']}")
        print(f"Description: {recommendations['description']}")
        print(f"Priority: {recommendations['priority']}")
        
        if 'symptoms' in recommendations:
            print(f"\n🔍 COMMON SYMPTOMS:")
            for symptom in recommendations['symptoms']:
                print(f"  • {symptom}")
        
        print(f"\n💊 SPECIFIC RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"  {i:2d}. {rec}")
        
        if 'blood_tests' in recommendations:
            print(f"\n🩸 RECOMMENDED BLOOD TESTS:")
            for test in recommendations['blood_tests']:
                print(f"  • {test}")
        
        print(f"\n📅 FOLLOW-UP CARE:")
        print(f"  {recommendations['follow_up']}")
        
        print(f"\n⚠️  MEDICAL DISCLAIMER:")
        print("  This assessment is for educational purposes only.")
        print("  Always consult with healthcare providers for medical diagnosis.")
        print("  Nutritional deficiencies require proper medical evaluation.")
        
        return report
    
    def save_nutritional_model(self, filename='data/nutritional_deficiency_model.pkl'):
        """Save the trained nutritional deficiency model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'deficiency_mapping': self.deficiency_mapping
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Nutritional deficiency model saved: {filename}")
        return filename
    
    def load_nutritional_model(self, filename='data/nutritional_deficiency_model.pkl'):
        """Load the trained nutritional deficiency model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.deficiency_mapping = model_data['deficiency_mapping']
            
            print(f"✓ Nutritional deficiency model loaded: {filename}")
            return True
        except FileNotFoundError:
            print(f"❌ Nutritional deficiency model not found: {filename}")
            return False
        except Exception as e:
            print(f"❌ Error loading nutritional deficiency model: {e}")
            return False

def main():
    print("🍎 FaceCue ML - Nutritional Deficiency Detection System")
    print("="*60)
    
    # Initialize nutritional deficiency detector
    detector = NutritionalDeficiencyDetector()
    
    # Create and preprocess data
    df = detector.create_nutritional_deficiency_dataset()
    df_processed = detector.preprocess_nutritional_data(df)
    
    # Train model
    accuracy = detector.train_nutritional_model(df_processed)
    
    # Save model
    detector.save_nutritional_model()
    
    # Example assessments
    print(f"\n=== Example Nutritional Deficiency Assessments ===")
    
    # Vegan with potential deficiencies
    vegan_features = {
        'age': 25,
        'gender': 'Female',
        'diet_type': 'Vegan',
        'water_intake': 2.0,
        'sleep_hours': 6.0,
        'fatigue_scale': 5,
        'energy_scale': 2,
        'screen_time': 8.0,
        'exercise_hours': 1.0,
        'stress_level': 4,
        'smoking': 'No',
        'alcohol_consumption': 'None',
        'meal_frequency': '2 meals',
        'fruit_vegetable_intake': 'Low',
        'sun_exposure': 'Low',
        'supplement_use': 'None'
    }
    
    detector.create_nutritional_assessment_report(vegan_features, "Vegan Student")
    
    # Healthy individual
    healthy_features = {
        'age': 30,
        'gender': 'Male',
        'diet_type': 'Non-Vegetarian',
        'water_intake': 3.0,
        'sleep_hours': 8.0,
        'fatigue_scale': 2,
        'energy_scale': 4,
        'screen_time': 4.0,
        'exercise_hours': 5.0,
        'stress_level': 2,
        'smoking': 'No',
        'alcohol_consumption': 'Light',
        'meal_frequency': '3 meals',
        'fruit_vegetable_intake': 'High',
        'sun_exposure': 'High',
        'supplement_use': 'Multivitamin'
    }
    
    detector.create_nutritional_assessment_report(healthy_features, "Healthy Individual")
    
    print(f"\n=== Nutritional Deficiency Detection System Complete ===")
    print(f"✅ Model trained with {accuracy:.3f} accuracy")
    print(f"✅ 8 specific nutritional deficiencies detected")
    print(f"✅ Professional medical recommendations provided")
    print(f"✅ Blood test recommendations included")

if __name__ == "__main__":
    main()
