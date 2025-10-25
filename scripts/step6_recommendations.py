# FaceCue ML - Step 6: Health Recommendations System
# Generates personalized health recommendations based on predictions

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HealthRecommendationSystem:
    def __init__(self, model_path='data/best_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()
        
        # Health condition mappings
        self.condition_mapping = {
            0: 'Normal',
            1: 'Anemia',
            2: 'Vitamin D Deficiency', 
            3: 'Dehydration',
            4: 'Sleep Deficiency',
            5: 'Multiple Deficiencies'
        }
        
        # Initialize recommendations database
        self.recommendations_db = self._initialize_recommendations()
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"❌ Model not found at {self.model_path}")
            print("Please run Step 4 (Model Building) first")
    
    def _initialize_recommendations(self):
        """Initialize comprehensive recommendations database"""
        recommendations = {
            'Normal': {
                'title': 'Good Health Status',
                'description': 'Your lifestyle indicators suggest good overall health.',
                'recommendations': [
                    'Continue maintaining your current healthy lifestyle',
                    'Keep up with regular exercise and balanced diet',
                    'Maintain adequate sleep (7-9 hours)',
                    'Stay hydrated (2-3 liters of water daily)',
                    'Schedule regular health checkups'
                ],
                'priority': 'Low',
                'urgency': 'None',
                'follow_up': 'Annual health checkup recommended'
            },
            
            'Anemia': {
                'title': 'Potential Iron Deficiency (Anemia)',
                'description': 'Your indicators suggest possible iron deficiency anemia.',
                'recommendations': [
                    'Increase iron-rich foods: spinach, red meat, beans, lentils',
                    'Consume vitamin C with iron-rich foods to enhance absorption',
                    'Consider iron supplements (consult doctor first)',
                    'Get a complete blood count (CBC) test',
                    'Avoid tea/coffee with iron-rich meals (reduces absorption)',
                    'Include fortified cereals and bread in your diet'
                ],
                'priority': 'High',
                'urgency': 'Moderate',
                'follow_up': 'Blood test within 2-4 weeks recommended'
            },
            
            'Vitamin D Deficiency': {
                'title': 'Potential Vitamin D Deficiency',
                'description': 'Your indicators suggest possible vitamin D deficiency.',
                'recommendations': [
                    'Increase sun exposure: 15-30 minutes daily (with sunscreen)',
                    'Include vitamin D-rich foods: fatty fish, egg yolks, mushrooms',
                    'Consider vitamin D supplements (consult doctor for dosage)',
                    'Get vitamin D blood test (25-hydroxyvitamin D)',
                    'Spend time outdoors during peak sun hours (10 AM - 3 PM)',
                    'Include fortified dairy products and cereals'
                ],
                'priority': 'High',
                'urgency': 'Moderate',
                'follow_up': 'Vitamin D blood test within 2-4 weeks recommended'
            },
            
            'Dehydration': {
                'title': 'Potential Dehydration',
                'description': 'Your indicators suggest possible dehydration.',
                'recommendations': [
                    'Increase daily water intake to 2.5-3 liters',
                    'Drink water throughout the day, not just when thirsty',
                    'Include water-rich foods: watermelon, cucumber, oranges',
                    'Monitor urine color (should be light yellow)',
                    'Reduce caffeine and alcohol intake',
                    'Consider electrolyte drinks during hot weather or exercise'
                ],
                'priority': 'High',
                'urgency': 'High',
                'follow_up': 'Monitor hydration levels daily'
            },
            
            'Sleep Deficiency': {
                'title': 'Sleep Quality Issues',
                'description': 'Your indicators suggest sleep quality or quantity issues.',
                'recommendations': [
                    'Maintain consistent sleep schedule (same bedtime/wake time)',
                    'Create bedtime routine: no screens 1 hour before bed',
                    'Keep bedroom cool, dark, and quiet',
                    'Avoid caffeine after 2 PM',
                    'Limit screen time before bed',
                    'Consider relaxation techniques: meditation, deep breathing',
                    'Exercise regularly but not close to bedtime'
                ],
                'priority': 'High',
                'urgency': 'Moderate',
                'follow_up': 'Track sleep patterns for 2 weeks'
            },
            
            'Multiple Deficiencies': {
                'title': 'Multiple Health Concerns',
                'description': 'Your indicators suggest multiple potential health issues.',
                'recommendations': [
                    'Schedule comprehensive health checkup',
                    'Get complete blood panel including iron, vitamin D, B12',
                    'Consult with healthcare provider for personalized plan',
                    'Focus on balanced nutrition with variety of foods',
                    'Improve sleep hygiene and stress management',
                    'Increase physical activity gradually',
                    'Consider working with a nutritionist'
                ],
                'priority': 'Critical',
                'urgency': 'High',
                'follow_up': 'Healthcare provider consultation within 1 week'
            }
        }
        
        return recommendations
    
    def predict_health_status(self, features):
        """Predict health status from features"""
        if self.model is None:
            return None, None
        
        try:
            # Ensure features are in correct format
            if isinstance(features, dict):
                feature_df = pd.DataFrame([features])
            else:
                feature_df = features
            
            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            probabilities = self.model.predict_proba(feature_df)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Map prediction to condition name
            condition = self.condition_mapping.get(prediction, 'Unknown')
            
            return condition, probabilities
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None
    
    def generate_recommendations(self, condition, confidence=None):
        """Generate personalized recommendations for a health condition"""
        if condition not in self.recommendations_db:
            return {
                'title': 'Unknown Condition',
                'description': 'Unable to generate recommendations for this condition.',
                'recommendations': ['Please consult with a healthcare provider'],
                'priority': 'Unknown',
                'urgency': 'Unknown',
                'follow_up': 'Healthcare consultation recommended'
            }
        
        recommendations = self.recommendations_db[condition].copy()
        
        # Add confidence information if available
        if confidence is not None:
            recommendations['confidence'] = f"{confidence:.1%}"
            
            # Adjust urgency based on confidence
            if confidence > 0.8:
                recommendations['urgency'] = 'High' if recommendations['urgency'] == 'High' else 'Moderate'
            elif confidence < 0.5:
                recommendations['urgency'] = 'Low'
                recommendations['follow_up'] = 'Consider lifestyle improvements and monitor symptoms'
        
        return recommendations
    
    def create_personalized_report(self, features, user_name="User"):
        """Create a comprehensive personalized health report"""
        print(f"\n=== Personalized Health Report for {user_name} ===")
        
        # Make prediction
        condition, probabilities = self.predict_health_status(features)
        
        if condition is None:
            return "Unable to generate health report. Please check your model."
        
        # Calculate confidence
        confidence = max(probabilities) if probabilities is not None else None
        
        # Generate recommendations
        recommendations = self.generate_recommendations(condition, confidence)
        
        # Create report
        report = {
            'user_name': user_name,
            'predicted_condition': condition,
            'confidence': confidence,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Print report
        print(f"\n🏥 HEALTH ASSESSMENT")
        print(f"Predicted Condition: {condition}")
        if confidence:
            print(f"Confidence Level: {confidence:.1%}")
        
        print(f"\n📋 RECOMMENDATIONS")
        print(f"Title: {recommendations['title']}")
        print(f"Description: {recommendations['description']}")
        print(f"Priority: {recommendations['priority']}")
        print(f"Urgency: {recommendations['urgency']}")
        
        print(f"\n💡 ACTION ITEMS:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📅 FOLLOW-UP:")
        print(f"  {recommendations['follow_up']}")
        
        print(f"\n⚠️  IMPORTANT DISCLAIMER:")
        print("  This is not a medical diagnosis. Please consult with a healthcare")
        print("  provider for proper medical advice and treatment.")
        
        return report
    
    def save_report(self, report, filename=None):
        """Save health report to file"""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/health_report_{timestamp}.json"
        
        import json
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✓ Health report saved: {filename}")
        return filename
    
    def create_batch_recommendations(self, features_list, user_names=None):
        """Create recommendations for multiple users"""
        if user_names is None:
            user_names = [f"User_{i+1}" for i in range(len(features_list))]
        
        reports = []
        for i, (features, name) in enumerate(zip(features_list, user_names)):
            print(f"\n--- Processing {name} ---")
            report = self.create_personalized_report(features, name)
            reports.append(report)
        
        return reports

def main():
    print("=== FaceCue ML - Health Recommendations System ===")
    
    # Initialize recommendation system
    recommender = HealthRecommendationSystem()
    
    if recommender.model is None:
        print("Please run Step 4 (Model Building) first to train a model")
        return
    
    # Example usage with sample data
    print("\n=== Example Health Recommendations ===")
    
    # Sample feature sets for different scenarios
    sample_features = [
        {
            'name': 'Healthy Person',
            'features': {
                'age': 0.0,  # Normalized
                'water_intake_liters': 0.5,
                'sleep_hours': 0.8,
                'fatigue_scale': -0.5,
                'energy_scale': 0.7,
                'screen_time_hours': -0.3,
                'exercise_hours_week': 0.4,
                'stress_level': -0.2
            }
        },
        {
            'name': 'Potential Anemia',
            'features': {
                'age': 0.2,
                'water_intake_liters': -0.8,
                'sleep_hours': -0.5,
                'fatigue_scale': 1.2,
                'energy_scale': -1.0,
                'screen_time_hours': 0.5,
                'exercise_hours_week': -0.6,
                'stress_level': 0.8
            }
        },
        {
            'name': 'Dehydration Risk',
            'features': {
                'age': 0.1,
                'water_intake_liters': -1.5,
                'sleep_hours': 0.2,
                'fatigue_scale': 0.8,
                'energy_scale': -0.3,
                'screen_time_hours': 1.0,
                'exercise_hours_week': 0.1,
                'stress_level': 0.5
            }
        }
    ]
    
    # Generate recommendations for each sample
    reports = []
    for sample in sample_features:
        report = recommender.create_personalized_report(
            sample['features'], 
            sample['name']
        )
        reports.append(report)
        
        # Save individual report
        filename = f"data/report_{sample['name'].replace(' ', '_').lower()}.json"
        recommender.save_report(report, filename)
    
    print(f"\n=== Recommendations System Ready ===")
    print(f"✓ Generated {len(reports)} sample health reports")
    print(f"✓ Recommendations database initialized")
    print(f"✓ Ready for UI integration (Step 7)")

if __name__ == "__main__":
    main()
