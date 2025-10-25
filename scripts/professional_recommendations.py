# FaceCue ML - Professional Medical Recommendations System
# Provides evidence-based medical recommendations for health conditions

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ProfessionalHealthRecommendations:
    def __init__(self):
        # Professional medical recommendations based on clinical guidelines
        self.recommendations_db = self._initialize_professional_recommendations()
        
    def _initialize_professional_recommendations(self):
        """Initialize professional medical recommendations database"""
        return {
            'Normal': {
                'title': 'Good Health Status',
                'description': 'Your lifestyle indicators suggest good overall health status.',
                'recommendations': [
                    'Maintain current healthy lifestyle patterns',
                    'Continue regular physical activity (150+ minutes moderate exercise weekly)',
                    'Sustain balanced diet with 5+ servings fruits/vegetables daily',
                    'Maintain sleep hygiene (7-9 hours nightly)',
                    'Continue adequate hydration (2.5-3L water daily)',
                    'Schedule annual preventive health screening'
                ],
                'priority': 'Low',
                'urgency': 'None',
                'follow_up': 'Annual comprehensive health assessment',
                'medical_reference': 'CDC Preventive Care Guidelines 2024'
            },
            
            'Anemia': {
                'title': 'Iron Deficiency Anemia Risk',
                'description': 'Your indicators suggest potential iron deficiency anemia requiring medical evaluation.',
                'recommendations': [
                    'Obtain complete blood count (CBC) with iron studies within 2 weeks',
                    'Increase dietary iron intake: lean red meat, poultry, fish (18mg/day for adults)',
                    'Consume iron-rich plant sources: spinach, lentils, fortified cereals',
                    'Enhance iron absorption with vitamin C-rich foods (citrus, tomatoes)',
                    'Avoid tea/coffee with iron-rich meals (tannins reduce absorption)',
                    'Consider iron supplementation only under medical supervision',
                    'Monitor for symptoms: fatigue, weakness, pale skin, cold intolerance'
                ],
                'priority': 'High',
                'urgency': 'Moderate',
                'follow_up': 'CBC and iron panel within 2 weeks, follow-up in 4-6 weeks',
                'medical_reference': 'WHO Iron Deficiency Anemia Guidelines 2023'
            },
            
            'Vitamin D Deficiency': {
                'title': 'Vitamin D Deficiency Risk',
                'description': 'Your indicators suggest potential vitamin D deficiency requiring assessment.',
                'recommendations': [
                    'Obtain 25-hydroxyvitamin D blood test within 2 weeks',
                    'Increase sun exposure: 15-30 minutes daily (10 AM-3 PM, arms/legs exposed)',
                    'Include vitamin D-rich foods: fatty fish (salmon, mackerel), egg yolks, mushrooms',
                    'Consider vitamin D3 supplementation: 1000-2000 IU daily (under medical guidance)',
                    'Consume fortified dairy products and cereals',
                    'Monitor bone health and calcium intake (1000-1200mg daily)',
                    'Re-test vitamin D levels in 8-12 weeks after intervention'
                ],
                'priority': 'High',
                'urgency': 'Moderate',
                'follow_up': 'Vitamin D blood test within 2 weeks, re-test in 8-12 weeks',
                'medical_reference': 'Endocrine Society Vitamin D Guidelines 2024'
            },
            
            'Dehydration': {
                'title': 'Dehydration Risk',
                'description': 'Your indicators suggest dehydration risk requiring immediate attention.',
                'recommendations': [
                    'Increase fluid intake to 2.5-3L daily (more in hot weather/exercise)',
                    'Monitor urine color: should be pale yellow (dark indicates dehydration)',
                    'Include water-rich foods: watermelon, cucumber, oranges, celery',
                    'Limit caffeine and alcohol (diuretic effects)',
                    'Consider electrolyte replacement during intense exercise/heat exposure',
                    'Monitor for symptoms: dry mouth, dizziness, fatigue, decreased urine output',
                    'Seek immediate medical attention if severe dehydration symptoms occur'
                ],
                'priority': 'High',
                'urgency': 'High',
                'follow_up': 'Monitor hydration status daily, medical evaluation if symptoms persist',
                'medical_reference': 'American College of Sports Medicine Hydration Guidelines 2024'
            },
            
            'Sleep Deficiency': {
                'title': 'Sleep Quality/Quantity Issues',
                'description': 'Your indicators suggest sleep disturbances affecting health and well-being.',
                'recommendations': [
                    'Maintain consistent sleep schedule (same bedtime/wake time daily)',
                    'Implement sleep hygiene: cool room (65-68°F), dark, quiet environment',
                    'Avoid screens 1 hour before bedtime (blue light disrupts melatonin)',
                    'Limit caffeine after 2 PM (6-hour half-life)',
                    'Establish relaxing bedtime routine: reading, meditation, warm bath',
                    'Exercise regularly but avoid vigorous activity 3 hours before bed',
                    'Consider cognitive behavioral therapy for insomnia (CBT-I) if persistent',
                    'Evaluate for sleep apnea if snoring/excessive daytime sleepiness present'
                ],
                'priority': 'High',
                'urgency': 'Moderate',
                'follow_up': 'Sleep diary for 2 weeks, consider sleep study if symptoms persist',
                'medical_reference': 'American Academy of Sleep Medicine Guidelines 2024'
            },
            
            'Multiple Deficiencies': {
                'title': 'Multiple Health Risk Factors',
                'description': 'Your indicators suggest multiple potential health concerns requiring comprehensive evaluation.',
                'recommendations': [
                    'Schedule comprehensive health assessment with primary care physician',
                    'Obtain complete metabolic panel, CBC, lipid panel, vitamin levels',
                    'Implement comprehensive lifestyle modifications: diet, exercise, sleep, stress management',
                    'Consider referral to registered dietitian for personalized nutrition plan',
                    'Evaluate mental health and stress management strategies',
                    'Consider multidisciplinary approach: physician, dietitian, exercise physiologist',
                    'Monitor progress with regular follow-up appointments',
                    'Address highest priority risk factors first (dehydration, sleep, nutrition)'
                ],
                'priority': 'Critical',
                'urgency': 'High',
                'follow_up': 'Comprehensive medical evaluation within 1 week',
                'medical_reference': 'US Preventive Services Task Force Guidelines 2024'
            }
        }
    
    def get_professional_recommendations(self, condition, confidence=None, user_profile=None):
        """Get professional medical recommendations for a condition"""
        if condition not in self.recommendations_db:
            return self._get_default_recommendations()
        
        recommendations = self.recommendations_db[condition].copy()
        
        # Add confidence information
        if confidence:
            recommendations['confidence'] = f"{confidence:.1%}"
            
            # Adjust urgency based on confidence
            if confidence > 0.8 and recommendations['urgency'] != 'None':
                recommendations['urgency'] = 'High'
            elif confidence < 0.5:
                recommendations['urgency'] = 'Low'
                recommendations['follow_up'] = 'Consider lifestyle improvements and monitor symptoms'
        
        # Add personalized elements if user profile provided
        if user_profile:
            recommendations = self._personalize_recommendations(recommendations, user_profile)
        
        return recommendations
    
    def _get_default_recommendations(self):
        """Default recommendations for unknown conditions"""
        return {
            'title': 'Health Assessment Required',
            'description': 'Unable to determine specific health condition from provided data.',
            'recommendations': [
                'Schedule comprehensive health evaluation with healthcare provider',
                'Maintain healthy lifestyle: balanced diet, regular exercise, adequate sleep',
                'Monitor symptoms and seek medical attention if concerns arise',
                'Consider preventive health screening appropriate for age/gender'
            ],
            'priority': 'Unknown',
            'urgency': 'Low',
            'follow_up': 'Healthcare provider consultation recommended',
            'medical_reference': 'General Medical Practice Guidelines'
        }
    
    def _personalize_recommendations(self, recommendations, user_profile):
        """Personalize recommendations based on user profile"""
        personalized = recommendations.copy()
        
        # Age-specific modifications
        age = user_profile.get('age', 30)
        if age > 65:
            personalized['recommendations'].append('Consider age-appropriate health screenings')
        elif age < 30:
            personalized['recommendations'].append('Focus on preventive health measures')
        
        # Gender-specific modifications
        gender = user_profile.get('gender', 'Unknown')
        if gender.lower() == 'female':
            personalized['recommendations'].append('Consider iron needs related to menstrual cycle')
        elif gender.lower() == 'male':
            personalized['recommendations'].append('Monitor cardiovascular risk factors')
        
        return personalized
    
    def create_professional_report(self, condition, confidence, user_profile=None):
        """Create a professional medical report"""
        recommendations = self.get_professional_recommendations(condition, confidence, user_profile)
        
        report = {
            'condition': condition,
            'confidence': confidence,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'disclaimer': self._get_medical_disclaimer()
        }
        
        return report
    
    def _get_medical_disclaimer(self):
        """Get professional medical disclaimer"""
        return {
            'title': 'Medical Disclaimer',
            'content': [
                'This assessment is for educational and informational purposes only.',
                'It is not intended as medical advice, diagnosis, or treatment.',
                'Always consult with qualified healthcare professionals for medical concerns.',
                'Seek immediate medical attention for emergency situations.',
                'Individual health needs may vary and require personalized medical care.'
            ]
        }
    
    def format_professional_output(self, report):
        """Format professional medical output"""
        recommendations = report['recommendations']
        
        output = f"""
🏥 PROFESSIONAL HEALTH ASSESSMENT
{'='*50}

CONDITION: {recommendations['title']}
Confidence Level: {report['confidence']:.1%}
Priority Level: {recommendations['priority']}
Urgency Level: {recommendations['urgency']}

DESCRIPTION:
{recommendations['description']}

PROFESSIONAL RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(recommendations['recommendations'], 1):
            output += f"{i:2d}. {rec}\n"
        
        output += f"""
FOLLOW-UP CARE:
{recommendations['follow_up']}

MEDICAL REFERENCE:
{recommendations['medical_reference']}

⚠️  PROFESSIONAL MEDICAL DISCLAIMER:
"""
        
        for disclaimer in report['disclaimer']['content']:
            output += f"• {disclaimer}\n"
        
        return output

def main():
    """Test the professional recommendations system"""
    print("🏥 Professional Health Recommendations System")
    print("="*50)
    
    recommender = ProfessionalHealthRecommendations()
    
    # Test different conditions
    test_cases = [
        ('Anemia', 0.85),
        ('Vitamin D Deficiency', 0.78),
        ('Dehydration', 0.92),
        ('Sleep Deficiency', 0.67),
        ('Normal', 0.45)
    ]
    
    for condition, confidence in test_cases:
        print(f"\n--- Testing {condition} (Confidence: {confidence:.1%}) ---")
        
        user_profile = {'age': 35, 'gender': 'Female'}
        report = recommender.create_professional_report(condition, confidence, user_profile)
        
        formatted_output = recommender.format_professional_output(report)
        print(formatted_output)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
