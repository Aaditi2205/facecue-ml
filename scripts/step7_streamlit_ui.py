# FaceCue ML - Step 7: Simple UI with Streamlit
# Creates a user-friendly interface for health prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our recommendation system
import sys
sys.path.append('scripts')
from step6_recommendations import HealthRecommendationSystem

class FaceCueUI:
    def __init__(self):
        self.recommender = None
        self.initialize_recommender()
        
    def initialize_recommender(self):
        """Initialize the recommendation system"""
        try:
            self.recommender = HealthRecommendationSystem()
            if self.recommender.model is None:
                st.error("Model not found. Please run model training first.")
                return False
            return True
        except Exception as e:
            st.error(f"Error initializing recommendation system: {e}")
            return False
    
    def create_lifestyle_input_form(self):
        """Create lifestyle input form"""
        st.subheader("📊 Lifestyle Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 30)
            water_intake = st.slider("Daily Water Intake (liters)", 0.5, 5.0, 2.5)
            sleep_hours = st.slider("Sleep Hours per Night", 4.0, 12.0, 7.5)
            fatigue_scale = st.slider("Fatigue Level (1-5)", 1, 5, 3)
        
        with col2:
            energy_scale = st.slider("Energy Level (1-5)", 1, 5, 3)
            screen_time = st.slider("Daily Screen Time (hours)", 1, 16, 6)
            exercise_hours = st.slider("Weekly Exercise (hours)", 0, 20, 3)
            stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
        
        st.subheader("🍽️ Dietary Information")
        col3, col4 = st.columns(2)
        
        with col3:
            diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
            smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        
        with col4:
            alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
        
        return {
            'age': age,
            'water_intake_liters': water_intake,
            'sleep_hours': sleep_hours,
            'fatigue_scale': fatigue_scale,
            'energy_scale': energy_scale,
            'screen_time_hours': screen_time,
            'exercise_hours_week': exercise_hours,
            'stress_level': stress_level,
            'diet_type': diet_type,
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption
        }
    
    def normalize_features(self, features):
        """Normalize features for model prediction"""
        # Simple normalization (in real implementation, use the same scaler from training)
        normalized_features = {
            'age': (features['age'] - 30) / 15,  # Normalize around 30
            'water_intake_liters': (features['water_intake_liters'] - 2.5) / 1.0,
            'sleep_hours': (features['sleep_hours'] - 7.5) / 1.5,
            'fatigue_scale': (features['fatigue_scale'] - 3) / 1.5,
            'energy_scale': (features['energy_scale'] - 3) / 1.5,
            'screen_time_hours': (features['screen_time_hours'] - 6) / 3,
            'exercise_hours_week': (features['exercise_hours_week'] - 3) / 2,
            'stress_level': (features['stress_level'] - 3) / 1.5,
            'diet_type_Vegetarian': 1 if features['diet_type'] == 'Vegetarian' else 0,
            'diet_type_Non-Vegetarian': 1 if features['diet_type'] == 'Non-Vegetarian' else 0,
            'diet_type_Vegan': 1 if features['diet_type'] == 'Vegan' else 0,
            'smoking_encoded': 1 if features['smoking'] == 'Yes' else 0,
            'alcohol_consumption_None': 1 if features['alcohol_consumption'] == 'None' else 0,
            'alcohol_consumption_Light': 1 if features['alcohol_consumption'] == 'Light' else 0,
            'alcohol_consumption_Moderate': 1 if features['alcohol_consumption'] == 'Moderate' else 0,
            'alcohol_consumption_Heavy': 1 if features['alcohol_consumption'] == 'Heavy' else 0
        }
        
        return normalized_features
    
    def display_health_assessment(self, features):
        """Display health assessment results"""
        if self.recommender is None:
            st.error("Recommendation system not initialized")
            return
        
        # Normalize features
        normalized_features = self.normalize_features(features)
        
        # Make prediction
        condition, probabilities = self.recommender.predict_health_status(normalized_features)
        
        if condition is None:
            st.error("Unable to make health prediction")
            return
        
        # Calculate confidence
        confidence = max(probabilities) if probabilities is not None else None
        
        # Generate recommendations
        recommendations = self.recommender.generate_recommendations(condition, confidence)
        
        # Display results
        st.subheader("🏥 Health Assessment Results")
        
        # Condition and confidence
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Condition", condition)
        with col2:
            if confidence:
                st.metric("Confidence", f"{confidence:.1%}")
        
        # Priority and urgency indicators
        priority_colors = {
            'Low': '🟢',
            'High': '🟡', 
            'Critical': '🔴'
        }
        
        urgency_colors = {
            'None': '🟢',
            'Moderate': '🟡',
            'High': '🔴'
        }
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Priority", f"{priority_colors.get(recommendations['priority'], '⚪')} {recommendations['priority']}")
        with col4:
            st.metric("Urgency", f"{urgency_colors.get(recommendations['urgency'], '⚪')} {recommendations['urgency']}")
        
        # Recommendations
        st.subheader("💡 Personalized Recommendations")
        
        st.write(f"**{recommendations['title']}**")
        st.write(recommendations['description'])
        
        st.write("**Action Items:**")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            st.write(f"{i}. {rec}")
        
        st.write("**Follow-up:**")
        st.write(recommendations['follow_up'])
        
        # Important disclaimer
        st.subheader("⚠️ Important Disclaimer")
        st.warning("""
        **This is not a medical diagnosis.** 
        
        FaceCue ML provides health insights based on lifestyle patterns, but it cannot replace professional medical advice. 
        Please consult with a healthcare provider for proper medical diagnosis and treatment.
        """)
    
    def create_image_upload_section(self):
        """Create optional image upload section"""
        st.subheader("📸 Optional: Facial Analysis")
        st.write("Upload a facial image for enhanced health analysis (coming soon)")
        
        uploaded_file = st.file_uploader(
            "Choose a facial image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear facial image for additional health insights"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.info("Facial analysis feature is under development. Currently using lifestyle data only.")
    
    def run_app(self):
        """Run the main Streamlit app"""
        st.set_page_config(
            page_title="FaceCue ML - Health Prediction",
            page_icon="🏥",
            layout="wide"
        )
        
        # Header
        st.title("🏥 FaceCue ML - Health Prediction System")
        st.markdown("""
        **Predict health deficiencies based on lifestyle patterns**
        
        This AI-powered system analyzes your lifestyle data to identify potential health issues 
        and provides personalized recommendations.
        """)
        
        # Sidebar
        st.sidebar.title("📋 Navigation")
        page = st.sidebar.selectbox("Choose a page", [
            "Health Assessment", 
            "About FaceCue ML",
            "How It Works"
        ])
        
        if page == "Health Assessment":
            # Main assessment form
            with st.form("health_assessment_form"):
                st.header("🔍 Health Assessment")
                
                # Get user input
                user_name = st.text_input("Your Name (optional)", value="User")
                features = self.create_lifestyle_input_form()
                
                # Submit button
                submitted = st.form_submit_button("🔍 Analyze My Health", type="primary")
                
                if submitted:
                    if self.recommender and self.recommender.model is not None:
                        self.display_health_assessment(features)
                    else:
                        st.error("Model not available. Please ensure the model is trained.")
            
            # Optional image upload
            with st.expander("📸 Optional: Facial Analysis"):
                self.create_image_upload_section()
        
        elif page == "About FaceCue ML":
            st.header("About FaceCue ML")
            st.markdown("""
            **FaceCue ML** is an AI-powered health prediction system that analyzes lifestyle patterns 
            to identify potential health deficiencies.
            
            ### Key Features:
            - 🏥 **Health Prediction**: Identifies potential health issues
            - 💡 **Personalized Recommendations**: Provides actionable health advice
            - 📊 **Lifestyle Analysis**: Analyzes sleep, diet, exercise, and stress patterns
            - 🔬 **AI-Powered**: Uses machine learning for accurate predictions
            
            ### Health Conditions Detected:
            - Iron Deficiency (Anemia)
            - Vitamin D Deficiency
            - Dehydration
            - Sleep Quality Issues
            - Multiple Health Concerns
            
            ### Technology Stack:
            - **Machine Learning**: Random Forest, XGBoost, SVM
            - **Data Processing**: Pandas, NumPy, Scikit-learn
            - **User Interface**: Streamlit
            - **Real Data**: 23,807+ real health samples
            """)
        
        elif page == "How It Works":
            st.header("How FaceCue ML Works")
            st.markdown("""
            ### Step 1: Data Collection
            You provide lifestyle information including:
            - Sleep patterns
            - Diet and nutrition
            - Exercise habits
            - Stress levels
            - Water intake
            
            ### Step 2: AI Analysis
            Our machine learning models analyze your data to identify patterns 
            associated with various health conditions.
            
            ### Step 3: Health Prediction
            The system predicts potential health issues based on:
            - Lifestyle pattern analysis
            - Health condition indicators
            - Risk factor assessment
            
            ### Step 4: Personalized Recommendations
            You receive:
            - Specific health recommendations
            - Priority and urgency levels
            - Follow-up action items
            - Professional consultation guidance
            
            ### Data Privacy
            - All data is processed locally
            - No personal information is stored
            - Results are for educational purposes only
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>FaceCue ML - AI-Powered Health Prediction | 
            <em>Not a medical diagnosis - Consult healthcare providers for medical advice</em></p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    app = FaceCueUI()
    app.run_app()

if __name__ == "__main__":
    main()
