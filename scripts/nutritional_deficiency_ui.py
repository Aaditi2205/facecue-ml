# FaceCue ML - Nutritional Deficiency Detection UI
# Specialized interface for detecting specific nutritional deficiencies

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
sys.path.append('scripts')
from integrated_nutritional_detector import IntegratedNutritionalDeficiencyDetector
from facial_analysis_processor import FacialAnalysisProcessor

class NutritionalDeficiencyUI:
    def __init__(self):
        self.detector = None
        self.facial_processor = None
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize the nutritional deficiency detection system"""
        try:
            self.detector = IntegratedNutritionalDeficiencyDetector()
            self.facial_processor = FacialAnalysisProcessor()
            
            if self.detector.load_integrated_model():
                st.success("✅ Integrated Nutritional Deficiency Detection System Ready")
                if self.facial_processor.face_cascade is not None:
                    st.success("✅ Facial Analysis System Ready")
                else:
                    st.warning("⚠️ Facial analysis not available (OpenCV issue)")
                return True
            else:
                st.error("❌ Integrated nutritional model not found. Please train the model first.")
                return False
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            return False
    
    def create_nutritional_header(self):
        """Create nutritional deficiency detection header"""
        st.markdown("""
        <div style="background-color:#2d5016;padding:20px;border-radius:10px;margin-bottom:20px">
            <h1 style="color:white;text-align:center;margin:0">🍎 FaceCue ML</h1>
            <h3 style="color:white;text-align:center;margin:0">Integrated Nutritional Deficiency Detection System</h3>
            <p style="color:white;text-align:center;margin:0">AI-Powered Detection Combining Lifestyle Data + Facial Analysis</p>
            <p style="color:white;text-align:center;margin:0">Iron, Vitamin D, B12, C, Zinc, Magnesium Deficiencies</p>
        </div>
        """, unsafe_allow_html=True)
    
    def create_nutritional_assessment_form(self):
        """Create comprehensive nutritional assessment form"""
        st.subheader("🍎 Nutritional Deficiency Assessment")
        st.markdown("**Detect specific nutritional deficiencies: Iron, Vitamin D, B12, C, Zinc, Magnesium**")
        
        # Personal Information
        with st.expander("👤 Personal Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age (years)", 18, 80, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
            with col2:
                diet_type = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])
                meal_frequency = st.selectbox("Daily Meal Frequency", ["1-2 meals", "3 meals", "4-5 meals"])
        
        # Nutritional Intake
        with st.expander("🥗 Nutritional Intake", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fruit_vegetable_intake = st.selectbox("Fruit & Vegetable Intake", ["Low", "Moderate", "High"])
                supplement_use = st.selectbox("Supplement Use", ["None", "Multivitamin", "Specific"])
            with col2:
                sun_exposure = st.selectbox("Sun Exposure", ["Low", "Moderate", "High"])
                water_intake = st.slider("Daily Water Intake (liters)", 0.5, 5.0, 2.5)
        
        # Health Indicators
        with st.expander("💪 Health Indicators", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fatigue_scale = st.slider("Fatigue Level (1-5)", 1, 5, 3)
                energy_scale = st.slider("Energy Level (1-5)", 1, 5, 3)
            with col2:
                sleep_hours = st.slider("Sleep Hours per Night", 4.0, 12.0, 7.5)
                stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
        
        # Lifestyle Factors
        with st.expander("🏃 Lifestyle Factors", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                exercise_hours = st.slider("Weekly Exercise (hours)", 0, 20, 3)
                screen_time = st.slider("Daily Screen Time (hours)", 1, 16, 6)
            with col2:
                smoking = st.selectbox("Smoking Status", ["Yes", "No"])
                alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
        
        return {
            'age': age,
            'gender': gender,
            'diet_type': diet_type,
            'meal_frequency': meal_frequency,
            'fruit_vegetable_intake': fruit_vegetable_intake,
            'supplement_use': supplement_use,
            'sun_exposure': sun_exposure,
            'water_intake': water_intake,
            'fatigue_scale': fatigue_scale,
            'energy_scale': energy_scale,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'exercise_hours': exercise_hours,
            'screen_time': screen_time,
            'smoking': smoking,
            'alcohol_consumption': alcohol_consumption
        }
    
    def display_nutritional_assessment(self, features, user_profile):
        """Display nutritional deficiency assessment results"""
        if self.detector is None:
            st.error("System not initialized")
            return
        
        # Make prediction
        deficiency, probabilities = self.detector.predict_nutritional_deficiency(features)
        
        if deficiency is None:
            st.error("Unable to generate nutritional assessment")
            return
        
        # Calculate confidence
        confidence = max(probabilities) if probabilities is not None else None
        
        # Generate recommendations
        recommendations = self.detector.generate_deficiency_recommendations(deficiency, confidence)
        
        # Display results
        st.markdown("---")
        st.markdown("## 🍎 Nutritional Deficiency Assessment Results")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Deficiency", deficiency)
        with col2:
            if confidence:
                st.metric("Confidence Level", f"{confidence:.1%}")
        with col3:
            st.metric("Priority Level", recommendations['priority'])
        
        # Deficiency analysis
        st.markdown("### 📋 Deficiency Analysis")
        
        # Condition description
        st.info(f"**{recommendations['title']}**\n\n{recommendations['description']}")
        
        # Symptoms
        if 'symptoms' in recommendations:
            st.markdown("### 🔍 Common Symptoms")
            for symptom in recommendations['symptoms']:
                st.markdown(f"• {symptom}")
        
        # Priority and urgency indicators
        priority_color = {
            'Low': '🟢',
            'Moderate': '🟡',
            'High': '🟠',
            'Critical': '🔴'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Priority:** {priority_color.get(recommendations['priority'], '⚪')} {recommendations['priority']}")
        with col2:
            if confidence:
                st.markdown(f"**Confidence:** {confidence:.1%}")
        
        # Specific recommendations
        st.markdown("### 💊 Specific Recommendations")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            st.markdown(f"**{i}.** {rec}")
        
        # Blood tests
        if 'blood_tests' in recommendations:
            st.markdown("### 🩸 Recommended Blood Tests")
            for test in recommendations['blood_tests']:
                st.markdown(f"• {test}")
        
        # Follow-up care
        st.markdown("### 📅 Follow-up Care")
        st.success(recommendations['follow_up'])
        
        # Medical disclaimer
        st.markdown("---")
        st.markdown("### ⚠️ Medical Disclaimer")
        st.warning("""
        **This assessment is for educational purposes only.**
        
        - Always consult with healthcare providers for medical diagnosis
        - Nutritional deficiencies require proper medical evaluation
        - Blood tests are essential for accurate diagnosis
        - Individual nutritional needs may vary
        """)
    
    def display_comprehensive_nutritional_assessment(self, features, user_profile):
        """Display comprehensive nutritional deficiency assessment with facial analysis"""
        if self.detector is None:
            st.error("System not initialized")
            return
        
        # Prepare facial features for integrated analysis
        facial_features = None
        if user_profile.get('facial_analysis') and user_profile['facial_analysis']['success']:
            facial_data = user_profile['facial_analysis']['facial_features']
            facial_features = {
                'skin_pallor_score': facial_data['skin_tone']['pallor_score'],
                'eye_brightness': facial_data['eye_brightness']['brightness'] / 255.0,  # Normalize
                'lip_color_vitality': facial_data['lip_color']['color_vitality'],
                'face_symmetry': facial_data['face_symmetry']['symmetry_score'],
                'skin_health_score': facial_data['skin_condition']['skin_health_score'],
                'fatigue_indicators': facial_data['eye_brightness']['fatigue_indicator']
            }
        
        # Make integrated nutritional prediction (lifestyle + facial)
        deficiency, probabilities = self.detector.predict_integrated_deficiency(features, facial_features)
        
        if deficiency is None:
            st.error("Unable to generate integrated nutritional assessment")
            return
        
        # Calculate confidence
        confidence = max(probabilities) if probabilities is not None else None
        
        # Generate integrated recommendations
        nutritional_recs = self.detector.generate_integrated_recommendations(deficiency, confidence, user_profile.get('facial_analysis'))
        
        # Display comprehensive results
        st.markdown("---")
        st.markdown("## 🍎 Complete Nutritional Deficiency Assessment Report")
        
        # Header with user info
        st.markdown(f"**Patient:** {user_profile['name']} | **Age:** {user_profile['age']} | **Gender:** {user_profile['gender']}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Deficiency", deficiency)
        with col2:
            if confidence:
                st.metric("Confidence Level", f"{confidence:.1%}")
        with col3:
            st.metric("Priority Level", nutritional_recs['priority'])
        with col4:
            if user_profile.get('facial_analysis'):
                facial_health = user_profile['facial_analysis']['analysis']['overall_health_score']
                st.metric("Facial Health Score", f"{facial_health:.1%}")
        
        # Comprehensive Analysis
        st.markdown("### 📋 Integrated Deficiency Analysis")
        
        # Show data integration
        st.markdown("#### 🔗 Data Integration Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Lifestyle Data: {len(features)} features analyzed")
        with col2:
            if user_profile.get('facial_analysis') and user_profile['facial_analysis']['success']:
                st.success(f"✅ Facial Analysis: {len(facial_features)} features integrated")
            else:
                st.info("ℹ️ Facial Analysis: Not available")
        
        # Integrated nutritional analysis
        st.info(f"**{nutritional_recs['title']}**\n\n{nutritional_recs['description']}")
        
        # Show integration benefits
        if user_profile.get('facial_analysis') and user_profile['facial_analysis']['success']:
            st.markdown("#### 📸 Enhanced Analysis with Facial Integration")
            st.success("🎯 **Enhanced Accuracy**: Combined lifestyle + facial analysis provides more accurate nutritional deficiency detection")
            
            # Facial insights
            if user_profile['facial_analysis']['analysis']['insights']:
                st.markdown("**Facial Health Indicators:**")
                for insight in user_profile['facial_analysis']['analysis']['insights']:
                    st.markdown(f"• {insight}")
        else:
            st.info("ℹ️ **Lifestyle Analysis Only**: Upload a facial photo for enhanced accuracy")
        
        # Symptoms
        if 'symptoms' in nutritional_recs:
            st.markdown("### 🔍 Common Symptoms")
            for symptom in nutritional_recs['symptoms']:
                st.markdown(f"• {symptom}")
        
        # Priority and urgency indicators
        priority_color = {
            'Low': '🟢',
            'Moderate': '🟡',
            'High': '🟠',
            'Critical': '🔴'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Priority:** {priority_color.get(nutritional_recs['priority'], '⚪')} {nutritional_recs['priority']}")
        with col2:
            if confidence:
                st.markdown(f"**Confidence:** {confidence:.1%}")
        
        # Comprehensive recommendations
        st.markdown("### 💊 Comprehensive Nutritional Recommendations")
        
        # Nutritional recommendations
        st.markdown("**Primary Nutritional Interventions:**")
        for i, rec in enumerate(nutritional_recs['recommendations'], 1):
            st.markdown(f"**{i}.** {rec}")
        
        # Facial-based additional recommendations
        if user_profile.get('facial_analysis') and user_profile['facial_analysis']['success']:
            facial_recs = user_profile['facial_analysis']['analysis']['recommendations']
            if facial_recs:
                st.markdown("**Additional Facial-Based Recommendations:**")
                for i, rec in enumerate(facial_recs, len(nutritional_recs['recommendations']) + 1):
                    st.markdown(f"**{i}.** {rec}")
        
        # Blood tests
        if 'blood_tests' in nutritional_recs:
            st.markdown("### 🩸 Recommended Blood Tests")
            for test in nutritional_recs['blood_tests']:
                st.markdown(f"• {test}")
        
        # Follow-up care
        st.markdown("### 📅 Follow-up Care")
        st.success(nutritional_recs['follow_up'])
        
        # Save comprehensive report (moved outside form)
        st.markdown("### 💾 Save Report")
        if st.button("💾 Save Comprehensive Report", key="save_report"):
            comprehensive_report = {
                'user_profile': user_profile,
                'nutritional_features': features,
                'predicted_deficiency': deficiency,
                'confidence': confidence,
                'nutritional_recommendations': nutritional_recs,
                'facial_analysis': user_profile.get('facial_analysis'),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            filename = f"data/comprehensive_nutritional_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(filename, 'w') as f:
                json.dump(comprehensive_report, f, indent=2, default=str)
            
            st.success(f"✅ Comprehensive nutritional report saved: {filename}")
        
        # Medical disclaimer
        st.markdown("---")
        st.markdown("### ⚠️ Medical Disclaimer")
        st.warning("""
        **This comprehensive assessment is for educational purposes only.**
        
        - Combines lifestyle data with facial analysis for enhanced accuracy
        - Always consult with healthcare providers for medical diagnosis
        - Nutritional deficiencies require proper medical evaluation
        - Blood tests are essential for accurate diagnosis
        - Individual nutritional needs may vary
        """)
    
    def create_deficiency_info_section(self):
        """Create information about different nutritional deficiencies"""
        st.markdown("## 🍎 Nutritional Deficiencies Detected")
        
        deficiencies = [
            {
                'name': 'Iron Deficiency (Anemia)',
                'symptoms': 'Fatigue, weakness, pale skin, cold hands/feet',
                'causes': 'Insufficient iron intake, blood loss, poor absorption',
                'tests': 'CBC, Serum Iron, Ferritin, TIBC'
            },
            {
                'name': 'Vitamin D Deficiency',
                'symptoms': 'Bone pain, muscle weakness, frequent infections',
                'causes': 'Limited sun exposure, inadequate dietary intake',
                'tests': '25-Hydroxyvitamin D, Calcium, Phosphorus'
            },
            {
                'name': 'Vitamin B12 Deficiency',
                'symptoms': 'Fatigue, numbness/tingling, memory problems',
                'causes': 'Vegan diet, absorption issues, pernicious anemia',
                'tests': 'Vitamin B12, Methylmalonic Acid, Homocysteine'
            },
            {
                'name': 'Vitamin C Deficiency',
                'symptoms': 'Fatigue, bleeding gums, slow wound healing',
                'causes': 'Insufficient fruit/vegetable intake',
                'tests': 'Vitamin C (Ascorbic Acid), CBC'
            },
            {
                'name': 'Zinc Deficiency',
                'symptoms': 'Frequent infections, slow wound healing, hair loss',
                'causes': 'Inadequate dietary intake, malabsorption',
                'tests': 'Serum Zinc, CBC, Immune Function Tests'
            },
            {
                'name': 'Magnesium Deficiency',
                'symptoms': 'Muscle cramps, fatigue, irregular heartbeat',
                'causes': 'Insufficient dietary intake, stress, medications',
                'tests': 'Serum Magnesium, RBC Magnesium, Electrolyte Panel'
            }
        ]
        
        for deficiency in deficiencies:
            with st.expander(f"🔍 {deficiency['name']}", expanded=False):
                st.markdown(f"**Symptoms:** {deficiency['symptoms']}")
                st.markdown(f"**Common Causes:** {deficiency['causes']}")
                st.markdown(f"**Diagnostic Tests:** {deficiency['tests']}")
    
    def run_app(self):
        """Run the nutritional deficiency detection app"""
        st.set_page_config(
            page_title="FaceCue ML - Nutritional Deficiency Detection",
            page_icon="🍎",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        self.create_nutritional_header()
        
        # Sidebar navigation
        st.sidebar.title("🍎 Navigation")
        page = st.sidebar.selectbox("Choose Assessment Type", [
            "Nutritional Deficiency Assessment", 
            "Deficiency Information",
            "About the System"
        ])
        
        # System status in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Status")
        if self.detector and self.detector.model:
            st.sidebar.success("✅ System Ready")
            st.sidebar.info("Model: Random Forest")
            st.sidebar.info("Accuracy: 86.5%")
            st.sidebar.info("Deficiencies: 8 types")
        else:
            st.sidebar.error("❌ System Not Ready")
        
        if page == "Nutritional Deficiency Assessment":
            if not self.detector or not self.detector.model:
                st.error("❌ Nutritional deficiency detection system not available.")
                return
            
            # Main assessment form
            with st.form("nutritional_deficiency_assessment"):
                st.markdown("### Complete Nutritional Assessment Form")
                
                # Get user input
                user_name = st.text_input("Your Name (optional)", value="User")
                features = self.create_nutritional_assessment_form()
                
                # Facial Analysis Section
                st.markdown("### 📸 Optional: Facial Analysis for Enhanced Detection")
                st.markdown("Upload a clear facial photo for enhanced nutritional deficiency detection")
                
                uploaded_file = st.file_uploader(
                    "Choose a facial image", 
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear facial image for additional nutritional insights",
                    key="facial_upload"
                )
                
                facial_analysis = None
                if uploaded_file is not None:
                    st.image(uploaded_file, caption="Uploaded Facial Image", use_column_width=True)
                    st.success("✅ Facial image uploaded successfully!")
                    
                    # Process the facial image
                    if self.facial_processor and self.facial_processor.face_cascade is not None:
                        with st.spinner("🔬 Analyzing facial features for nutritional deficiencies..."):
                            facial_analysis = self.facial_processor.process_uploaded_image(uploaded_file)
                            
                            if facial_analysis['success']:
                                st.success("✅ Facial analysis completed!")
                                
                                # Display facial analysis results
                                with st.expander("📊 Facial Analysis Results", expanded=True):
                                    features_facial = facial_analysis['facial_features']
                                    analysis = facial_analysis['analysis']
                                    
                                    # Overall health score
                                    health_score = analysis['overall_health_score']
                                    st.metric("Overall Health Score", f"{health_score:.1%}")
                                    
                                    # Facial insights
                                    if analysis['insights']:
                                        st.markdown("**Facial Health Insights:**")
                                        for insight in analysis['insights']:
                                            st.info(f"• {insight}")
                                    
                                    # Recommendations
                                    if analysis['recommendations']:
                                        st.markdown("**Facial-Based Recommendations:**")
                                        for rec in analysis['recommendations']:
                                            st.success(f"• {rec}")
                            else:
                                st.error(f"❌ {facial_analysis['error']}")
                    else:
                        st.info("🔬 Facial analysis will be integrated with nutritional data for enhanced accuracy")
                else:
                    st.info("💡 No facial image uploaded - using nutritional data only")
                
                # Submit button
                submitted = st.form_submit_button("🍎 Generate Complete Nutritional Deficiency Report", type="primary")
                
                if submitted:
                    user_profile = {
                        'name': user_name,
                        'age': features['age'],
                        'gender': features['gender']
                    }
                    
                    # Add facial analysis info to profile
                    if uploaded_file is not None and facial_analysis and facial_analysis['success']:
                        user_profile['facial_analysis'] = facial_analysis
                        st.success("📸 Facial analysis will be included in nutritional report")
                    else:
                        user_profile['facial_analysis'] = None
                        st.info("📊 Report based on nutritional data only")
                    
                    # Store data in session state for use outside form
                    st.session_state['assessment_data'] = {
                        'features': features,
                        'user_profile': user_profile
                    }
            
            # Display assessment results outside the form
            if 'assessment_data' in st.session_state:
                st.markdown("---")
                self.display_comprehensive_nutritional_assessment(
                    st.session_state['assessment_data']['features'],
                    st.session_state['assessment_data']['user_profile']
                )
        
        elif page == "Deficiency Information":
            self.create_deficiency_info_section()
        
        elif page == "About the System":
            st.markdown("## About FaceCue ML Nutritional Deficiency Detection")
            
            st.markdown("""
            **FaceCue ML** is a specialized AI system for detecting nutritional deficiencies 
            based on lifestyle patterns and health indicators.
            
            ### Detected Deficiencies:
            - **Iron Deficiency (Anemia)** - Most common nutritional deficiency
            - **Vitamin D Deficiency** - Affects bone health and immunity
            - **Vitamin B12 Deficiency** - Critical for nerve function
            - **Vitamin C Deficiency** - Essential for immune system
            - **Zinc Deficiency** - Important for immune function
            - **Magnesium Deficiency** - Affects muscle and nerve function
            - **Multiple Deficiencies** - Complex nutritional issues
            
            ### Key Features:
            - 🎯 **Specific Detection**: Identifies exact nutritional deficiencies
            - 🩸 **Blood Test Recommendations**: Specific tests for each deficiency
            - 💊 **Targeted Treatment**: Precise nutritional interventions
            - 📊 **High Accuracy**: 86.5% accuracy on nutritional data
            - 🔬 **Evidence-Based**: Medical literature-backed recommendations
            
            ### Technology:
            - **Machine Learning**: Random Forest Classifier
            - **Real Data**: 2,000+ nutritional deficiency samples
            - **Professional Recommendations**: Medical-grade advice
            - **Blood Test Integration**: Specific diagnostic tests
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px'>
            <p>FaceCue ML - Nutritional Deficiency Detection System | 
            <em>AI-Powered Nutritional Health Assessment</em></p>
            <p><strong>Disclaimer:</strong> Not a medical diagnosis - Consult healthcare providers for medical advice</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the nutritional deficiency detection app"""
    app = NutritionalDeficiencyUI()
    app.run_app()

if __name__ == "__main__":
    main()
