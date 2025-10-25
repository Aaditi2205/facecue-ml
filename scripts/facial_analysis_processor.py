# FaceCue ML - Facial Analysis Processor
# Processes uploaded facial images for health analysis

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

class FacialAnalysisProcessor:
    def __init__(self):
        self.face_cascade = None
        self.initialize_face_detection()
    
    def initialize_face_detection(self):
        """Initialize OpenCV face detection"""
        try:
            # Try to load Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                st.warning("⚠️ Face detection model not available")
                return False
            return True
        except Exception as e:
            st.error(f"❌ Error initializing face detection: {e}")
            return False
    
    def process_uploaded_image(self, uploaded_file):
        """Process uploaded facial image for health analysis"""
        try:
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_file)
            
            # Convert to OpenCV format
            img_array = np.array(image)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Detect face
            faces = self.detect_faces(img_cv)
            
            if len(faces) == 0:
                return {
                    'success': False,
                    'error': 'No face detected in image',
                    'recommendations': ['Please upload a clear facial image with visible face']
                }
            
            # Analyze facial features
            facial_features = self.extract_facial_features(img_cv, faces[0])
            
            return {
                'success': True,
                'facial_features': facial_features,
                'face_detected': True,
                'analysis': self.interpret_facial_features(facial_features)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing image: {e}',
                'recommendations': ['Please try uploading a different image']
            }
    
    def detect_faces(self, img):
        """Detect faces in image using OpenCV"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def extract_facial_features(self, img, face_rect):
        """Extract facial features for health analysis"""
        x, y, w, h = face_rect
        
        # Extract face region
        face_region = img[y:y+h, x:x+w]
        
        # Analyze different facial regions
        features = {}
        
        # 1. Skin tone analysis
        features['skin_tone'] = self.analyze_skin_tone(face_region)
        
        # 2. Eye region analysis
        features['eye_brightness'] = self.analyze_eye_region(face_region)
        
        # 3. Lip color analysis
        features['lip_color'] = self.analyze_lip_region(face_region)
        
        # 4. Overall face symmetry
        features['face_symmetry'] = self.analyze_face_symmetry(face_region)
        
        # 5. General skin condition
        features['skin_condition'] = self.analyze_skin_condition(face_region)
        
        return features
    
    def analyze_skin_tone(self, face_region):
        """Analyze skin tone for health indicators"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Calculate average skin tone
        skin_pixels = face_region[skin_mask > 0]
        if len(skin_pixels) > 0:
            avg_bgr = np.mean(skin_pixels, axis=0)
            return {
                'average_color': avg_bgr.tolist(),
                'pallor_score': self.calculate_pallor_score(avg_bgr),
                'skin_area': np.sum(skin_mask > 0)
            }
        
        return {'average_color': [0, 0, 0], 'pallor_score': 0, 'skin_area': 0}
    
    def calculate_pallor_score(self, bgr_color):
        """Calculate pallor score (higher = more pale)"""
        # Convert BGR to RGB
        rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]
        
        # Calculate brightness
        brightness = sum(rgb_color) / 3
        
        # Calculate pallor (higher brightness = more pale)
        pallor_score = brightness / 255.0
        
        return pallor_score
    
    def analyze_eye_region(self, face_region):
        """Analyze eye region for health indicators"""
        h, w = face_region.shape[:2]
        
        # Define eye region (upper third of face)
        eye_region = face_region[0:h//3, :]
        
        # Convert to grayscale
        gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        brightness = np.mean(gray_eyes)
        
        # Calculate contrast
        contrast = np.std(gray_eyes)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'fatigue_indicator': self.calculate_fatigue_indicator(brightness, contrast)
        }
    
    def calculate_fatigue_indicator(self, brightness, contrast):
        """Calculate fatigue indicator from eye analysis"""
        # Lower brightness and contrast may indicate fatigue
        fatigue_score = (255 - brightness) / 255.0 + (255 - contrast) / 255.0
        return min(fatigue_score, 1.0)
    
    def analyze_lip_region(self, face_region):
        """Analyze lip region for health indicators"""
        h, w = face_region.shape[:2]
        
        # Define lip region (lower third of face)
        lip_region = face_region[2*h//3:, :]
        
        # Convert to HSV
        hsv_lips = cv2.cvtColor(lip_region, cv2.COLOR_BGR2HSV)
        
        # Define lip color range
        lower_lip = np.array([0, 50, 50], dtype=np.uint8)
        upper_lip = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create lip mask
        lip_mask = cv2.inRange(hsv_lips, lower_lip, upper_lip)
        
        # Calculate average lip color
        lip_pixels = lip_region[lip_mask > 0]
        if len(lip_pixels) > 0:
            avg_lip_color = np.mean(lip_pixels, axis=0)
            return {
                'average_color': avg_lip_color.tolist(),
                'color_vitality': self.calculate_color_vitality(avg_lip_color),
                'lip_area': np.sum(lip_mask > 0)
            }
        
        return {'average_color': [0, 0, 0], 'color_vitality': 0, 'lip_area': 0}
    
    def calculate_color_vitality(self, bgr_color):
        """Calculate color vitality (higher = more vibrant)"""
        # Convert BGR to RGB
        rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]
        
        # Calculate saturation (vibrancy)
        max_val = max(rgb_color)
        min_val = min(rgb_color)
        
        if max_val == 0:
            return 0
        
        saturation = (max_val - min_val) / max_val
        return saturation
    
    def analyze_face_symmetry(self, face_region):
        """Analyze face symmetry"""
        h, w = face_region.shape[:2]
        
        # Split face into left and right halves
        left_half = face_region[:, :w//2]
        right_half = face_region[:, w//2:]
        
        # Flip right half to compare with left
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Calculate difference
        diff = cv2.absdiff(left_half, right_half_flipped)
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        return {
            'symmetry_score': symmetry_score,
            'asymmetry_level': 1.0 - symmetry_score
        }
    
    def analyze_skin_condition(self, face_region):
        """Analyze general skin condition"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate smoothness
        smoothness = 1.0 / (1.0 + laplacian_var / 1000.0)
        
        return {
            'texture_variance': laplacian_var,
            'smoothness': smoothness,
            'skin_health_score': smoothness
        }
    
    def interpret_facial_features(self, features):
        """Interpret facial features for health insights"""
        insights = []
        
        # Skin tone analysis
        pallor_score = features['skin_tone']['pallor_score']
        if pallor_score > 0.7:
            insights.append("Pale skin tone detected - may indicate anemia or poor circulation")
        elif pallor_score < 0.3:
            insights.append("Healthy skin tone detected")
        
        # Eye analysis
        fatigue_indicator = features['eye_brightness']['fatigue_indicator']
        if fatigue_indicator > 0.6:
            insights.append("Eye region suggests fatigue or sleep deficiency")
        
        # Lip color analysis
        color_vitality = features['lip_color']['color_vitality']
        if color_vitality < 0.3:
            insights.append("Pale lip color - may indicate poor circulation or anemia")
        
        # Face symmetry
        asymmetry = features['face_symmetry']['asymmetry_level']
        if asymmetry > 0.3:
            insights.append("Facial asymmetry detected - may indicate stress or fatigue")
        
        # Skin condition
        skin_health = features['skin_condition']['skin_health_score']
        if skin_health < 0.5:
            insights.append("Skin texture suggests possible dehydration or poor nutrition")
        
        return {
            'insights': insights,
            'overall_health_score': self.calculate_overall_health_score(features),
            'recommendations': self.generate_facial_recommendations(features)
        }
    
    def calculate_overall_health_score(self, features):
        """Calculate overall health score from facial features"""
        scores = []
        
        # Skin tone score (lower pallor = better)
        scores.append(1.0 - features['skin_tone']['pallor_score'])
        
        # Eye health score (lower fatigue = better)
        scores.append(1.0 - features['eye_brightness']['fatigue_indicator'])
        
        # Lip vitality score
        scores.append(features['lip_color']['color_vitality'])
        
        # Face symmetry score
        scores.append(features['face_symmetry']['symmetry_score'])
        
        # Skin condition score
        scores.append(features['skin_condition']['skin_health_score'])
        
        return np.mean(scores)
    
    def generate_facial_recommendations(self, features):
        """Generate recommendations based on facial analysis"""
        recommendations = []
        
        pallor_score = features['skin_tone']['pallor_score']
        if pallor_score > 0.7:
            recommendations.append("Consider iron-rich foods and vitamin C for better absorption")
            recommendations.append("Get a complete blood count (CBC) to check for anemia")
        
        fatigue_indicator = features['eye_brightness']['fatigue_indicator']
        if fatigue_indicator > 0.6:
            recommendations.append("Improve sleep hygiene and reduce screen time")
            recommendations.append("Consider stress management techniques")
        
        color_vitality = features['lip_color']['color_vitality']
        if color_vitality < 0.3:
            recommendations.append("Increase hydration and iron intake")
            recommendations.append("Monitor circulation and blood pressure")
        
        return recommendations

def main():
    """Test the facial analysis processor"""
    print("📸 Facial Analysis Processor")
    print("="*40)
    
    processor = FacialAnalysisProcessor()
    
    if processor.face_cascade is None:
        print("❌ Face detection not available")
        return
    
    print("✅ Facial analysis processor ready")
    print("✅ Face detection initialized")
    print("✅ Ready for image processing")

if __name__ == "__main__":
    main()
