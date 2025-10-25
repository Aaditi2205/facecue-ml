
import cv2
import numpy as np
import dlib
from pathlib import Path
import pandas as pd

class FacialFeatureExtractor:
    def __init__(self):
        # Initialize face detector and landmark predictor
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Download dlib shape predictor (68 facial landmarks)
        # You'll need to download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        try:
            self.landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        except:
            print("Warning: dlib shape predictor not found. Using basic features only.")
            self.landmark_predictor = None
    
    def extract_basic_features(self, image_path):
        """Extract basic facial features from image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            # Extract basic features
            features = {
                'face_width': w,
                'face_height': h,
                'face_area': w * h,
                'aspect_ratio': w / h,
                'mean_brightness': np.mean(face_region),
                'std_brightness': np.std(face_region),
                'skin_tone_mean': np.mean(image[y:y+h, x:x+w]),
                'skin_tone_std': np.std(image[y:y+h, x:x+w])
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_advanced_features(self, image_path):
        """Extract advanced facial features using landmarks"""
        if self.landmark_predictor is None:
            return self.extract_basic_features(image_path)
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Convert to dlib rectangle
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            
            # Get facial landmarks
            landmarks = self.landmark_predictor(gray, dlib_rect)
            
            # Extract landmark-based features
            features = self.extract_landmark_features(landmarks, image)
            
            return features
            
        except Exception as e:
            print(f"Error extracting advanced features: {e}")
            return self.extract_basic_features(image_path)
    
    def extract_landmark_features(self, landmarks, image):
        """Extract features based on facial landmarks"""
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Eye features
        left_eye = points[36:42]
        right_eye = points[42:48]
        
        # Nose features
        nose = points[27:36]
        
        # Mouth features
        mouth = points[48:68]
        
        # Extract geometric features
        features = {
            'eye_distance': np.linalg.norm(points[39] - points[42]),
            'nose_width': np.max(nose[:, 0]) - np.min(nose[:, 0]),
            'nose_height': np.max(nose[:, 1]) - np.min(nose[:, 1]),
            'mouth_width': np.max(mouth[:, 0]) - np.min(mouth[:, 0]),
            'mouth_height': np.max(mouth[:, 1]) - np.min(mouth[:, 1]),
            'face_symmetry': self.calculate_symmetry(points),
            'eye_brightness': self.calculate_eye_brightness(image, left_eye, right_eye),
            'lip_color': self.calculate_lip_color(image, mouth)
        }
        
        return features
    
    def calculate_symmetry(self, points):
        """Calculate facial symmetry"""
        # Simple symmetry calculation based on nose center
        nose_center = np.mean(points[27:36], axis=0)
        left_points = points[points[:, 0] < nose_center[0]]
        right_points = points[points[:, 0] > nose_center[0]]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0
        
        left_mean = np.mean(left_points, axis=0)
        right_mean = np.mean(right_points, axis=0)
        
        symmetry = 1 - np.linalg.norm(left_mean - right_mean) / 100
        return max(0, min(1, symmetry))
    
    def calculate_eye_brightness(self, image, left_eye, right_eye):
        """Calculate eye brightness for health analysis"""
        try:
            # Extract eye regions
            left_eye_region = self.extract_region(image, left_eye)
            right_eye_region = self.extract_region(image, right_eye)
            
            if left_eye_region is None or right_eye_region is None:
                return 0
            
            # Calculate brightness
            left_brightness = np.mean(left_eye_region)
            right_brightness = np.mean(right_eye_region)
            
            return (left_brightness + right_brightness) / 2
            
        except:
            return 0
    
    def calculate_lip_color(self, image, mouth):
        """Calculate lip color for health analysis"""
        try:
            mouth_region = self.extract_region(image, mouth)
            if mouth_region is None:
                return 0
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            # Calculate average hue (lip color indicator)
            avg_hue = np.mean(hsv[:, :, 0])
            return avg_hue
            
        except:
            return 0
    
    def extract_region(self, image, points):
        """Extract region of interest from image"""
        try:
            x_min, y_min = np.min(points, axis=0).astype(int)
            x_max, y_max = np.max(points, axis=0).astype(int)
            
            # Add padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(image.shape[1], x_max + padding)
            y_max = min(image.shape[0], y_max + padding)
            
            return image[y_min:y_max, x_min:x_max]
            
        except:
            return None

# Usage example
if __name__ == "__main__":
    extractor = FacialFeatureExtractor()
    
    # Process a single image
    image_path = "path/to/face/image.jpg"
    features = extractor.extract_advanced_features(image_path)
    
    if features:
        print("Extracted facial features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
