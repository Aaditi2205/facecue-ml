# FaceCue ML - Simple UTKFace Processor
# Processes real UTKFace images for health prediction

import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class SimpleUTKFaceProcessor:
    def __init__(self, utkface_dir="data/images/utkface"):
        self.utkface_dir = Path(utkface_dir)
        self.processed_data = None
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def check_utkface_availability(self):
        """Check if UTKFace dataset is available"""
        if not self.utkface_dir.exists():
            print(f"ERROR: UTKFace directory not found: {self.utkface_dir}")
            return False
        
        image_files = list(self.utkface_dir.glob("*.jpg")) + list(self.utkface_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"ERROR: No images found in {self.utkface_dir}")
            print("Please download UTKFace dataset from: https://susanqq.github.io/UTKFace/")
            return False
        
        print(f"SUCCESS: Found {len(image_files)} UTKFace images")
        return True
    
    def parse_utkface_filename(self, filename):
        """Parse UTKFace filename to extract age, gender, ethnicity"""
        try:
            # UTKFace filename format: age_gender_race_date&time.jpg
            parts = filename.split('_')
            if len(parts) >= 3:
                age = int(parts[0])
                gender = int(parts[1])  # 0=male, 1=female
                ethnicity = int(parts[2])  # 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
                return age, gender, ethnicity
        except:
            pass
        return None, None, None
    
    def extract_facial_features(self, image_path):
        """Extract facial features from UTKFace image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Basic image features
            height, width = image.shape[:2]
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Extract face region
                face_region = image[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                
                # Calculate features
                face_area = w * h
                aspect_ratio = w / h
                
                # Brightness analysis
                mean_brightness = np.mean(face_gray)
                std_brightness = np.std(face_gray)
                
                # Color analysis
                mean_color = np.mean(face_region, axis=(0, 1))
                
                # Skin tone analysis (using HSV)
                face_hsv = hsv[y:y+h, x:x+w]
                skin_tone_mean = np.mean(face_hsv[:, :, 0])  # Hue
                
                # Eye region analysis (approximate)
                eye_region_y = int(h * 0.25)
                eye_region_h = int(h * 0.25)
                eye_region = face_gray[eye_region_y:eye_region_y+eye_region_h, :]
                
                if eye_region.size > 0:
                    eye_brightness = np.mean(eye_region)
                else:
                    eye_brightness = mean_brightness
                
                # Lip region analysis (approximate)
                lip_region_y = int(h * 0.6)
                lip_region_h = int(h * 0.2)
                lip_region = face_gray[lip_region_y:lip_region_y+lip_region_h, :]
                
                if lip_region.size > 0:
                    lip_brightness = np.mean(lip_region)
                else:
                    lip_brightness = mean_brightness
                
                features = {
                    'image_width': width,
                    'image_height': height,
                    'face_width': w,
                    'face_height': h,
                    'face_area': face_area,
                    'aspect_ratio': aspect_ratio,
                    'mean_brightness': mean_brightness,
                    'std_brightness': std_brightness,
                    'skin_tone_mean': skin_tone_mean,
                    'eye_brightness': eye_brightness,
                    'lip_brightness': lip_brightness,
                    'mean_color_b': mean_color[0],
                    'mean_color_g': mean_color[1],
                    'mean_color_r': mean_color[2]
                }
                
                return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        
        return None
    
    def calculate_health_score(self, features, age):
        """Calculate health score based on facial features"""
        health_score = 0
        
        # Age-related health indicators
        if age > 60:
            health_score += 1
        
        # Skin tone indicators
        if features['skin_tone_mean'] < 10:  # Very pale
            health_score += 2
        elif features['skin_tone_mean'] < 15:  # Pale
            health_score += 1
        
        # Brightness indicators
        if features['mean_brightness'] < 100:  # Dull appearance
            health_score += 1
        
        # Eye brightness indicators
        if features['eye_brightness'] < 80:  # Dark circles
            health_score += 1
        
        # Lip brightness indicators
        if features['lip_brightness'] < 90:  # Pale lips
            health_score += 1
        
        return health_score
    
    def determine_health_status(self, health_score):
        """Determine health status based on score"""
        if health_score >= 4:
            return 'Multiple Health Issues'
        elif health_score >= 2:
            return 'Mild Health Issues'
        elif health_score >= 1:
            return 'Possible Health Issues'
        else:
            return 'Good Health'
    
    def process_utkface_dataset(self, max_images=1000):
        """Process UTKFace dataset and extract features"""
        if not self.check_utkface_availability():
            return None
        
        print(f"Processing UTKFace dataset (max {max_images} images)...")
        
        processed_data = []
        processed_count = 0
        
        # Get all image files
        image_files = list(self.utkface_dir.glob("*.jpg")) + list(self.utkface_dir.glob("*.png"))
        
        print(f"Found {len(image_files)} images in UTKFace dataset")
        
        for image_file in image_files:
            if processed_count >= max_images:
                break
            
            # Parse filename
            age, gender, ethnicity = self.parse_utkface_filename(image_file.name)
            
            if age is None:
                continue
            
            # Extract facial features
            features = self.extract_facial_features(image_file)
            
            if features is not None:
                # Add metadata
                features['filename'] = image_file.name
                features['age'] = age
                features['gender'] = gender
                features['ethnicity'] = ethnicity
                
                # Create health indicators based on facial features
                health_score = self.calculate_health_score(features, age)
                features['health_score'] = health_score
                features['health_status'] = self.determine_health_status(health_score)
                
                processed_data.append(features)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
        
        self.processed_data = pd.DataFrame(processed_data)
        print(f"SUCCESS: Processed {len(self.processed_data)} UTKFace images")
        
        return self.processed_data
    
    def train_health_model(self):
        """Train health prediction model on UTKFace data"""
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_utkface_dataset() first.")
        
        print("Training health prediction model on UTKFace data...")
        
        # Prepare features
        feature_cols = ['age', 'face_width', 'face_height', 'aspect_ratio', 
                       'mean_brightness', 'skin_tone_mean', 'eye_brightness', 
                       'lip_brightness', 'mean_color_b', 'mean_color_g', 'mean_color_r']
        
        X = self.processed_data[feature_cols]
        y = self.processed_data['health_status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"UTKFace Health Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return accuracy, feature_importance
    
    def visualize_utkface_analysis(self):
        """Create visualizations of UTKFace health analysis"""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        plt.figure(figsize=(15, 12))
        
        # Health status distribution
        plt.subplot(2, 3, 1)
        self.processed_data['health_status'].value_counts().plot(kind='bar')
        plt.title('Health Status Distribution (UTKFace)')
        plt.xticks(rotation=45)
        
        # Age vs health
        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.processed_data, x='health_status', y='age')
        plt.title('Age vs Health Status')
        plt.xticks(rotation=45)
        
        # Skin tone vs health
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.processed_data, x='health_status', y='skin_tone_mean')
        plt.title('Skin Tone vs Health Status')
        plt.xticks(rotation=45)
        
        # Brightness vs health
        plt.subplot(2, 3, 4)
        sns.boxplot(data=self.processed_data, x='health_status', y='mean_brightness')
        plt.title('Brightness vs Health Status')
        plt.xticks(rotation=45)
        
        # Gender distribution
        plt.subplot(2, 3, 5)
        gender_labels = ['Male', 'Female']
        gender_counts = self.processed_data['gender'].value_counts()
        plt.pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%')
        plt.title('Gender Distribution')
        
        # Ethnicity distribution
        plt.subplot(2, 3, 6)
        ethnicity_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
        ethnicity_counts = self.processed_data['ethnicity'].value_counts()
        plt.pie(ethnicity_counts.values, labels=ethnicity_labels, autopct='%1.1f%%')
        plt.title('Ethnicity Distribution')
        
        plt.tight_layout()
        plt.savefig('data/utkface_health_analysis.png', dpi=300, bbox_inches='tight')
        print("SUCCESS: Saved UTKFace health analysis: data/utkface_health_analysis.png")
    
    def save_processed_data(self):
        """Save processed UTKFace data"""
        if self.processed_data is not None:
            output_path = 'data/utkface_processed_data.csv'
            self.processed_data.to_csv(output_path, index=False)
            print(f"SUCCESS: Saved processed UTKFace data: {output_path}")
            print(f"  Shape: {self.processed_data.shape}")
            print(f"  Features: {list(self.processed_data.columns)}")
            return output_path
        else:
            print("ERROR: No processed data to save")
            return None

def main():
    print("=== UTKFace Dataset Processor ===")
    print("Processing real UTKFace images for health prediction...")
    
    processor = SimpleUTKFaceProcessor()
    
    # Process dataset
    processed_data = processor.process_utkface_dataset(max_images=1000)
    
    if processed_data is not None:
        # Train model
        accuracy, feature_importance = processor.train_health_model()
        
        # Create visualizations
        processor.visualize_utkface_analysis()
        
        # Save data
        output_path = processor.save_processed_data()
        
        print("\n=== UTKFace Processing Complete ===")
        print("SUCCESS: Real UTKFace dataset processed for health prediction!")
        print(f"✓ Processed {len(processed_data)} real facial images")
        print(f"✓ Trained health prediction model")
        print(f"✓ Model accuracy: {accuracy:.3f}")
        print(f"✓ Real-world facial health analysis ready")
    else:
        print("\n=== UTKFace Not Available ===")
        print("Please download UTKFace dataset first:")
        print("1. Go to: https://susanqq.github.io/UTKFace/")
        print("2. Download all 3 parts")
        print("3. Extract to: data/images/utkface/")
        print("4. Run this script again")

if __name__ == "__main__":
    main()
