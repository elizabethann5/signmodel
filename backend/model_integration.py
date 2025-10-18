import cv2
import numpy as np
import os
from ai_model import ComprehensiveSignLanguageModel

class SignLanguagePredictor:
    def __init__(self, model_path=None):
        """
        Initialize the sign language predictor with comprehensive AI model.
        """
        self.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'del', 'nothing', 'space'
        ]
        
        # Initialize the comprehensive AI model
        try:
            self.ai_model = ComprehensiveSignLanguageModel(model_path)
            self.use_ai_model = True
            print("✅ Comprehensive AI model initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing AI model: {e}")
            print("Falling back to mock mode")
            self.use_ai_model = False
            self.ai_model = None
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for the model"""
        try:
            # Resize frame to standard size
            resized = cv2.resize(frame, (64, 64))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = rgb_frame.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def predict(self, frame):
        """Predict sign language from frame using comprehensive AI model"""
        try:
            # Basic motion detection - check if there's significant change in the frame
            if not self.detect_motion(frame):
                return ""  # No motion detected, no prediction
            
            # Use AI model if available
            if self.use_ai_model and self.ai_model:
                prediction = self.ai_model.predict(frame, confidence_threshold=0.6)
                if prediction:
                    return prediction
                else:
                    return ""
            else:
                # Fallback to mock prediction
                return self.mock_prediction()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return ""
    
    def detect_motion(self, frame):
        """Simple motion detection - returns True if motion is detected"""
        try:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame variance - higher variance indicates more motion
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Threshold for motion detection (lower = more sensitive)
            motion_threshold = 50
            
            # Debug: Print motion detection info
            print(f"Motion variance: {variance:.2f}, threshold: {motion_threshold}")
            
            # Return True if there's enough motion
            return variance > motion_threshold
            
        except Exception as e:
            print(f"Motion detection error: {e}")
            return False
    
    def mock_prediction(self):
        """Return mock prediction for testing - only when motion is detected"""
        import random
        
        # Simulate more realistic behavior - only return predictions occasionally
        # and only when there might be actual hand movement
        if random.random() < 0.05:  # Only 5% chance of returning something
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            return random.choice(letters)
        
        # Most of the time, return empty string (no sign detected)
        return ""