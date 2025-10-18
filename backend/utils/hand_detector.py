"""
Hand Detection using OpenCV and simple background subtraction
Since MediaPipe doesn't support Python 3.13, we'll use OpenCV-based hand detection
"""

import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        """Initialize hand detector with background subtractor"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect_hand(self, frame):
        """
        Detect hand region in frame using background subtraction and contour detection
        Returns: hand region (cropped image) or None
        """
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            # Remove shadows
            fg_mask[fg_mask == 127] = 0
            
            # Morphological operations to remove noise
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # Get largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Filter small contours (noise)
            if area < 5000:  # Minimum area threshold
                return None
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Extract hand region
            hand_region = frame[y:y+h, x:x+w]
            
            # Return hand region info
            return {
                'image': hand_region,
                'bbox': (x, y, w, h),
                'area': area,
                'contour': largest_contour
            }
            
        except Exception as e:
            print(f"Hand detection error: {e}")
            return None
    
    def extract_features(self, hand_region):
        """
        Extract features from hand region for classification
        Returns: feature vector
        """
        if hand_region is None:
            return None
        
        try:
            # Resize to standard size
            resized = cv2.resize(hand_region['image'], (64, 64))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Normalize
            normalized = gray.astype(np.float32) / 255.0
            
            # Flatten to feature vector
            features = normalized.flatten()
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def preprocess_for_model(self, frame):
        """
        Preprocess frame for model prediction
        Returns: processed image ready for model input
        """
        try:
            # Detect hand region
            hand_info = self.detect_hand(frame)
            
            if hand_info is None:
                # If no hand detected, use full frame
                hand_img = frame
            else:
                hand_img = hand_info['image']
            
            # Resize to model input size
            resized = cv2.resize(hand_img, (64, 64))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            batch_img = np.expand_dims(normalized, axis=0)
            
            return batch_img
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

