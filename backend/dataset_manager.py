#!/usr/bin/env python3
"""
Dataset Management Script for MNIST and RWTH-Phoenix-Weather 2014T
"""

import os
import sys
import argparse
from ai_model import ComprehensiveSignLanguageModel

def download_datasets():
    """Download and prepare datasets"""
    print("=" * 60)
    print("DATASET DOWNLOAD AND PREPARATION")
    print("=" * 60)
    
    model = ComprehensiveSignLanguageModel()
    
    print("\n1. MNIST Dataset:")
    print("   - Automatically downloaded by TensorFlow/Keras")
    print("   - No additional setup required")
    
    print("\n2. RWTH-Phoenix-Weather 2014T Dataset:")
    print("   - Manual download required")
    print("   - Visit: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/")
    print("   - For now, using synthetic data for demonstration")
    
    # Test dataset loading
    print("\n3. Testing dataset loading...")
    
    # Test MNIST
    x_mnist, y_mnist, _, _ = model.load_mnist_data()
    if x_mnist is not None:
        print(f"   ✅ MNIST: {x_mnist.shape[0]} samples loaded")
    else:
        print("   ❌ MNIST loading failed")
    
    # Test RWTH-Phoenix
    x_phoenix, y_phoenix = model.load_rwth_phoenix_data()
    if x_phoenix is not None:
        print(f"   ✅ RWTH-Phoenix: {x_phoenix.shape[0]} samples loaded")
    else:
        print("   ❌ RWTH-Phoenix loading failed")
    
    print("\n✅ Dataset preparation completed!")

def train_model():
    """Train the model with all datasets"""
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    model = ComprehensiveSignLanguageModel()
    model.create_and_train_model()
    
    print("✅ Model training completed!")

def evaluate_model():
    """Evaluate the trained model"""
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    model = ComprehensiveSignLanguageModel()
    
    if model.model is not None:
        print("✅ Model loaded successfully")
        print(f"   - Classes: {model.num_classes}")
        print(f"   - Model path: {model.model_path}")
        
        # Test with dummy data
        dummy_frame = np.random.rand(64, 64, 3).astype(np.uint8)
        prediction = model.predict(dummy_frame)
        print(f"   - Test prediction: '{prediction}'")
    else:
        print("❌ No trained model found")

def main():
    parser = argparse.ArgumentParser(description='Dataset Management for Sign Language Model')
    parser.add_argument('action', choices=['download', 'train', 'evaluate', 'all'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        download_datasets()
    elif args.action == 'train':
        train_model()
    elif args.action == 'evaluate':
        evaluate_model()
    elif args.action == 'all':
        download_datasets()
        train_model()
        evaluate_model()

if __name__ == "__main__":
    main()
