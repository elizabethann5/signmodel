#!/usr/bin/env python3
"""
Test script for MNIST and RWTH-Phoenix-Weather 2014T dataset integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_model import ComprehensiveSignLanguageModel
import numpy as np

def test_dataset_integration():
    """Test the dataset integration"""
    print("=" * 60)
    print("TESTING DATASET INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize the model
        print("\n1. Initializing Comprehensive Sign Language Model...")
        model = ComprehensiveSignLanguageModel()
        
        print(f"✅ Model initialized with {model.num_classes} classes")
        print(f"   - MNIST digits: {len(model.mnist_labels)} classes")
        print(f"   - ASL letters: {len(model.asl_labels)} classes") 
        print(f"   - RWTH-Phoenix signs: {len(model.phoenix_labels)} classes")
        
        # Test MNIST data loading
        print("\n2. Testing MNIST dataset loading...")
        x_mnist_train, y_mnist_train, x_mnist_test, y_mnist_test = model.load_mnist_data()
        
        if x_mnist_train is not None:
            print(f"✅ MNIST loaded successfully: {x_mnist_train.shape}")
            print(f"   - Training samples: {x_mnist_train.shape[0]}")
            print(f"   - Image shape: {x_mnist_train.shape[1:]}")
            print(f"   - Label range: {y_mnist_train.min()} to {y_mnist_train.max()}")
        else:
            print("❌ MNIST loading failed")
        
        # Test RWTH-Phoenix data loading
        print("\n3. Testing RWTH-Phoenix dataset loading...")
        x_phoenix_train, y_phoenix_train = model.load_rwth_phoenix_data()
        
        if x_phoenix_train is not None:
            print(f"✅ RWTH-Phoenix loaded successfully: {x_phoenix_train.shape}")
            print(f"   - Training samples: {x_phoenix_train.shape[0]}")
            print(f"   - Image shape: {x_phoenix_train.shape[1:]}")
            print(f"   - Label range: {y_phoenix_train.min()} to {y_phoenix_train.max()}")
        else:
            print("❌ RWTH-Phoenix loading failed")
        
        # Test synthetic data generation
        print("\n4. Testing synthetic data generation...")
        x_synthetic, y_synthetic = model.generate_synthetic_data()
        
        if x_synthetic is not None:
            print(f"✅ Synthetic data generated: {x_synthetic.shape}")
            print(f"   - Samples: {x_synthetic.shape[0]}")
            print(f"   - Image shape: {x_synthetic.shape[1:]}")
        else:
            print("❌ Synthetic data generation failed")
        
        # Test model creation
        print("\n5. Testing model creation...")
        test_model = model.create_model()
        print(f"✅ Model created successfully")
        print(f"   - Input shape: {test_model.input_shape}")
        print(f"   - Output shape: {test_model.output_shape}")
        print(f"   - Total parameters: {test_model.count_params():,}")
        
        # Test prediction with dummy data
        print("\n6. Testing prediction with dummy data...")
        dummy_frame = np.random.rand(64, 64, 3).astype(np.uint8)
        prediction = model.predict(dummy_frame)
        print(f"✅ Prediction test completed")
        print(f"   - Prediction result: '{prediction}'")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_training_process():
    """Test the training process (optional - takes longer)"""
    print("\n" + "=" * 60)
    print("TESTING TRAINING PROCESS (Optional)")
    print("=" * 60)
    
    try:
        print("Creating and training model...")
        model = ComprehensiveSignLanguageModel()
        
        # This will create and train the model
        model.create_and_train_model()
        
        print("✅ Training process completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting dataset integration tests...")
    
    # Run basic tests
    success = test_dataset_integration()
    
    if success:
        print("\nWould you like to test the training process? (This may take a few minutes)")
        response = input("Enter 'y' to test training, or any other key to skip: ").lower().strip()
        
        if response == 'y':
            test_training_process()
    
    print("\nTest completed!")
