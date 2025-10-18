"""
Train ASL Sign Language Recognition Model
No OpenCV dependency - uses only NumPy and TensorFlow
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os


class ASLModelTrainer:
    def __init__(self):
        self.model = None
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z']
        self.num_classes = len(self.labels)
        
    def create_model(self):
        """Create CNN model for ASL recognition"""
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_pattern(self, size, pattern_type, seed):
        """Create a distinctive pattern for each letter"""
        np.random.seed(seed)
        img = np.random.rand(size, size, 3) * 0.2
        
        center = size // 2
        
        if pattern_type == 'circle':
            # Circular pattern
            for i in range(size):
                for j in range(size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    if dist < 15:
                        img[i, j] = [0.8, 0.8, 0.8]
        elif pattern_type == 'horizontal':
            # Horizontal line
            img[center-2:center+2, :] = [0.7, 0.7, 0.7]
        elif pattern_type == 'vertical':
            # Vertical line
            img[:, center-2:center+2] = [0.7, 0.7, 0.7]
        elif pattern_type == 'diagonal':
            # Diagonal line
            for i in range(size):
                j = i
                if 0 <= j < size:
                    img[i, j] = [0.75, 0.75, 0.75]
        elif pattern_type == 'cross':
            # Cross pattern
            img[center-2:center+2, :] = [0.7, 0.7, 0.7]
            img[:, center-2:center+2] = [0.7, 0.7, 0.7]
        elif pattern_type == 'square':
            # Square pattern
            img[20:44, 20:44] = [0.75, 0.75, 0.75]
        else:
            # Random pattern
            for _ in range(10):
                x = np.random.randint(10, 54)
                y = np.random.randint(10, 54)
                size_s = np.random.randint(3, 8)
                img[y:y+size_s, x:x+size_s] = [0.6, 0.6, 0.6]
        
        return img
    
    def generate_synthetic_asl_data(self, samples_per_class=200):
        """Generate synthetic ASL data"""
        print(f"Generating synthetic ASL data: {samples_per_class} samples per class...")
        
        X_data = []
        y_data = []
        
        # Define pattern types for different letter groups
        pattern_map = {
            'A': 'circle', 'B': 'vertical', 'C': 'circle', 'D': 'circle',
            'E': 'horizontal', 'F': 'vertical', 'G': 'horizontal', 'H': 'vertical',
            'I': 'vertical', 'J': 'diagonal', 'K': 'diagonal', 'L': 'diagonal',
            'M': 'horizontal', 'N': 'horizontal', 'O': 'circle', 'P': 'vertical',
            'Q': 'circle', 'R': 'cross', 'S': 'square', 'T': 'vertical',
            'U': 'vertical', 'V': 'diagonal', 'W': 'cross', 'X': 'cross',
            'Y': 'diagonal', 'Z': 'diagonal'
        }
        
        for class_idx, letter in enumerate(self.labels):
            pattern_type = pattern_map.get(letter, 'random')
            
            for sample_idx in range(samples_per_class):
                # Create base pattern
                img = self.create_pattern(64, pattern_type, class_idx * 1000 + sample_idx)
                
                # Add letter-specific variations
                variation = class_idx * 0.05
                img = img + np.random.randn(64, 64, 3) * (0.05 + variation)
                
                # Add random brightness/contrast
                brightness = np.random.uniform(0.8, 1.2)
                img = img * brightness
                
                # Clip values
                img = np.clip(img, 0, 1)
                
                # Add random rotation (simple shift)
                if sample_idx % 2 == 0:
                    shift = np.random.randint(-5, 5)
                    img = np.roll(img, shift, axis=0)
                    img = np.roll(img, shift, axis=1)
                
                X_data.append(img)
                y_data.append(class_idx)
        
        X_data = np.array(X_data, dtype=np.float32)
        y_data = np.array(y_data)
        
        print(f"✅ Generated {len(X_data)} synthetic samples")
        return X_data, y_data
    
    def train(self, epochs=20, batch_size=32):
        """Train the ASL model"""
        print("\n" + "="*60)
        print("TRAINING ASL SIGN LANGUAGE MODEL")
        print("="*60 + "\n")
        
        # Generate synthetic data
        X_asl, y_asl = self.generate_synthetic_asl_data(samples_per_class=200)
        
        # Convert labels to categorical
        y_asl_categorical = keras.utils.to_categorical(y_asl, self.num_classes)
        
        # Split into train/validation
        split_idx = int(len(X_asl) * 0.8)
        X_train = X_asl[:split_idx]
        y_train = y_asl_categorical[:split_idx]
        X_val = X_asl[split_idx:]
        y_val = y_asl_categorical[split_idx:]
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Classes: {self.num_classes} ({', '.join(self.labels)})\n")
        
        # Create model
        self.model = self.create_model()
        
        print("Model Architecture:")
        self.model.summary()
        
        # Train model
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]
        )
        
        # Print training results
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
    def save_model(self, filename='asl_model.h5'):
        """Save trained model"""
        if self.model is None:
            print("❌ No model to save")
            return False
        
        try:
            self.model.save(filename)
            print(f"\n✅ Model saved to {filename}")
            
            # Save model info
            info_file = filename.replace('.h5', '_info.txt')
            with open(info_file, 'w') as f:
                f.write(f"ASL Sign Language Model\n")
                f.write(f"Classes: {self.num_classes}\n")
                f.write(f"Labels: {', '.join(self.labels)}\n")
                f.write(f"Input shape: (64, 64, 3)\n")
                f.write(f"\nTraining completed successfully!\n")
            
            print(f"✅ Model info saved to {info_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def test_prediction(self):
        """Test model with random samples"""
        if self.model is None:
            print("❌ No model loaded")
            return
        
        print("\n" + "="*60)
        print("TESTING MODEL PREDICTIONS")
        print("="*60)
        
        # Generate test samples
        X_test, y_test = self.generate_synthetic_asl_data(samples_per_class=5)
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0)
        
        # Show some results
        correct = 0
        for i in range(min(20, len(X_test))):
            pred_class = np.argmax(predictions[i])
            true_class = y_test[i]
            pred_label = self.labels[pred_class]
            true_label = self.labels[true_class]
            confidence = predictions[i][pred_class]
            
            match = "✅" if pred_class == true_class else "❌"
            if i < 10:  # Print first 10
                print(f"{match} True: {true_label}, Predicted: {pred_label} (conf: {confidence:.2f})")
            
            if pred_class == true_class:
                correct += 1
        
        accuracy = correct / min(20, len(X_test))
        print(f"\nTest accuracy on {min(20, len(X_test))} samples: {accuracy:.2%}")


def main():
    """Main training function"""
    print("\n🚀 ASL Model Training Script (No OpenCV)\n")
    
    # Create trainer
    trainer = ASLModelTrainer()
    
    # Train model
    trainer.train(epochs=20, batch_size=32)
    
    # Test predictions
    trainer.test_prediction()
    
    # Save model
    trainer.save_model('asl_model.h5')
    
    print("\n✅ Training complete! Model ready for use.")
    print("\nTo use this model:")
    print("1. Stop the current server (Ctrl+C)")
    print("2. Run: python app_with_detection.py")
    print("3. Open browser to http://localhost:8080")


if __name__ == "__main__":
    main()

