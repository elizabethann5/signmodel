# Model Training Scripts

This folder contains scripts for training ASL sign language recognition models.

## Available Scripts

### `train_asl_model.py`
- **Description**: Full training pipeline with OpenCV data augmentation
- **Dataset**: Synthetic ASL data (A-Z letters)
- **Status**: ⚠️ Not functional (OpenCV compatibility issues in Python 3.13)
- **Features**: Data augmentation, CNN training, model evaluation

### `train_asl_model_no_cv.py`
- **Description**: Training without OpenCV dependency
- **Dataset**: Pure NumPy synthetic data
- **Status**: ⚠️ Not functional (TensorFlow internally uses OpenCV)
- **Features**: Pattern-based data generation, model training

### `test_datasets.py`
- **Description**: Test script for dataset loading
- **Purpose**: Verify MNIST and RWTH-Phoenix dataset integration
- **Status**: Testing/debugging tool

### `dataset_manager.py`
- **Description**: Dataset management utilities
- **Purpose**: Download, prepare, and manage training datasets
- **Features**: MNIST loading, RWTH-Phoenix preparation

## Training Process

### Prerequisites
```bash
# Requires compatible environment
Python 3.11 (recommended, not 3.13)
pip install tensorflow opencv-python numpy
```

### Running Training
```bash
cd backend/training
python train_asl_model_no_cv.py
```

### Expected Output
- Trained model: `asl_model.h5`
- Model info: `asl_model_info.txt`
- Training metrics in console

## Current Model

The existing model at `backend/models/sign_language_model.h5` includes:
- **Size**: 31MB
- **Classes**: 63 total (digits 0-9, letters A-Z, German signs)
- **Input**: 64x64 RGB images
- **Architecture**: CNN with 3 conv blocks + dense layers

## Known Issues

1. **Python 3.13 Compatibility**: NumPy 2.x causes issues with OpenCV and TensorFlow
2. **Solution**: Use Python 3.11 or wait for library updates
3. **Workaround**: Use existing trained model with current system

## Training Your Own Model

### Step 1: Fix Environment
```bash
# Install Python 3.11
pyenv install 3.11
pyenv local 3.11

# Create new venv
python -m venv venv_training
source venv_training/bin/activate
pip install tensorflow==2.15.0 opencv-python numpy<2
```

### Step 2: Prepare Data
```bash
# Use real ASL dataset (recommended)
# Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

# Or use synthetic data (quick test)
python train_asl_model_no_cv.py
```

### Step 3: Train Model
```bash
python train_asl_model.py --epochs 20 --batch_size 32
```

### Step 4: Deploy Model
```bash
cp asl_model.h5 ../models/sign_language_model.h5
```

## Model Architecture

```
Input (64, 64, 3)
    ↓
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.3)
    ↓
Dense(num_classes) → Softmax
```

## Future Improvements

- [ ] Real ASL dataset integration
- [ ] MediaPipe hand landmarks as features
- [ ] Transfer learning from pre-trained models
- [ ] Data augmentation improvements
- [ ] Ensemble models for better accuracy

Last Updated: October 19, 2025

