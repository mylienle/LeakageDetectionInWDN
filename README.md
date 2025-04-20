# Leakage Detection in Water Distribution Networks

This project implements the methodology described in the paper **"Leakage detection in water distribution networks using hybrid feedforward artificial neural networks"** published in the Journal of Water Supply: Research and Technology-Aqua.

## Overview

Water leakage is a significant issue in water distribution systems, leading to water loss, energy waste, and potential infrastructure damage. This project implements a hybrid feedforward artificial neural network (ANN) approach to detect leaks in water distribution networks based on pressure and flow measurements.

## Features

- Synthetic data generation for water distribution networks
- Feature preprocessing and extraction
- Hybrid feedforward neural network model for leak detection
- Model evaluation and performance metrics
- Visualization of results

## Requirements

- Python 3.6+
- TensorFlow 2.6.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Core implementation of the leakage detection model
- `data_utils.py`: Utilities for data processing and features extraction
- `generate_data.py`: Synthetic data generation for water distribution networks
- `train_model.py`: Script to train and evaluate the model

## How to Use

1. Generate synthetic data:

```bash
python generate_data.py
```

This will create a CSV file with synthetic water network data and save plots in the `plots` directory.

2. Train and evaluate the model:

```bash
python train_model.py
```

This will:
- Load the generated data (or create it if not found)
- Preprocess the data and extract features
- Train the hybrid neural network model
- Evaluate the model and generate performance metrics
- Save the model and results in the `results` directory

## Model Architecture

The hybrid feedforward neural network consists of:

- Input layer: Accepts time-series data of pressure and flow measurements
- Hidden layers: Multiple dense layers with ReLU activation
- Dropout layers for regularization
- Output layer: Single neuron with sigmoid activation for binary classification

## Results

The model performance is evaluated using:
- Accuracy, precision, recall, and F1 score
- False alarm rate
- Confusion matrix
- Classification report

Visual outputs include:
- Pressure and flow readings with leak periods highlighted
- Model training history (accuracy and loss)
- Feature importance
- Detection results

## Implementation Details

The implementation follows these key steps:

1. **Data preprocessing**:
   - Normalization of pressure and flow readings
   - Feature extraction using sliding window approach
   - Handling temporal patterns

2. **Model training**:
   - Hybrid neural network with dropout regularization
   - Binary classification (leak or no leak)
   - Optimization using Adam optimizer

3. **Performance evaluation**:
   - Standard classification metrics
   - Visual inspection of model predictions

## Extending the Project

To use with real water distribution network data:
1. Replace the synthetic data with real measurements
2. Adjust the feature extraction parameters based on the sampling rate
3. Optimize model hyperparameters for your specific dataset

## Attribution

This project is an implementation of the methodology described in:
"Leakage detection in water distribution networks using hybrid feedforward artificial neural networks"
Journal of Water Supply: Research and Technology-Aqua
[https://iwaponline.com/aqua/article/70/5/637/82207/Leakage-detection-in-water-distribution-networks](https://iwaponline.com/aqua/article/70/5/637/82207/Leakage-detection-in-water-distribution-networks)

## License

MIT 