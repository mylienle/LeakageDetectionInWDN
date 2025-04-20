#!/usr/bin/env python3
# Train and evaluate the leakage detection model

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_utils import load_pressure_flow_data, preprocess_features, extract_features_for_detection
from data_utils import calculate_detection_metrics, plot_detection_results, create_feature_importance_plot
from main import LeakageDetectionModel

def load_and_preprocess_data(file_path, window_size=24):
    """
    Load and preprocess data for training
    
    Args:
        file_path: Path to the CSV file with water network data
        window_size: Size of sliding window for feature extraction
        
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    """
    # Load data
    data = load_pressure_flow_data(file_path)
    if data is None:
        raise ValueError(f"Failed to load data from {file_path}")
    
    # Preprocess features
    pressure_cols = [col for col in data.columns if 'pressure' in col]
    flow_cols = [col for col in data.columns if 'flow' in col]
    data_processed = preprocess_features(data, pressure_cols, flow_cols, time_col='timestamp')
    
    # Extract features with sliding window
    X_windows, _ = extract_features_for_detection(data_processed, window_size=window_size)
    
    # Target variable (leak status)
    # Use the leak status of the last time step in each window
    y = data['leak'].iloc[window_size-1:].values
    
    # Create feature names for the window features
    feature_names = []
    for t in range(window_size):
        for col in data_processed.columns:
            if col not in ['leak', 'timestamp', 'hour', 'day_of_week']:
                feature_names.append(f"{col}_t-{window_size-t}")
    
    return X_windows, y, feature_names

def create_and_train_model(X_train, y_train, hidden_layers=[64, 32, 16], dropout_rate=0.2,
                          epochs=100, batch_size=32, validation_split=0.2):
    """
    Create and train the hybrid neural network model
    
    Args:
        X_train: Training features
        y_train: Training labels
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Proportion of training data to use for validation
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Create model
    model = LeakageDetectionModel()
    model.create_hybrid_model(input_shape=X_train.shape[1],
                             hidden_layers=hidden_layers,
                             dropout_rate=dropout_rate)
    
    # Train model
    history = model.train(X_train, y_train,
                         validation_split=validation_split,
                         epochs=epochs,
                         batch_size=batch_size)
    
    return model, history

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        
    Returns:
        metrics: Evaluation metrics
    """
    # Evaluate model
    results, y_pred = model.evaluate(X_test, y_test)
    
    # Get prediction probabilities
    X_test_scaled, _ = model.preprocess_data(X_test, y_test, train=False)
    y_prob = model.model.predict(X_test_scaled).flatten()
    
    # Plot results
    timestamps = np.arange(len(y_test))
    plot_detection_results(timestamps, y_test, y_pred, y_prob)
    
    # Calculate additional metrics
    metrics = calculate_detection_metrics(y_test, y_pred)
    print("\nDetection Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create feature importance plot
    create_feature_importance_plot(model.model, feature_names)
    
    return metrics

def save_results(model, history, metrics, output_dir='results'):
    """
    Save model and results
    
    Args:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
        output_dir: Directory to save results
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model.save(model_path=f"{output_dir}/model")
    
    # Save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f"{output_dir}/training_history.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv")

def main():
    print("Leakage Detection in Water Distribution Networks")
    print("Training and Evaluation")
    print("-" * 60)
    
    # Set file paths
    data_file = 'synthetic_water_network_data.csv'
    
    # Check if data file exists, if not generate it
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Generating synthetic data...")
        from generate_data import generate_water_network_data, plot_data
        
        # Define leak periods
        leak_periods = [
            (5, 7),    # 2-day leak in the first week
            (15, 17),  # 2-day leak in the third week
            (25, 28)   # 3-day leak in the fourth week
        ]
        
        # Generate data
        data = generate_water_network_data(
            n_days=30,
            sampling_rate_mins=15,
            leak_periods=leak_periods,
            output_file=data_file
        )
        
        # Plot generated data
        plot_data(data)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data(data_file, window_size=12)
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Create and train model
    print("\nCreating and training model...")
    model, history = create_and_train_model(
        X_train, y_train,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    
    # Save results
    print("\nSaving results...")
    save_results(model, history, metrics)
    
    print("\nTraining and evaluation completed.")

if __name__ == "__main__":
    main() 