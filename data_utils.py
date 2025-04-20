#!/usr/bin/env python3
# Utilities for data processing in leakage detection

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_pressure_flow_data(file_path):
    """
    Load pressure and flow data from CSV file
    Expected format: timestamps as rows, sensor readings as columns
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with pressure and flow data
    """
    try:
        data = pd.read_csv(file_path)
        print("Loaded data with shape: {}".format(data.shape))
        return data
    except Exception as e:
        print("Error loading data: {}".format(e))
        return None

def preprocess_features(data, pressure_cols=None, flow_cols=None, time_col=None):
    """
    Preprocess features according to the paper's methodology:
    - Normalize pressure readings
    - Normalize flow rates
    - Create cyclical features for time (if available)
    
    Args:
        data: DataFrame with raw sensor data
        pressure_cols: List of column names containing pressure readings
        flow_cols: List of column names containing flow readings
        time_col: Column name containing time information
        
    Returns:
        DataFrame with preprocessed features
    """
    # Create a copy to avoid modifying the original data
    processed = data.copy()
    
    # If columns are not specified, try to infer based on naming conventions
    if pressure_cols is None:
        pressure_cols = [col for col in data.columns if 'pressure' in col.lower() or 'p_' in col.lower()]
    if flow_cols is None:
        flow_cols = [col for col in data.columns if 'flow' in col.lower() or 'f_' in col.lower()]
    
    # Normalize pressure data
    if pressure_cols:
        pressure_scaler = MinMaxScaler()
        processed[pressure_cols] = pressure_scaler.fit_transform(processed[pressure_cols])
    
    # Normalize flow data
    if flow_cols:
        flow_scaler = MinMaxScaler()
        processed[flow_cols] = flow_scaler.fit_transform(processed[flow_cols])
    
    # Process time data (if available) to create cyclical features
    if time_col is not None and time_col in data.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            processed[time_col] = pd.to_datetime(data[time_col])
        
        # Extract hour and create cyclical features (sin and cos transformations)
        hour = processed[time_col].dt.hour
        processed['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        processed['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Extract day of week and create cyclical features
        day = processed[time_col].dt.dayofweek
        processed['day_sin'] = np.sin(2 * np.pi * day / 7)
        processed['day_cos'] = np.cos(2 * np.pi * day / 7)
        
    return processed

def extract_features_for_detection(data, window_size=24):
    """
    Extract features for leak detection based on sliding window
    The paper suggests using temporal patterns for detection
    
    Args:
        data: DataFrame with preprocessed features
        window_size: Size of the sliding window (in time steps)
        
    Returns:
        X: Feature matrix with sliding window patterns
        timestamps: Corresponding timestamps (if available)
    """
    # Get column names excluding any target or timestamp columns
    feature_cols = [col for col in data.columns if col not in ['leak', 'timestamp', 'date', 'time']]
    
    # Initialize lists to store features and timestamps
    X_windows = []
    timestamps = []
    
    # Create sliding windows
    for i in range(len(data) - window_size + 1):
        window = data[feature_cols].iloc[i:i+window_size].values.flatten()
        X_windows.append(window)
        
        # If timestamps are available, use the last timestamp in the window
        if 'timestamp' in data.columns:
            timestamps.append(data['timestamp'].iloc[i+window_size-1])
    
    return np.array(X_windows), timestamps

def calculate_detection_metrics(y_true, y_pred):
    """
    Calculate leakage detection metrics as described in the paper
    
    Args:
        y_true: True labels (1 for leak, 0 for no leak)
        y_pred: Predicted labels
        
    Returns:
        metrics: Dictionary with detection metrics
    """
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics from the paper
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    detection_delay = None  # This would need time-series analysis
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_alarm_rate': false_alarm_rate,
        'detection_delay': detection_delay
    }
    
    return metrics

def plot_detection_results(timestamps, y_true, y_pred, y_prob=None):
    """
    Plot the results of leak detection
    
    Args:
        timestamps: Array of timestamps
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
    """
    plt.figure(figsize=(15, 7))
    
    # If timestamps are not provided, use indices
    if timestamps is None:
        timestamps = np.arange(len(y_true))
    
    # Plot true leaks
    plt.scatter(timestamps, y_true, label='Actual Leaks', color='blue', alpha=0.5, s=50, marker='o')
    
    # Plot predictions
    plt.scatter(timestamps, y_pred, label='Predicted Leaks', color='red', alpha=0.5, s=30, marker='x')
    
    # If probabilities are available, plot them
    if y_prob is not None:
        plt.plot(timestamps, y_prob, label='Leak Probability', color='green', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Leak Status / Probability')
    plt.title('Leakage Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('detection_results.png')
    
def create_feature_importance_plot(model, feature_names):
    """
    Create a plot showing the importance of each feature
    
    Args:
        model: Trained model
        feature_names: List of feature names
    """
    # This implementation depends on the model type and might need to be adjusted
    try:
        # For neural networks, we'll use the weights of the first dense layer as a proxy
        if hasattr(model, 'layers'):
            weights = model.layers[0].get_weights()[0]
            
            # Handle case where dimensions don't match
            if len(weights) != len(feature_names):
                print("Warning: Model weights shape ({}) doesn't match feature names shape ({})".format(
                    weights.shape[0], len(feature_names)))
                
                # Use only the number of features we have names for, or truncate feature names
                n_features = min(len(weights), len(feature_names))
                weights = weights[:n_features]
                feature_names = feature_names[:n_features]
            
            importances = np.mean(np.abs(weights), axis=1)
            
            # Sort features by importance
            indices = np.argsort(importances)
            sorted_feature_names = [feature_names[i] for i in indices[-20:]]  # Top 20 features
            sorted_importances = importances[indices[-20:]]  # Top 20 importances
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.barh(sorted_feature_names, sorted_importances)
            plt.xlabel('Mean Absolute Weight')
            plt.ylabel('Feature')
            plt.title('Top 20 Features Importance Based on First Layer Weights')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
    except Exception as e:
        print("Could not create feature importance plot: {}".format(e)) 