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
from data_utils import calculate_detection_metrics, plot_detection_results, create_feature_importance_plot, load_duc_data
from main import LeakageDetectionModel

def load_and_preprocess_real_data(file_path='real_water_network_data.csv', window_size=12):
    """
    Load and preprocess the real water network data from the combined CSV file.
    
    Args:
        file_path: Path to the combined CSV file with real data.
        window_size: Size of sliding window for feature extraction.
        
    Returns:
        Tuple containing:
        - DataFrame: The loaded and preprocessed data.
        - List: Feature names relevant for the model (excluding metadata and target).
    """
    print(f"Loading real data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Identify pressure and flow columns based on naming convention
    pressure_cols = [col for col in data.columns if col.startswith('pressure_')]
    flow_cols = [col for col in data.columns if col.startswith('flow_')]
    
    # Preprocess features (normalize pressure/flow, create time features if needed)
    # Assuming 'timestamp' column exists from the combined data creation
    data_processed = preprocess_features(data, pressure_cols, flow_cols, time_col='timestamp')
    
    # Identify feature columns for the model (excluding metadata and target)
    feature_cols = [
        col for col in data_processed.columns 
        if col not in ['leak_scenario', 'leak_flow_rate', 'pattern_coefficient', 'leak', 'timestamp']
    ]
    
    print(f"Number of features identified for model input: {len(feature_cols)}")
    
    return data_processed, feature_cols

def split_data_custom(data, feature_cols, window_size=12, leak_train_split=0.6, random_state=42):
    """
    Splits data into training and validation sets based on custom strategy:
    - Training set: All 'no leak' data + leak_train_split% of 'leak' data.
    - Validation set: Remaining (1 - leak_train_split)% of 'leak' data.
    
    Args:
        data: DataFrame with preprocessed data (including 'leak' column).
        feature_cols: List of columns to use as features.
        window_size: Sliding window size.
        leak_train_split: Proportion of leak data to use for training (0.0 to 1.0).
        random_state: Random seed for reproducibility.
        
    Returns:
        X_train, y_train, X_val, y_val, feature_names_window
    """
    # Separate no leak and leak data
    no_leak_data = data[data['leak'] == 0].copy()
    leak_data = data[data['leak'] == 1].copy()
    
    print(f"No leak data shape: {no_leak_data.shape}")
    print(f"Leak data shape: {leak_data.shape}")
    
    # Split leak data into training and validation sets
    leak_train, leak_val = train_test_split(
        leak_data, 
        train_size=leak_train_split, 
        random_state=random_state,
        shuffle=True # Shuffle leak data before splitting
    )
    
    print(f"Leak training data shape: {leak_train.shape}")
    print(f"Leak validation data shape: {leak_val.shape}")
    
    # Combine no leak data with leak training data
    train_data = pd.concat([no_leak_data, leak_train], ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True) # Shuffle combined train data
    val_data = leak_val.reset_index(drop=True)
    
    print(f"Final training data shape: {train_data.shape}")
    print(f"Final validation data shape: {val_data.shape}")
    
    # Extract features using sliding window for training data
    print("Extracting features for training set...")
    X_train_windows, _ = extract_features_for_detection(train_data[feature_cols + ['timestamp']], window_size=window_size)
    y_train = train_data['leak'].iloc[window_size-1:].values
    
    # Extract features using sliding window for validation data
    print("Extracting features for validation set...")
    X_val_windows, _ = extract_features_for_detection(val_data[feature_cols + ['timestamp']], window_size=window_size)
    y_val = val_data['leak'].iloc[window_size-1:].values
    
    # Adjust shapes if windowing caused mismatch (take labels corresponding to generated windows)
    if len(X_train_windows) != len(y_train):
        y_train = train_data['leak'].iloc[window_size-1 : window_size-1 + len(X_train_windows)].values
        print(f"Adjusted y_train shape: {y_train.shape}")
        
    if len(X_val_windows) != len(y_val):
         y_val = val_data['leak'].iloc[window_size-1 : window_size-1 + len(X_val_windows)].values
         print(f"Adjusted y_val shape: {y_val.shape}")

    assert len(X_train_windows) == len(y_train), "X_train and y_train shape mismatch after windowing."
    assert len(X_val_windows) == len(y_val), "X_val and y_val shape mismatch after windowing."

    print(f"Training features shape: {X_train_windows.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation features shape: {X_val_windows.shape}, Validation labels shape: {y_val.shape}")
    
    # Create feature names for the window features
    feature_names_window = []
    for t in range(window_size):
        for col in feature_cols:
            feature_names_window.append(f"{col}_t-{window_size-t}")
    
    return X_train_windows, y_train, X_val_windows, y_val, feature_names_window

def create_and_train_model(X_train, y_train, X_val, y_val,
                          hidden_layers=[64, 32, 16], dropout_rate=0.2,
                          epochs=100, batch_size=32):
    """
    Create and train the hybrid neural network model using separate validation data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        hidden_layers: List of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Create model
    model = LeakageDetectionModel()
    # Preprocess training data (fit scaler)
    X_train_scaled, y_train_proc = model.preprocess_data(X_train, y_train, train=True) 
    # Preprocess validation data (use fitted scaler)
    X_val_scaled, y_val_proc = model.preprocess_data(X_val, y_val, train=False)
    
    model.create_hybrid_model(input_shape=X_train_scaled.shape[1],
                             hidden_layers=hidden_layers,
                             dropout_rate=dropout_rate)
    
    # Train model using validation_data argument
    history = model.train_with_validation_data(X_train_scaled, y_train_proc, 
                                               X_val_scaled, y_val_proc,
                         epochs=epochs,
                         batch_size=batch_size)
    
    return model, history

def evaluate_model(model, X_eval, y_eval, feature_names):
    """
    Evaluate the trained model using the provided evaluation set.
    
    Args:
        model: Trained model
        X_eval: Evaluation features
        y_eval: Evaluation labels
        feature_names: List of feature names generated by windowing
        
    Returns:
        metrics: Evaluation metrics
    """
    # Evaluate model
    results, y_pred = model.evaluate(X_eval, y_eval)
    
    # Get prediction probabilities
    X_eval_scaled, _ = model.preprocess_data(X_eval, y_eval, train=False)
    y_prob = model.model.predict(X_eval_scaled).flatten()
    
    # Plot results
    timestamps = np.arange(len(y_eval))
    plot_detection_results(timestamps, y_eval, y_pred, y_prob)
    
    # Calculate additional metrics
    metrics = calculate_detection_metrics(y_eval, y_pred)
    print("\nDetection Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print("{}: {:.4f}".format(metric, value))
        else:
            print("{}: N/A".format(metric))
    
    # Create feature importance plot if model has weights
    if hasattr(model.model, 'layers') and model.model.layers:
         try:
             # Assuming the first dense layer's weights indicate importance
             first_dense_layer = next((layer for layer in model.model.layers if isinstance(layer, tf.keras.layers.Dense)), None)
             if first_dense_layer:
                 create_feature_importance_plot(first_dense_layer, feature_names)
             else:
                 print("Could not find a Dense layer to extract feature importance.")
         except Exception as e:
             print(f"Could not create feature importance plot: {e}")
    else:
         print("Model does not have layers or weights attribute for feature importance.")
    
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
    model.save(model_path="{}/model".format(output_dir))
    
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
    plt.savefig("{}/training_history.png".format(output_dir))
    
    # Save history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("{}/training_history.csv".format(output_dir))
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("{}/evaluation_metrics.csv".format(output_dir))

def plot_predicted_vs_actual(y_true, y_pred, output_file='pred_vs_actual.png'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = y_true == y_pred
    incorrect = ~correct
    plt.figure(figsize=(12, 6))
    # Actual points (all, in blue)
    plt.scatter(range(len(y_true)), y_true, label='Actual', alpha=0.5, s=15, color='blue')
    # Correct predictions (green)
    plt.scatter(np.where(correct)[0], y_pred[correct], label='Correct Prediction', alpha=0.7, s=15, color='green')
    # Incorrect predictions (orange)
    plt.scatter(np.where(incorrect)[0], y_pred[incorrect], label='Incorrect Prediction', alpha=0.7, s=15, color='orange')
    plt.xlabel('Sample Index')
    plt.ylabel('Leak Location (0 = No Leak, 1-122 = Node)')
    plt.title('Predicted vs. Actual Leak Locations')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_training_history(history, output_file='training_history.png'):
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
    plt.savefig(output_file)
    plt.close()

def run_sensitivity_analysis_case1(X_flow, X_pressure, y, metadata, n_repeats=5, percentages=None):
    if percentages is None:
        percentages = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_pipes = metadata['n_pipes']
    results = []

    for pct in percentages:
        n_selected = max(1, int(np.round(n_pipes * pct / 100)))
        accs, mrles = [], []
        print(f"\n--- Sensitivity analysis: {pct}% pipes ({n_selected} pipes) ---")
        for repeat in range(n_repeats):
            np.random.seed(repeat)
            selected_pipes = np.random.choice(n_pipes, n_selected, replace=False)
            # Prepare data for this subset
            X_flow_sub = X_flow[:, selected_pipes]
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flow_sub, y, test_size=0.2, random_state=42, stratify=y
            )
            # Train model
            model = LeakageDetectionModel(use_flow=True, use_pressure=False)
            model.create_hybrid_model(n_pipes=n_selected, n_nodes=0, n_classes=metadata['n_nodes'] + 1,
                                     hidden_layers=[128, 64], dropout_rate=0.3)
            history = model.train(X_train, None, y_train, validation_split=0.2, epochs=30, batch_size=32)
            # Evaluate
            _, y_pred, y_true = model.evaluate(X_val, None, y_val)
            acc = np.mean(y_pred == y_true)
            # MRLE calculation (for leak cases only)
            leak_mask = y_true > 0
            if np.any(leak_mask):
                mrle = np.max(np.abs(y_pred[leak_mask] - y_true[leak_mask]) / (np.max(y_true[leak_mask]))) * 100
            else:
                mrle = np.nan
            accs.append(acc)
            mrles.append(mrle)
        # Store average results
        results.append({
            'percentage': pct,
            'n_pipes': n_selected,
            'accuracy_mean': np.mean(accs),
            'accuracy_std': np.std(accs),
            'mrle_mean': np.nanmean(mrles),
            'mrle_std': np.nanstd(mrles)
        })
        print(f"  Mean accuracy: {np.mean(accs):.4f}, Mean MRLE: {np.nanmean(mrles):.2f}%")
    # Plot
    percentages = [r['percentage'] for r in results]
    acc_means = [r['accuracy_mean'] for r in results]
    acc_stds = [r['accuracy_std'] for r in results]
    mrle_means = [r['mrle_mean'] for r in results]
    mrle_stds = [r['mrle_std'] for r in results]
    plt.figure(figsize=(10, 5))
    plt.errorbar(percentages, acc_means, yerr=acc_stds, fmt='-o', label='Accuracy')
    plt.xlabel('% of Pipes Used')
    plt.ylabel('Accuracy')
    plt.title('Sensitivity Analysis: Accuracy vs. % of Pipes Used')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sensitivity_accuracy.png')
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.errorbar(percentages, mrle_means, yerr=mrle_stds, fmt='-o', color='red', label='MRLE')
    plt.xlabel('% of Pipes Used')
    plt.ylabel('MRLE (%)')
    plt.title('Sensitivity Analysis: MRLE vs. % of Pipes Used')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sensitivity_mrle.png')
    plt.close()
    # Save results as CSV
    pd.DataFrame(results).to_csv('sensitivity_results.csv', index=False)
    print('\nSensitivity analysis completed. Results saved to sensitivity_accuracy.png, sensitivity_mrle.png, and sensitivity_results.csv.')

def main():
    print("Leakage Detection in Water Distribution Networks")
    print("Training and Evaluation using Real Data")
    print("-" * 60)
    
    # Load real data
    print("\nLoading real data...")
    X_flow, X_pressure, y, metadata = load_duc_data()
    
    print(f"Loaded data:")
    print(f"- Number of pipes: {metadata['n_pipes']}")
    print(f"- Number of nodes: {metadata['n_nodes']}")
    print(f"- Number of samples: {len(y)}")
    print(f"- Number of leak scenarios: {np.sum(y > 0)}")
    print(f"- Number of no-leak scenarios: {np.sum(y == 0)}")
    
    # Split data into train/validation sets
    # We'll use stratified split to ensure balanced classes
    X_flow_train, X_flow_val, X_pressure_train, X_pressure_val, y_train, y_val = train_test_split(
        X_flow, X_pressure, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Create and train model
    print("\nCreating and training model...")
    model = LeakageDetectionModel(use_flow=True, use_pressure=True)
    model.create_hybrid_model(
        n_pipes=metadata['n_pipes'],
        n_nodes=metadata['n_nodes'],
        n_classes=metadata['n_nodes'] + 1,  # nodes + no leak
        hidden_layers=[256, 128, 64],
        dropout_rate=0.3
    )
    
    # Train model
    history = model.train(
        X_flow_train, X_pressure_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    results, y_pred, y_true = model.evaluate(X_flow_val, X_pressure_val, y_val)
    
    # Visualize predicted vs. actual leak locations
    plot_predicted_vs_actual(y_true, y_pred, output_file='pred_vs_actual.png')
    
    # Save model
    print("\nSaving model...")
    model.save()
    
    # Sensitivity analysis for Case 1
    print("\nRunning sensitivity analysis for Case 1...")
    run_sensitivity_analysis_case1(X_flow, X_pressure, y, metadata, n_repeats=3)
    
    print("\nTraining completed.")

if __name__ == "__main__":
    main() 