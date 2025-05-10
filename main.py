#!/usr/bin/env python3
# Implementation of "Leakage detection in water distribution networks using hybrid feedforward artificial neural networks"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LeakageDetectionModel:
    def __init__(self, use_flow=True, use_pressure=True):
        self.model = None
        self.flow_scaler = None
        self.pressure_scaler = None
        self.use_flow = use_flow
        self.use_pressure = use_pressure
        
    def create_hybrid_model(self, n_pipes=198, n_nodes=122, n_classes=123, hidden_layers=[256, 128, 64], dropout_rate=0.2):
        """
        Create a hybrid feedforward neural network for leakage detection
        
        Args:
            n_pipes: Number of pipes (flow features)
            n_nodes: Number of nodes (pressure features)
            n_classes: Number of output classes (122 nodes + 1 for no leak)
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        model = tf.keras.Sequential()
        
        # Input layer size depends on which features we use
        input_size = 0
        if self.use_flow:
            input_size += n_pipes
        if self.use_pressure:
            input_size += n_nodes
            
        model.add(tf.keras.layers.Input(shape=(input_size,)))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rate))
            
        # Output layer - multi-class classification (123 classes: 122 nodes + no leak)
        model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X_flow=None, X_pressure=None, y=None, train=True):
        """
        Preprocess flow and/or pressure data using standardization
        """
        features = []
        
        if self.use_flow and X_flow is not None:
            if train:
                self.flow_scaler = StandardScaler()
                X_flow_scaled = self.flow_scaler.fit_transform(X_flow)
            else:
                X_flow_scaled = self.flow_scaler.transform(X_flow)
            features.append(X_flow_scaled)
            
        if self.use_pressure and X_pressure is not None:
            if train:
                self.pressure_scaler = StandardScaler()
                X_pressure_scaled = self.pressure_scaler.fit_transform(X_pressure)
            else:
                X_pressure_scaled = self.pressure_scaler.transform(X_pressure)
            features.append(X_pressure_scaled)
            
        if len(features) > 0:
            X = np.hstack(features)
            return X, y
        else:
            raise ValueError("No features provided for preprocessing")
    
    def train(self, X_flow=None, X_pressure=None, y=None, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the hybrid neural network model
        """
        # Preprocess data
        X, y = self.preprocess_data(X_flow, X_pressure, y, train=True)
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def train_with_validation_data(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the hybrid neural network model using separate validation data.
        
        Args:
            X_train: Preprocessed training features
            y_train: Training labels
            X_val: Preprocessed validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Train model using validation_data argument
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val), # Pass validation data here
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_flow=None, X_pressure=None, y_true=None):
        """
        Evaluate the model on test data, compute MRLE and RMSE, and return predictions
        """
        # Preprocess test data
        X, y_true = self.preprocess_data(X_flow, X_pressure, y_true, train=False)
        
        # Evaluate
        results = self.model.evaluate(X, y_true, verbose=1)
        y_pred_probs = self.model.predict(X)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # RMSE: Root Mean Squared Error between predicted and true class (for leak location)
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        print(f"\nRMSE (leak location index): {rmse:.4f}")
        
        # MRLE: Maximum Relative Leakage Error (for leak cases only, as in the paper)
        # MRLE = max(|predicted - true| / n_nodes) * 100%
        leak_mask = y_true > 0
        if np.any(leak_mask):
            mrle = np.max(np.abs(y_pred[leak_mask] - y_true[leak_mask]) / (np.max(y_true[leak_mask])) ) * 100
            print(f"MRLE (for leak cases): {mrle:.2f}%")
        else:
            mrle = None
            print("No leak cases in evaluation set for MRLE calculation.")
        
        # Classification report and confusion matrix
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        
        return results, y_pred, y_true
    
    def save(self, model_path='model'):
        """
        Save the trained model
        """
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save('{}/leakage_detection_model.h5'.format(model_path))
        
    def load(self, model_path='model/leakage_detection_model.h5'):
        """
        Load a pre-trained model
        """
        self.model = tf.keras.models.load_model(model_path)

def generate_synthetic_data(n_samples=1000, n_features=8, leak_ratio=0.3):
    """
    Generate synthetic data for testing the implementation
    In a real scenario, this would be replaced with actual water distribution network data
    
    According to the paper, features would include:
    - Pressure readings from different nodes
    - Flow rate measurements
    - Time of day (can be cyclical features)
    - Water demand patterns
    """
    np.random.seed(42)
    
    # Create features (pressure readings, flow rates, etc.)
    X = np.random.rand(n_samples, n_features)
    
    # Create target variable (leak or no leak)
    y = np.zeros(n_samples)
    
    # Introduce some patterns for leaks
    # Higher pressure differences and abnormal flow rates indicate leaks
    leak_samples = np.random.choice(n_samples, int(n_samples * leak_ratio), replace=False)
    
    # Simulate leaks by adjusting feature values
    for idx in leak_samples:
        # Decrease pressure in some nodes
        X[idx, 0:3] *= 0.7
        # Increase flow rates
        X[idx, 3:6] *= 1.3
        # Mark as leak
        y[idx] = 1
    
    return X, y

def main():
    print("Leakage Detection in Water Distribution Networks")
    print("Using Hybrid Feedforward Artificial Neural Networks")
    print("-" * 60)
    
    # Generate data
    # In a real scenario, you would load actual water distribution network data
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=5000, n_features=10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    print("\nCreating and training the hybrid neural network model...")
    model = LeakageDetectionModel()
    model.create_hybrid_model(input_shape=X_train.shape[1], hidden_layers=[64, 32, 16])
    
    history = model.train(X_train, y_train, epochs=50, batch_size=32)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Evaluate the model
    print("\nEvaluating the model...")
    results, y_pred, y_true = model.evaluate(X_test, y_test)
    
    # Save the model
    print("\nSaving the model...")
    model.save()
    
    print("\nLeakage detection model implementation completed.")

if __name__ == "__main__":
    main() 