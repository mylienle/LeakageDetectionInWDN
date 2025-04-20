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
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def create_hybrid_model(self, input_shape, hidden_layers=[64, 32], dropout_rate=0.2):
        """
        Create a hybrid feedforward neural network for leakage detection
        Based on the paper's architecture
        """
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(input_shape,)))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))
            
        # Output layer - binary classification (leak or no leak)
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def preprocess_data(self, X, y=None, train=True):
        """
        Preprocess data using standardization or normalization
        According to the paper, the pressure and flow rate data need to be normalized
        """
        if train:
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, y
        else:
            # For validation/test data
            X_scaled = self.scaler.transform(X)
            return X_scaled, y
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the hybrid neural network model
        """
        # Preprocess data
        X_scaled, y = self.preprocess_data(X, y, train=True)
        
        # Train model
        history = self.model.fit(
            X_scaled, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        """
        # Preprocess test data
        X_test_scaled, y_test = self.preprocess_data(X_test, y_test, train=False)
        
        # Evaluate
        results = self.model.evaluate(X_test_scaled, y_test, verbose=1)
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
        
        # Generate classification report and confusion matrix
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        
        return results, y_pred
    
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
    results, y_pred = model.evaluate(X_test, y_test)
    
    # Save the model
    print("\nSaving the model...")
    model.save()
    
    print("\nLeakage detection model implementation completed.")

if __name__ == "__main__":
    main() 