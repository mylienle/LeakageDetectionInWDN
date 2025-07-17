import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import sys
import os

# Usage: python3 test_model.py new_test_data.csv output_predictions.csv

def convert_excel_to_csv(excel_file):
    """
    Convert Excel file to CSV format matching our training data structure.
    Expected Excel format:
    - First 3 columns: scenario, leak_rate, pattern
    - Next 198 columns: flow measurements
    - Next 122 columns: pressure measurements
    """
    try:
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Generate output CSV filename
        base_name = os.path.splitext(excel_file)[0]
        output_csv = f"{base_name}_converted.csv"
        
        # Verify column count
        expected_columns = 3 + 198 + 122  # scenario + leak_rate + pattern + flows + pressures
        if len(df.columns) != expected_columns:
            raise ValueError(f"Excel file must have exactly {expected_columns} columns. Found {len(df.columns)} columns.")
        
        # Save as CSV
        df.to_csv(output_csv, index=False)
        print(f"Converted Excel file to CSV: {output_csv}")
        return output_csv
        
    except Exception as e:
        print(f"Error converting Excel file: {str(e)}")
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 test_model.py <input_file.xlsx or .csv> <output_predictions.csv>")
        return
    
    input_file = sys.argv[1]
    output_csv = sys.argv[2]
    
    # Check if input is Excel file
    if input_file.endswith(('.xlsx', '.xls')):
        input_file = convert_excel_to_csv(input_file)
        if input_file is None:
            return
    
    # Load model and scalers
    try:
        model = load_model('leakage_detection_model.h5')
        flow_scaler = joblib.load('flow_scaler.save')
        pressure_scaler = joblib.load('pressure_scaler.save')
    except Exception as e:
        print(f"Error loading model or scalers: {str(e)}")
        return

    # Load new test data
    try:
        df = pd.read_csv(input_file)
        # Extract features (assuming same format: scenario, leak_rate, pattern, flows..., pressures...)
        # For Case 1: all flows (198) and all pressures (122)
        flow_features = df.iloc[:, 3:3+198].values
        pressure_features = df.iloc[:, 3+198:3+198+122].values
    except Exception as e:
        print(f"Error processing input data: {str(e)}")
        return

    # Scale features
    X_flow = flow_scaler.transform(flow_features)
    X_pressure = pressure_scaler.transform(pressure_features)
    # Concatenate features as in training
    X = np.concatenate([X_flow, X_pressure], axis=1)

    # Predict
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Save predictions
    df_out = df.copy()
    df_out['predicted_leak_location'] = y_pred
    df_out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main() 