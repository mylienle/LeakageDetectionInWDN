import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import sys
import os
from data_utils import load_duc_data
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
    # if len(sys.argv) != 3:
    #     print("Usage: python3 test_model.py <input_file.xlsx or .csv> <output_predictions.csv>")
    #     return
    
    # Try different encodings for reading the CSV files
    try:
        flow_features, pressure_features, y, metadata = load_duc_data(
            "Data_thDuc/test_case2_PIPE.csv",
            "Data_thDuc/test_case2_JUNC.csv",
            encoding='latin1'  # Try latin1 encoding first
        )
    except Exception as e:
        print(f"Error with latin1 encoding: {str(e)}")
        try:
            flow_features, pressure_features, y, metadata = load_duc_data(
                "Data_thDuc/test_case2_PIPE.csv",
                "Data_thDuc/test_case2_JUNC.csv",
                encoding='cp1252'  # Try Windows-1252 encoding
            )
        except Exception as e:
            print(f"Error with cp1252 encoding: {str(e)}")
            return
    
    # Load model and scalers
    try:
        model = load_model('model/leakage_detection_model.h5')
        flow_scaler = joblib.load('model/flow_scaler.save')
        pressure_scaler = joblib.load('model/pressure_scaler.save')
    except Exception as e:
        print(f"Error loading model or scalers: {str(e)}")
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
    output_csv = "predictions.csv"  # Default output filename
    df_out = pd.DataFrame({
        'actual': y,
        'predicted_leak_location': y_pred
    })
    df_out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main() 