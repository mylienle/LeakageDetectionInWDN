#!/usr/bin/env python3
# Generate synthetic water distribution network data for leakage detection

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

def generate_water_network_data(
    n_days=30,
    sampling_rate_mins=15,
    n_pressure_sensors=5,
    n_flow_sensors=3,
    leak_periods=None,
    output_file='synthetic_water_network_data.csv'
):
    """
    Generate synthetic water distribution network data
    
    Args:
        n_days: Number of days to simulate
        sampling_rate_mins: Sampling rate in minutes
        n_pressure_sensors: Number of pressure sensors in the network
        n_flow_sensors: Number of flow sensors in the network
        leak_periods: List of tuples with (start_day, end_day) for leak periods
        output_file: Path to save the CSV file
        
    Returns:
        DataFrame with synthetic water network data
    """
    # Calculate number of samples
    n_samples = int(n_days * 24 * 60 / sampling_rate_mins)
    
    # Create timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(minutes=i*sampling_rate_mins) for i in range(n_samples)]
    
    # Create a DataFrame with timestamps
    data = pd.DataFrame({'timestamp': timestamps})
    
    # Add hour and day of week for cyclical patterns
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # Generate base pressure data (normal distribution around mean values)
    base_pressure = 60  # Base pressure in PSI
    for i in range(n_pressure_sensors):
        # Add some variation between sensors
        sensor_base = base_pressure + np.random.normal(0, 5)
        
        # Generate pressure with daily patterns
        pressure = np.zeros(n_samples)
        for j in range(n_samples):
            hour = data['hour'].iloc[j]
            day = data['day_of_week'].iloc[j]
            
            # Daily pattern: pressure drops during peak usage hours (morning and evening)
            hour_factor = -2 * np.sin(hour * np.pi / 12)
            
            # Weekly pattern: lower pressure on weekends
            day_factor = -1 if day >= 5 else 0  # Weekend effect
            
            # Add noise
            noise = np.random.normal(0, 1)
            
            # Combine factors
            pressure[j] = sensor_base + hour_factor + day_factor + noise
        
        data[f'pressure_{i+1}'] = pressure
    
    # Generate base flow data
    base_flow = 100  # Base flow in gallons per minute
    for i in range(n_flow_sensors):
        # Add some variation between sensors
        sensor_base = base_flow + np.random.normal(0, 20)
        
        # Generate flow with daily patterns
        flow = np.zeros(n_samples)
        for j in range(n_samples):
            hour = data['hour'].iloc[j]
            day = data['day_of_week'].iloc[j]
            
            # Daily pattern: flow increases during peak usage hours
            hour_factor = 15 * np.sin(hour * np.pi / 12) + 10 * np.sin(hour * np.pi / 6)
            
            # Weekly pattern: higher flow on weekends
            day_factor = 5 if day >= 5 else 0
            
            # Add noise
            noise = np.random.normal(0, 3)
            
            # Combine factors
            flow[j] = max(0, sensor_base + hour_factor + day_factor + noise)  # Ensure non-negative
        
        data[f'flow_{i+1}'] = flow
    
    # Initialize leak status (0 = no leak, 1 = leak)
    data['leak'] = 0
    
    # Introduce leaks during specified periods
    if leak_periods:
        for start_day, end_day in leak_periods:
            start_idx = int(start_day * 24 * 60 / sampling_rate_mins)
            end_idx = int(end_day * 24 * 60 / sampling_rate_mins)
            
            # Mark as leak
            data.loc[start_idx:end_idx, 'leak'] = 1
            
            # Adjust pressure and flow during leak
            for i in range(n_pressure_sensors):
                # Different sensors are affected differently by the leak
                effect = np.random.uniform(0.7, 0.95)  # Pressure drop effect
                data.loc[start_idx:end_idx, f'pressure_{i+1}'] *= effect
            
            for i in range(n_flow_sensors):
                # Flow sensors downstream of leak show decreased flow,
                # while sensors upstream might not show significant changes
                if i < n_flow_sensors // 2:
                    effect = np.random.uniform(0.8, 0.95)  # Flow decrease
                    data.loc[start_idx:end_idx, f'flow_{i+1}'] *= effect
                else:
                    # Add more variation to flow due to leak
                    data.loc[start_idx:end_idx, f'flow_{i+1}'] += np.random.normal(0, 5, size=end_idx-start_idx+1)
    
    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"Generated synthetic data with {n_samples} samples and saved to {output_file}")
    
    return data

def plot_data(data, output_dir='plots'):
    """
    Plot the generated data for visualization
    
    Args:
        data: DataFrame with synthetic data
        output_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot pressure data
    pressure_cols = [col for col in data.columns if 'pressure' in col]
    plt.figure(figsize=(15, 7))
    for col in pressure_cols:
        plt.plot(data['timestamp'], data[col], label=col)
    
    # Highlight leak periods
    leak_periods = data[data['leak'] == 1]['timestamp']
    if len(leak_periods) > 0:
        plt.axvspan(leak_periods.min(), leak_periods.max(), color='red', alpha=0.2, label='Leak Period')
    
    plt.xlabel('Time')
    plt.ylabel('Pressure (PSI)')
    plt.title('Pressure Sensor Readings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pressure_data.png')
    
    # Plot flow data
    flow_cols = [col for col in data.columns if 'flow' in col]
    plt.figure(figsize=(15, 7))
    for col in flow_cols:
        plt.plot(data['timestamp'], data[col], label=col)
    
    # Highlight leak periods
    if len(leak_periods) > 0:
        plt.axvspan(leak_periods.min(), leak_periods.max(), color='red', alpha=0.2, label='Leak Period')
    
    plt.xlabel('Time')
    plt.ylabel('Flow (GPM)')
    plt.title('Flow Sensor Readings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/flow_data.png')
    
    # Plot daily patterns
    plt.figure(figsize=(15, 7))
    data.groupby('hour')['pressure_1'].mean().plot(label='Avg Pressure')
    data.groupby('hour')['flow_1'].mean().plot(label='Avg Flow', secondary_y=True)
    plt.xlabel('Hour of Day')
    plt.title('Daily Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/daily_patterns.png')

if __name__ == "__main__":
    # Define leak periods (start_day, end_day)
    leak_periods = [
        (5, 7),    # 2-day leak in the first week
        (15, 17),  # 2-day leak in the third week
        (25, 28)   # 3-day leak in the fourth week
    ]
    
    # Generate data
    data = generate_water_network_data(
        n_days=30,
        sampling_rate_mins=15,
        n_pressure_sensors=5,
        n_flow_sensors=3,
        leak_periods=leak_periods,
        output_file='synthetic_water_network_data.csv'
    )
    
    # Plot data
    plot_data(data, output_dir='plots') 