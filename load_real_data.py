import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def load_excel_data(pressure_file='Pressure_Data.xlsx', flow_file='Flow_Data.xlsx'):
    """
    Load pressure and flow data from Excel files
    
    Args:
        pressure_file: Path to the Excel file with pressure data
        flow_file: Path to the Excel file with flow data
        
    Returns:
        Dictionary containing processed data for different leak scenarios
    """
    print(f"Loading pressure data from {pressure_file}")
    pressure_data = pd.read_excel(pressure_file)
    
    print(f"Loading flow data from {flow_file}")
    flow_data = pd.read_excel(flow_file)
    
    # Print basic information about the data
    print("\nPressure Data Info:")
    print(f"Shape: {pressure_data.shape}")
    print("\nFlow Data Info:")
    print(f"Shape: {flow_data.shape}")
    
    # Get the first row which contains the node/pipe names
    pressure_nodes = pressure_data.iloc[0, 3:].values
    flow_pipes = flow_data.iloc[0, 3:].values
    
    # Rename columns 
    pressure_data.columns = ['leak_scenario', 'leak_flow_rate', 'pattern_coefficient'] + list(pressure_nodes)
    flow_data.columns = ['leak_scenario', 'leak_flow_rate', 'pattern_coefficient'] + list(flow_pipes)
    
    # Drop the first row which was used for column names
    pressure_data = pressure_data.iloc[1:].reset_index(drop=True)
    flow_data = flow_data.iloc[1:].reset_index(drop=True)
    
    # Convert numeric columns
    pressure_data['leak_flow_rate'] = pd.to_numeric(pressure_data['leak_flow_rate'], errors='coerce')
    pressure_data['pattern_coefficient'] = pd.to_numeric(pressure_data['pattern_coefficient'], errors='coerce')
    
    flow_data['leak_flow_rate'] = pd.to_numeric(flow_data['leak_flow_rate'], errors='coerce')
    flow_data['pattern_coefficient'] = pd.to_numeric(flow_data['pattern_coefficient'], errors='coerce')
    
    # Convert remaining columns to numeric (skipping the first three)
    for col in pressure_data.columns[3:]:
        pressure_data[col] = pd.to_numeric(pressure_data[col], errors='coerce')
    
    for col in flow_data.columns[3:]:
        flow_data[col] = pd.to_numeric(flow_data[col], errors='coerce')
    
    # Print summary of leak scenarios
    leak_scenarios = pressure_data['leak_scenario'].unique()
    print(f"\nNumber of unique leak scenarios: {len(leak_scenarios)}")
    print(f"Leak scenarios: {', '.join(str(s) for s in leak_scenarios[:10])}...")
    
    # Print summary of pattern coefficients
    pattern_coeffs = pressure_data['pattern_coefficient'].unique()
    print(f"\nNumber of unique pattern coefficients: {len(pattern_coeffs)}")
    print(f"Pattern coefficients: {', '.join(str(round(p, 2)) for p in sorted(pattern_coeffs)[:10])}...")
    
    return {
        'pressure': pressure_data,
        'flow': flow_data,
        'pressure_nodes': pressure_nodes,
        'flow_pipes': flow_pipes
    }

def create_combined_dataset(data_dict, output_file='real_water_network_data.csv'):
    """
    Combine pressure and flow data into a single dataset for modeling
    
    Args:
        data_dict: Dictionary containing pressure and flow DataFrames
        output_file: Path to save the combined CSV file
        
    Returns:
        Combined DataFrame ready for modeling
    """
    pressure_data = data_dict['pressure']
    flow_data = data_dict['flow']
    
    # Define the key columns for merging
    merge_cols = ['leak_scenario', 'leak_flow_rate', 'pattern_coefficient']
    
    # Select only necessary columns from flow_data (keys + flow values)
    flow_pipe_cols = data_dict['flow_pipes'].tolist()
    flow_data_selected = flow_data[merge_cols + flow_pipe_cols]
    
    # Optional: Convert to more memory-efficient types before merging
    pressure_node_cols = data_dict['pressure_nodes'].tolist()
    for col in pressure_node_cols:
        pressure_data[col] = pd.to_numeric(pressure_data[col], errors='coerce').astype(np.float32)
    for col in flow_pipe_cols:
        flow_data_selected[col] = pd.to_numeric(flow_data_selected[col], errors='coerce').astype(np.float32)
    print("Converted data columns to float32 for memory efficiency.")
    
    # Perform an inner merge to align rows based on the key columns
    print("Merging pressure and flow data...")
    combined_data = pd.merge(
        pressure_data, 
        flow_data_selected, # Use the selected flow data
        on=merge_cols, 
        how='inner'
    )
    print(f"Shape after merging: {combined_data.shape}")

    # Check if the merge resulted in any data
    if combined_data.empty:
        print("Error: Merging resulted in an empty DataFrame. Check the key columns in both Excel files.")
        return None
        
    # Rename columns to the required format (pressure_NODE, flow_PIPE)
    rename_dict = {}
    for col in pressure_node_cols:
        rename_dict[col] = f'pressure_{col}'
    for col in flow_pipe_cols:
        rename_dict[col] = f'flow_{col}'
    combined_data.rename(columns=rename_dict, inplace=True)
    print("Renamed columns.")

    # Add a leak flag column (1 if leak, 0 if no leak)
    combined_data['leak'] = combined_data['leak_scenario'].astype(str).apply(lambda x: 0 if x == 'KO CÓ' else 1)
    
    # Add a timestamp column (synthetic, assuming 15-minute intervals)
    print("Generating timestamps...")
    combined_data = combined_data.sort_values(by=merge_cols).reset_index(drop=True)
    start_date = datetime(2023, 1, 1)
    combined_data['timestamp'] = [start_date + timedelta(minutes=i * 15) for i in range(len(combined_data))]

    # Select and reorder columns for the final output CSV
    final_cols = ['timestamp', 'leak_scenario', 'leak_flow_rate', 'pattern_coefficient', 'leak'] + \
                 [f'pressure_{col}' for col in pressure_node_cols] + \
                 [f'flow_{col}' for col in flow_pipe_cols]
                 
    final_cols = [col for col in final_cols if col in combined_data.columns]
    combined_data = combined_data[final_cols]
    print("Selected and reordered final columns.")
    
    # Save to CSV
    print(f"Saving combined dataset to {output_file}...")
    combined_data.to_csv(output_file, index=False)
    print(f"Saved combined dataset successfully.")
    
    return combined_data

def visualize_data(data_dict, output_dir='plots'):
    """
    Create visualizations of the pressure and flow data
    
    Args:
        data_dict: Dictionary containing pressure and flow DataFrames
        output_dir: Directory to save plots
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pressure_data = data_dict['pressure']
    flow_data = data_dict['flow']
    
    # Plot pressure data for different leak scenarios
    plt.figure(figsize=(15, 10))
    
    # Get a sample of nodes (first 5)
    sample_nodes = pressure_data.columns[3:8]
    
    # Plot for no leak scenario with different pattern coefficients
    no_leak_data = pressure_data[pressure_data['leak_scenario'] == 'KO CÓ']
    
    for node in sample_nodes:
        plt.plot(
            no_leak_data['pattern_coefficient'], 
            no_leak_data[node], 
            label=f'{node} (No Leak)'
        )
    
    plt.xlabel('Pattern Coefficient')
    plt.ylabel('Pressure')
    plt.title('Pressure vs Pattern Coefficient (No Leak Scenario)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/pressure_no_leak.png')
    
    # Plot pressure data for a sample leak scenario
    plt.figure(figsize=(15, 10))
    
    # Find a leak scenario with sufficient data points
    leak_scenarios = pressure_data['leak_scenario'].unique()
    leak_scenarios = [s for s in leak_scenarios if s != 'KO CÓ']
    
    if len(leak_scenarios) > 0:
        sample_leak = leak_scenarios[0]
        leak_data = pressure_data[pressure_data['leak_scenario'] == sample_leak]
        
        for node in sample_nodes:
            plt.plot(
                leak_data['pattern_coefficient'], 
                leak_data[node], 
                label=f'{node} (Leak at {sample_leak})'
            )
        
        plt.xlabel('Pattern Coefficient')
        plt.ylabel('Pressure')
        plt.title(f'Pressure vs Pattern Coefficient (Leak at {sample_leak})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/pressure_with_leak.png')
    
    # Plot flow data for different leak scenarios
    plt.figure(figsize=(15, 10))
    
    # Get a sample of pipes (first 5)
    sample_pipes = flow_data.columns[3:8]
    
    # Plot for no leak scenario with different pattern coefficients
    for pipe in sample_pipes:
        plt.plot(
            no_leak_data['pattern_coefficient'], 
            flow_data[flow_data['leak_scenario'] == 'KO CÓ'][pipe], 
            label=f'{pipe} (No Leak)'
        )
    
    plt.xlabel('Pattern Coefficient')
    plt.ylabel('Flow')
    plt.title('Flow vs Pattern Coefficient (No Leak Scenario)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/flow_no_leak.png')
    
    # Plot flow data for a sample leak scenario
    plt.figure(figsize=(15, 10))
    
    if len(leak_scenarios) > 0:
        sample_leak = leak_scenarios[0]
        leak_data = flow_data[flow_data['leak_scenario'] == sample_leak]
        
        for pipe in sample_pipes:
            plt.plot(
                leak_data['pattern_coefficient'], 
                leak_data[pipe], 
                label=f'{pipe} (Leak at {sample_leak})'
            )
        
        plt.xlabel('Pattern Coefficient')
        plt.ylabel('Flow')
        plt.title(f'Flow vs Pattern Coefficient (Leak at {sample_leak})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/flow_with_leak.png')
    
    # Compare pressure distribution for leak vs no leak
    plt.figure(figsize=(15, 10))
    
    # Select one node for analysis
    if len(sample_nodes) > 0:
        node = sample_nodes[0]
        
        no_leak_values = pressure_data[pressure_data['leak_scenario'] == 'KO CÓ'][node]
        
        if len(leak_scenarios) > 0:
            leak_values = pressure_data[pressure_data['leak_scenario'] == sample_leak][node]
            
            plt.hist(no_leak_values, alpha=0.5, label='No Leak', bins=20)
            plt.hist(leak_values, alpha=0.5, label='With Leak', bins=20)
            
            plt.xlabel('Pressure')
            plt.ylabel('Frequency')
            plt.title(f'Pressure Distribution at {node}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/pressure_distribution.png')
    
    print(f"Saved visualization plots to {output_dir}/")

def analyze_data_for_leak_detection(data_dict):
    """
    Analyze how effectively the data can be used for leak detection
    
    Args:
        data_dict: Dictionary containing pressure and flow DataFrames
    """
    pressure_data = data_dict['pressure']
    flow_data = data_dict['flow']
    
    # Get the set of leak scenarios
    # Convert to string to handle potential mixed types (NaNs, strings)
    leak_scenarios_raw = pressure_data['leak_scenario'].astype(str).unique()
    leak_scenarios = sorted(leak_scenarios_raw)
    
    # Calculate statistics for each scenario
    print("\nData Analysis for Leak Detection:")
    print("\nPressure Statistics by Leak Scenario:")
    
    # Sample a few nodes for analysis
    sample_nodes = pressure_data.columns[3:6]  # First 3 nodes
    
    for scenario in leak_scenarios[:5]:  # Show first 5 scenarios
        scenario_data = pressure_data[pressure_data['leak_scenario'] == scenario]
        stats = scenario_data[sample_nodes].describe().loc[['mean', 'std', 'min', 'max']]
        print(f"\nScenario: {scenario}")
        print(stats)
    
    print("\nFlow Statistics by Leak Scenario:")
    
    # Sample a few pipes for analysis
    sample_pipes = flow_data.columns[3:6]  # First 3 pipes
    
    for scenario in leak_scenarios[:5]:  # Show first 5 scenarios
        scenario_data = flow_data[flow_data['leak_scenario'] == scenario]
        stats = scenario_data[sample_pipes].describe().loc[['mean', 'std', 'min', 'max']]
        print(f"\nScenario: {scenario}")
        print(stats)
    
    # Check data separability for leak detection
    print("\nChecking data separability for leak detection...")
    
    # Mark leak vs no leak
    pressure_data['is_leak'] = pressure_data['leak_scenario'] != 'KO CÓ'
    
    # Simplified analysis of a few nodes
    for node in sample_nodes:
        no_leak_values = pressure_data[~pressure_data['is_leak']][node]
        leak_values = pressure_data[pressure_data['is_leak']][node]
        
        no_leak_mean = no_leak_values.mean()
        leak_mean = leak_values.mean()
        
        print(f"{node}: No leak mean = {no_leak_mean:.4f}, Leak mean = {leak_mean:.4f}, Difference = {abs(no_leak_mean - leak_mean):.4f}")

if __name__ == "__main__":
    # Load data
    data_dict = load_excel_data()
    
    # Analyze data for leak detection
    analyze_data_for_leak_detection(data_dict)
    
    # Create visualizations
    visualize_data(data_dict)
    
    # Create combined dataset for the model
    combined_data = create_combined_dataset(data_dict)
    
    print("\nNext steps:")
    print("1. Use the combined dataset with the existing model pipeline")
    print("2. Run the model training script with the real data")
    print("3. Evaluate model performance for leak detection") 