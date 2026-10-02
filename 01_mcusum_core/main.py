import os
import pandas as pd
import numpy as np
from numpy.linalg import inv 

from buffer import DataBuffer
from cov_mean_calculation import CovMeanCalculation
from vector_cusum import VectorCusum
from max_range_replacement import MaxRangeReplacement
from mahalanobis_range_replacement import MahalanobisRangeReplacement
from max_distance_expansion_replacement import MaxDistanceExpansionReplacement
from synopsis_exporter import SynopsisExporter

# 1. Configuration & Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(base_dir, '..')                                
output_base_dir = os.path.join(parent_dir, 'mcusum_outputs')

# Replacing methods 
replacement_methods = [
    'MaxRangeReplacement',
    'MahalanobisRangeReplacement',
    'MaxDistanceExpansionReplacement'
]

# Folders for each replacement method
for method in replacement_methods:
    method_dir = os.path.join(output_base_dir, method)
    if not os.path.exists(method_dir):
        os.makedirs(method_dir)

# Target parameters
parameters_grid = [
    {"k": 0.5, "h": 10.0},
    {"k": 0.5, "h": 11.0},
    {"k": 0.5, "h": 12.0},
    {"k": 0.5, "h": 13.0},
    {"k": 0.5, "h": 14.0},
    {"k": 0.5, "h": 15.0},
    {"k": 1.0, "h": 6.0},
    {"k": 1.0, "h": 7.0},
    {"k": 1.0, "h": 8.0},
    {"k": 1.0, "h": 9.0}
]

buffer_sizes = [50, 100, 150, 200]

# Initialize the stateless statistics calculator
calculator = CovMeanCalculation()

print("Starting overall simulation for 10 datasets...\n")

# OUTER LOOP: Iterate over the 10 datasets
for dataset_id in range(1, 11):
    print(f"=== Processing Dataset {dataset_id}/10 ===")
    
    # Define input path for the current dataset
    data_file_path = os.path.join(parent_dir, 'synthetic_data', f'synthetic_ship_dataset_{dataset_id}.xlsx')
    
    # Folders for each dataset
    for method in replacement_methods:
        dataset_output_dir = os.path.join(output_base_dir, method, f'Synthetic ship dataset {dataset_id}')
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

    # Load the Excel dataset
    df = pd.read_excel(data_file_path)

    # Exclude Timestamp if it exists to isolate numeric variables for linear algebra operations
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    feature_names = df.columns.tolist()
    data_stream = df.values.astype(np.float32)
    total_measurements = len(data_stream)
    variables_num = data_stream.shape[1]
    
    print(f"Dataset loaded: {total_measurements} measurements, {variables_num} variables.")

    # 2. Main Execution Loop (For the current dataset)
    for params in parameters_grid:
        k_val = params["k"]
        h_val = params["h"]
        
        for b_size in buffer_sizes:
            
            # For each replacement method
            for method_name in replacement_methods:
                print(f"  -> Running K={k_val}, H={h_val}, Buffer Size={b_size}, Method={method_name}...")
                
                # Initialize components for the current configuration
                buffer = DataBuffer(b_size, variables_num)
                exporter = SynopsisExporter(h_val, k_val, b_size)
                
                # Dynamically redirect the exporter's file path to the specific METHOD and DATASET sub-folder
                dataset_output_dir = os.path.join(output_base_dir, method_name, f'Synthetic ship dataset {dataset_id}') 
                exporter.file_name = os.path.join(dataset_output_dir, os.path.basename(exporter.file_name))
                
                epoch_id = 1
                start_index = 0
                
                # Pointers for our dynamic objects
                cusum = None
                replacer = None

                previous_measurement = None 
                
                for index, new_measurement in enumerate(data_stream):
                    # Phase 1: Sequential filling of the Reservoir
                    if not buffer.isFull():
                        buffer.insert(new_measurement, index)
                        
                        # Check if the buffer JUST became full (Transition trigger)
                        if buffer.isFull():
                            
                            # Retrieve buffer views for matrix calculation
                            buffer_data, buffer_ids = buffer.getDataAndIds()
                            
                            # Calculate Baseline (Σ and μ) using the differentials 
                            diff_data = calculator.differential_calc(buffer_data)
                            mean_vector_diff, covariance_matrix_diff = calculator.load_cov_mean(diff_data)
                            
                            # Initialize Crosier's MCUSUM
                            cusum = VectorCusum(mean_vector_diff, covariance_matrix_diff, h_val, k_val)

                            # Mean and covariance matrix of the raw data, not the differentials
                            mean_vector_raw, covariance_matrix_raw = calculator.load_cov_mean(buffer_data)
                            
                            # Replacement classes initialization 
                            if method_name == 'MaxRangeReplacement':
                                replacer = MaxRangeReplacement(buffer_data, buffer_ids)
                            elif method_name == 'MahalanobisRangeReplacement':
                                raw_inv_cov_matrix = inv(covariance_matrix_raw)
                                replacer = MahalanobisRangeReplacement(buffer_data, buffer_ids, raw_inv_cov_matrix)
                            elif method_name == 'MaxDistanceExpansionReplacement':
                                replacer = MaxDistanceExpansionReplacement(buffer_data, buffer_ids)
                            
                    # Phase 2: Steady state (Concept drift detection & Sampling)
                    else:
                        diff_measurement = new_measurement - previous_measurement 

                        # Update CUSUM with the new measurement and check for ALARM simultaneously
                        if cusum.update(diff_measurement):
                            
                            # ALARM: Concept Drift Detected - End of Epoch
                            end_index = index
                            
                            # Retrieve final snapshot of the reservoir and log it
                            buffer_data, _ = buffer.getDataAndIds()
                            global_indices = buffer.getGlobalIndices()
                            exporter.log_epoch(buffer_data, global_indices, epoch_id, start_index, end_index)
                            
                            # Reset the buffer and CUSUM to re-enter Phase 1 for the new Epoch
                            buffer.reset()
                            cusum.reset()
                            epoch_id += 1
                            start_index = index + 1
                            
                        else:
                            # IN-CONTROL: Execute Replacement Strategy
                            is_replaced, replace_idx = replacer.check_new_measurement(new_measurement)
                            
                            if is_replaced:
                                buffer.replacement(new_measurement, replace_idx, index)
                                replacer.sync_state() # State synchronization

                    previous_measurement = new_measurement 

                # Post-Processing: Log the final, incomplete epoch at the end of the data stream
                if buffer.total_samples_counter > 0:
                    end_index = total_measurements - 1
                    buffer_data, _ = buffer.getDataAndIds()
                    global_indices = buffer.getGlobalIndices()
                    
                    # Extract only the valid rows if the buffer wasn't completely full
                    if buffer.total_samples_counter < buffer.size:
                        valid_data = buffer_data[:buffer.total_samples_counter]
                        valid_indices = global_indices[:buffer.total_samples_counter]
                    else:
                        valid_data = buffer_data
                        valid_indices = global_indices
                    exporter.log_epoch(valid_data, valid_indices, epoch_id, start_index, end_index)
                    
                # Write the accumulated synopses to the CSV file
                exporter.export_results(feature_names)
                
print("\nAll simulations completed successfully for all replacement methods! Check the generated folders.")

