import numpy as np
from vector_cusum import VectorCusum

# The script performs a Monte Carlo simulation on ideal multivariate normal data 
# to calculate the theoretical ARL0 for various combinations of h and k.

print("1. Constructing the theoretical baseline (μ and Σ)...")

# Mean vector from the provided table
mean_vector = np.array([5.74, 17.29, 3268.67, 12.44, 12.14, 10.79, 6.56])

# Standard deviations from the provided table
std_devs = np.array([5.20, 22.22, 5553.74, 15.05, 8.95, 14.21, 9.27])

# Correlation matrix from the provided table
corr_matrix = np.array([
    [ 1.00,  0.85,  0.80, -0.25, -0.30, -0.15, -0.25],
    [ 0.85,  1.00,  0.95,  0.15,  0.20,  0.10,  0.15],
    [ 0.80,  0.95,  1.00,  0.20,  0.25,  0.15,  0.20],
    [-0.25,  0.15,  0.20,  1.00,  0.60,  0.70,  0.40],
    [-0.30,  0.20,  0.25,  0.60,  1.00,  0.40,  0.70],
    [-0.15,  0.10,  0.15,  0.70,  0.40,  1.00,  0.50],
    [-0.25,  0.15,  0.20,  0.40,  0.70,  0.50,  1.00]
])

# Covariance Matrix calculation: Σ_ij = R_ij * σ_i * σ_j
covariance_matrix = np.outer(std_devs, std_devs) * corr_matrix

print("2. Generating Ideal Data Stream...")
# Generate 500,000 samples 
TOTAL_MOCK_SAMPLES = 500000
ideal_data_stream = np.random.multivariate_normal(mean_vector, covariance_matrix, TOTAL_MOCK_SAMPLES)

# k values: from 0.5 to 5.0 with a step of 0.5
k_values = np.arange(0.5, 5.5, 0.5)  

# h values: from 5.0 to 20.0 with a step of 1.0 for precise threshold detection
h_values = np.arange(5.0, 21.0, 1.0) 

required_epochs = 50 # Number of test Epochs per combination to calculate average

print("\n3. Starting Monte Carlo Simulation for ARL0...\n")
print(f"{'k (Ref Value)':<15} | {'h (Dec Interval)':<15} | {'Theoretical ARL0'}")
print("-" * 55)

# Iteration through parameter grid
for k in k_values:
    for h in h_values:
        
        # Init method initiates the Crosier's MCUSUM for the specific parameters
        cusum = VectorCusum(mean_vector, covariance_matrix, h, k)
        
        run_lengths = []
        data_index = 0
        
        # Collect required_epochs to calculate a statistically significant average
        while len(run_lengths) < required_epochs:
            
            sample = ideal_data_stream[data_index % TOTAL_MOCK_SAMPLES]
            
            # Update method checks for theoretical shifts
            is_out_of_control = cusum.update(sample)
            
            if is_out_of_control:
                # Store the length of the current theoretical Epoch
                run_lengths.append(cusum.run_length)
                cusum.reset()
                
            data_index += 1
            
        # Calculate the theoretical ARL0 (Average Run Length in-control)
        theoretical_arl = np.mean(run_lengths)
        
        print(f"{k:<15.1f} | {h:<15.1f} | {theoretical_arl:.2f}")

        # Break mechanism: Stop evaluating higher h values if ARL0 exceeds 500
        if theoretical_arl > 500:
            print(f"--- Threshold exceeded (>500). Stopping h-search for k={k:.1f} ---")
            break # Breaks out of the inner (h) loop and moves to the next k

print("\nSimulation Complete. Select (k, h) combinations that yield an ARL0 between 150 and 500.")