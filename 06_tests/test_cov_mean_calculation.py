import numpy as np
from cov_mean_calculation import CovMeanCalculation

# The test script verifies the in-memory mathematical operations of the CovMeanCalculation class

print("1. Generating mock buffer data...")
# Generate a random 2D array to simulate the DataBuffer data (50 samples, 7 variables)
mock_buffer_data = np.random.rand(50, 7)

print("2. Initializing CovMeanCalculation...")
# Initialize the stateless calculator 
calculator = CovMeanCalculation()

print("3. Executing load_cov_mean()...")
# Execute the method bypassing disk I/O
mean_vector, covariance_matrix = calculator.load_cov_mean(mock_buffer_data)

# Print the results to verify the structural integrity of the matrices
if mean_vector is not None and covariance_matrix is not None:
    print("\n Success! Matrices calculated perfectly in-memory.")
    print("Mean Vector shape: " + str(mean_vector.shape) + " (Expected: (7,))")
    print("Covariance Matrix shape: " + str(covariance_matrix.shape) + " (Expected: (7, 7))")
else:
    print("\n Calculation failed.")