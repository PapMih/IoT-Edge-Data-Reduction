import sys
import os
import numpy as np
import pandas as pd

# Add the 04_evaluation folder to the system path to import the evaluator class
sys.path.append(os.path.abspath(os.path.join('..', '04_evaluation')))
from volume_comparison_evaluator import VolumeComparisonEvaluator

# Definition of the multivariate feature columns (sensors)
columns = [
    'SPEEDKNOTS', 'MERPM', 'MEPOWER', 
    'RANGEX', 'RANGEY', 'INCLINOMETERXZC', 'INCLINOMETERYZC'
]

# 1. Initialization of controlled synthetic data for the original stream
print("Initializing Test Data for Original Stream...")
np.random.seed(42)
original_data = np.random.rand(100, 7) * 10.0
original_df = pd.DataFrame(original_data, columns=columns)

# 2. Initialization of controlled buffer synopsis (Reservoir buffer)
print("Initializing Test Data for Buffer Synopsis...")
# Simulation of a single Epoch (e.g., indices 0 to 49) with 20 samples in the reservoir
buffer_data = original_data[0:20].copy() 
synopsis_df = pd.DataFrame(buffer_data, columns=columns)

# Insertion of Epoch metadata (as exported by the reservoir)
synopsis_df['Epoch_ID'] = 1
synopsis_df['Start_Index'] = 0
synopsis_df['End_Index'] = 49

print("Original Stream DataFrame head:\n" + str(original_df.head(2)))
print("Buffer Synopsis DataFrame head:\n" + str(synopsis_df.head(2)))

# 3. Initialization of the Volume Comparison Evaluator
print("\nInitializing VolumeComparisonEvaluator...")
evaluator = VolumeComparisonEvaluator(original_df, synopsis_df)

# Check dataset dimensions
print("Original Dataset dimensions: " + str(evaluator.original_dataset_df.shape))
print("Synopsis Dataset dimensions: " + str(evaluator.synopsis_df.shape))

# 4. Manual testing of the private volume calculation method
print("\nEvaluating single Bounding Box Volume calculation...")
test_matrix = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
])
# Expected ranges: [1, 2, 3, 4, 5, 6, 7] -> Product = 1*2*3*4*5*6*7 = 5040
calc_volume = evaluator._calculate_bounding_box_volume(test_matrix)
print("Test Matrix Volume: " + str(calc_volume))

# 5. Execution of the complete evaluation runner
print("\nExecuting run() for all Epochs...")
r_score = evaluator.run()

print("Final Sampling Quality Score - Volume Ratio (R): " + str(r_score))