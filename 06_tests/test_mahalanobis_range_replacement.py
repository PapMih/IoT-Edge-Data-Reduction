import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mahalanobis_range_replacement import MahalanobisRangeReplacement
import numpy as np

# MahalanobisRangeReplacement setup
buffer_data = np.array([
    [2.1, 2.1], 
    [0.0, 0.0], 
    [4.0, 4.0], 
    [1.9, 1.9]
])
buffer_ids = np.array([100, 101, 102, 103])
inv_cov_matrix = np.eye(2)   # Identity matrix

print("Initializing MahalanobisRangeReplacement...")
strategy = MahalanobisRangeReplacement(buffer_data, buffer_ids, inv_cov_matrix)

print("Buffer data:\n" + str(strategy.buffer))
print("Sample IDs: " + str(strategy.samples_ids))
print("Mean Vector: " + str(strategy.mean_vector))
print("Distances array: " + str(strategy.distances))
print("Min Index (Row): " + str(strategy.min_idx))
print("Min Distance ID: " + str(strategy.min_distance_id))
print("Max Distance ID: " + str(strategy.max_distance_id))
print("Max Distance (d_max): " + str(strategy.max_distance))

# Test rejection (False) – a point that does not increase the spread
new_val_false = np.array([2.1, 2.1]) 
print("\nEvaluating measurement that keeps the spread unchanged: " + str(new_val_false))
accepted, replaced_index = strategy.check_new_measurement(new_val_false)
print("Accepted? " + str(accepted))
print("Index to replace: " + str(replaced_index))

# Test acceptance due to centroid‑shift (True) – a point close to the mean that shifts the centroid
new_val_shift = np.array([2.0, 2.0]) 
print("\nEvaluating new measurement close to the mean (centroid‑shift case): " + str(new_val_shift))
accepted, replaced_index = strategy.check_new_measurement(new_val_shift)
print("Accepted? " + str(accepted))
print("Index to replace: " + str(replaced_index))

if accepted:
    new_id_1 = 104
    # Replace the point with the minimum Mahalanobis distance (min_idx)
    buffer_data[strategy.min_idx] = new_val_shift
    buffer_ids[strategy.min_idx] = new_id_1
    strategy.sync_state()

print("Updated Mean Vector after centroid‑shift acceptance: " + str(strategy.mean_vector))

# Test acceptance (True) – a genuine extreme outlier
new_val_far = np.array([10.0, 10.0])
print("\nEvaluating new extreme measurement (true expansion): " + str(new_val_far))
accepted, replaced_index = strategy.check_new_measurement(new_val_far)
print("Accepted? " + str(accepted))
print("Index to replace: " + str(replaced_index))

if accepted:
    new_id_2 = 105
    buffer_data[strategy.min_idx] = new_val_far
    buffer_ids[strategy.min_idx] = new_id_2
    strategy.sync_state()

print("Updated Mean Vector after true outlier acceptance: " + str(strategy.mean_vector))
