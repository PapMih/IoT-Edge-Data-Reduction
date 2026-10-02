from max_range_replacement import MaxRangeReplacement
import numpy as np

# MaxRangeReplacement setup
buffer_data = np.array([
    [2.1, 2.1], 
    [0.0, 0.0], 
    [4.0, 4.0], 
    [1.9, 1.9]
])
buffer_ids = np.array([100, 101, 102, 103])

print("Initializing MaxRangeReplacement...")
strategy = MaxRangeReplacement(buffer_data, buffer_ids)

print("Buffer data:\n" + str(strategy.buffer))
print("Sample IDs: " + str(strategy.samples_ids))
print("Mean Vector: " + str(strategy.mean_vector))
print("Distances array: " + str(strategy.distances))
print("Min Index (Row): " + str(strategy.min_idx))
print("Min Distance ID: " + str(strategy.min_distance_id))
print("Max Distance ID: " + str(strategy.max_distance_id))
print("Max Distance (d_max): " + str(strategy.max_distance))

# Test absolute rejection (False)
new_val_false = np.array([2.1, 2.1]) 
print("\nEvaluating measurement that keeps mean stable: " + str(new_val_false))
accepted, replaced_index = strategy.check_new_measurement(new_val_false)
print("Index to replace: " + str(replaced_index))

# Test acceptance (gives True due to geometric shift)
new_val_close = np.array([2.0, 2.0]) 
print("\nEvaluating new measurement close to mean: " + str(new_val_close))
accepted, replaced_index = strategy.check_new_measurement(new_val_close)
print("Index to replace: " + str(replaced_index))

if accepted:
    new_id_1 = 104
    buffer_data[strategy.min_idx] = new_val_close
    buffer_ids[strategy.min_idx] = new_id_1
    strategy.sync_state()

print("Updated Mean Vector: " + str(strategy.mean_vector))

# Test acceptance (Extreme Outlier)
new_val_far = np.array([10.0, 10.0])
print("\nEvaluating new extreme measurement: " + str(new_val_far))
accepted, replaced_index = strategy.check_new_measurement(new_val_far)
print("Index to replace: " + str(replaced_index)) 

if accepted:
    new_id_2 = 105
    buffer_data[strategy.min_idx] = new_val_far
    buffer_ids[strategy.min_idx] = new_id_2
    strategy.sync_state()

print("Updated Mean Vector: " + str(strategy.mean_vector))