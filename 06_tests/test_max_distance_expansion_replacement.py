import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from max_distance_expansion_replacement import MaxDistanceExpansionReplacement
import numpy as np

# MaxDistanceExpansionReplacement setup
buffer_data = np.array([
    [2.1, 2.1], 
    [2.0, 2.0], 
    [4.0, 4.0], 
    [8.0, 8.0]
])
buffer_ids = np.array([100, 101, 102, 103])

print("Initializing MaxDistanceExpansionReplacement...")
strategy = MaxDistanceExpansionReplacement(buffer_data, buffer_ids)

print("Buffer data:\n" + str(strategy.buffer))
print("Sample IDs: " + str(strategy.samples_ids))
print("Max Distance (d_max): " + str(strategy.max_distance))
print("Min Index 1 (Row): " + str(strategy.min_idx_1))
print("Min Index 2 (Row): " + str(strategy.min_idx_2))

# Test rejection (False)
new_val_false = np.array([3.0, 3.0]) 
print("\nEvaluating measurement inside the cluster bounds: " + str(new_val_false))
accepted, replaced_index = strategy.check_new_measurement(new_val_false)
print("Index to replace: " + str(replaced_index))

# Test acceptance (gives True due to geometric expansion)
new_val_expand = np.array([12.0, 12.0]) 
print("\nEvaluating new extreme measurement (Cluster Expansion): " + str(new_val_expand))
accepted, replaced_index = strategy.check_new_measurement(new_val_expand)
print("Index to replace: " + str(replaced_index))

if accepted:
    new_id_1 = 104
    # Replacing the redundant point from the closest pair to maintain geometric coverage
    buffer_data[replaced_index] = new_val_expand
    buffer_ids[replaced_index] = new_id_1
    strategy.sync_state()

print("Updated Max Distance (d_max): " + str(strategy.max_distance))
print("Updated Min Index Pair: (" + str(strategy.min_idx_1) + ", " + str(strategy.min_idx_2) + ")")