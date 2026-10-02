from buffer import DataBuffer
import numpy as np

# Buffer setup
size = 2
vars_num = 2
buf = DataBuffer(size, vars_num)

print("Buffer check - Full: " + str(buf.isFull()))

# Filling the buffer (Phase 1)
d1 = np.array([10.1, 15.2])
d2 = np.array([20.3, 25.4])

buf.insert(d1)
buf.insert(d2)
print("Data inserted. Is full: " + str(buf.isFull()))

# Read data and ids
data, ids = buf.getDataAndIds()
print("IDs in buffer: " + str(ids))
print("Current data:\n" + str(data))

# Test replacement (Phase 2)
new_val = np.array([99.0, 99.0])
target_index = 0
print("Replacing INDEX 0 with new data...")
success = buf.replacement(new_val, target_index)
print("Replacement success: " + str(success))

# Check new state
data, ids = buf.getDataAndIds()
print("Updated IDs: " + str(ids))
print("Updated data:\n" + str(data))

# Reset test
buf.reset()
print("After reset - Full: " + str(buf.isFull()) + ", Counter: " + str(buf.total_samples_counter))

print("Buffer check - Full: " + str(buf.isFull()))

# Filling the buffer (Phase 1)
d1 = np.array([10.1, 15.2])
d2 = np.array([20.3, 25.4])

buf.insert(d1)
buf.insert(d2)
print("Data inserted. Is full: " + str(buf.isFull()))

# Read data and ids
data, ids = buf.getDataAndIds()
print("IDs in buffer: " + str(ids))
print("Current data:\n" + str(data))

# Test replacement (Phase 2)
new_val = np.array([99.0, 99.0])
target_index = 0
print("Replacing INDEX 0 with new data...")
success = buf.replacement(new_val, target_index)
print("Replacement success: " + str(success))

# Check new state
data, ids = buf.getDataAndIds()
print("Updated IDs: " + str(ids))
print("Updated data:\n" + str(data))