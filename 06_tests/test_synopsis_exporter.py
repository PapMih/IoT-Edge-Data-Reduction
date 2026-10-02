import numpy as np
from synopsis_exporter import SynopsisExporter

# Test the SynopsisExporter class

h = 5.5
k = 0.5
size = 10
run_id = 1

# Initialize the exporter
exporter = SynopsisExporter(h, k, size, run_id)

# Create mock reservoir data (e.g. 2 features)
data_epoch_1 = np.array([
    [10.5, 45.1],
    [10.8, 45.3]
])

data_epoch_2 = np.array([
    [20.1, 80.5],
    [20.3, 80.7]
])

# Log epochs
print("Logging epoch 1...")
exporter.log_epoch(data_epoch_1, 1, 0, 100)

print("Logging epoch 2...")
exporter.log_epoch(data_epoch_2, 2, 101, 250)

# Export to csv
feature_names = ['MEPOWER', 'SPEEDKNOTS']
exporter.export_results(feature_names)

print("Export finished. Check the generated file.")