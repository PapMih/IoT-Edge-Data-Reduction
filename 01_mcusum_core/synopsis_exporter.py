import csv

class SynopsisExporter:
    # The SynopsisExporter class makes a csv file with the exported synopses from the reservoir at the end of each Epoch.

    def __init__(self, h_value, k_value, buffer_size):
        # Creates the file name of the csv file and initiates the synopsis list.

        self.file_name = f"synopsis__h_{h_value}__k_{k_value}__size_{buffer_size}.csv"
        self.synopsis = []

    def log_epoch(self, reservoir_data, global_indices, epoch_id, start_index, end_index):
        # Appends the buffer data to the synopsis list, tagged with epoch metadata.

        for i, row in enumerate(reservoir_data):
            # Rounding to 2 decimal places
            rounded_row = [round(float(val), 2) for val in row]
            
            # Combine metadata (epoch_id, start, end) and the multi-dimensional measurement into a single list.
            global_index = int(global_indices[i])
            row_data = [epoch_id, start_index, end_index, global_index] + rounded_row
            self.synopsis.append(row_data)

    def export_results(self, feature_names):
        # Writes the accumulated synopsis data to a CSV file.
        
        headers = ["Epoch_ID", "Start_Index", "End_Index", "Measurement_Index"] + feature_names
        
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(self.synopsis)
            
        print(f"Successfully exported {len(self.synopsis)} reduced data points to {self.file_name}")