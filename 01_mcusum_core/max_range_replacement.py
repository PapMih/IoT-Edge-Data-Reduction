import numpy as np
from numpy.linalg import norm

class MaxRangeReplacement:
    # The MaxRangeReplacement class implements the Max Range Replacement Strategy algorithm

    def __init__(self, buffer, samples_ids):
        # Init method initiates the class variables

        self.buffer = buffer
        self.samples_ids = samples_ids

        # Calculate the mean vector across variables (axis=0) to represent M1
        self.mean_vector = np.mean(self.buffer, axis=0)

        # Calculate distances
        self.distances = self.distance_calc(self.buffer, self.mean_vector)

        # Retrieve IDs and max distance 
        self.min_distance_id, self.max_distance_id, self.max_distance = self.min_max_calc(self.distances, self.samples_ids)
    
    def distance_calc(self, buffer, mean_vector):
        # Calculate Euclidean distances (di) for all points 
        
        distances = np.linalg.norm(buffer - mean_vector, axis=1)
        
        return distances

    def min_max_calc(self, distances, samples_ids):
        # Calculation of the max distance and the min and max distance ids

        # Identify the indices for the minimum (dmin) and maximum (dmax) distances
        self.min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)
        
        # Extract the corresponding IDs and maximum distance
        min_distance_id = samples_ids[self.min_idx]
        max_distance_id = samples_ids[max_idx]
        max_distance = distances[max_idx]
        
        return min_distance_id, max_distance_id, max_distance
    
    def check_new_measurement(self, new_measurement):
        #The method checks if the new measurement expands the cluster
        
        # Get the number of samples (N) currently in the buffer
        n_samples = self.buffer.shape[0]

        # Retrieve the actual vector to be replaced (d_min) using its index
        min_vector = self.buffer[self.min_idx]

        # Calculate the new mean vector to represent M2
        temp_mean_vector = self.mean_vector + (new_measurement - min_vector) / n_samples

        # Calculate the distance of all existing points to the temp mean (M2)
        temp_distances = self.distance_calc(self.buffer, temp_mean_vector)
        
        # Correct the distance for the newly inserted point which replaces d_min
        new_point_distance = np.linalg.norm(new_measurement - temp_mean_vector)
        temp_distances[self.min_idx] = new_point_distance

        # Find the new maximum distance (d_max')
        new_max_distance = np.max(temp_distances)

        # Evaluate the replacement criterion (d_max' > d_max)
        if new_max_distance > self.max_distance:
            # Update method's variables
            self.mean_vector = temp_mean_vector
            self.distances = temp_distances
        #    print("New distances array: " + str(self.distances)) #For test
            return True, self.min_idx #Returns true and the index of the measurement that will be replaced
        else:
            return False, None
        
    def sync_state(self):
        # Syncs the state
        
        self.min_distance_id, self.max_distance_id, self.max_distance = self.min_max_calc(self.distances, self.samples_ids)