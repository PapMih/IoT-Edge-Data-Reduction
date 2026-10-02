import numpy as np
from numpy.linalg import norm

class MaxDistanceExpansionReplacement:
    # The MaxDistanceExpansionReplacement class implements a distance-based replacement strategy that expands the overall spatial cluster

    def __init__(self, buffer, samples_ids):
        # Init method initiates the class variables

        self.buffer = buffer
        self.samples_ids = samples_ids

        # Perform the initial calculation of distances and states
        self.sync_state()

    def distance_calc(self, buffer, reference_vector):
        # Calculate Euclidean distances for all points to a reference vector
        
        distances = np.linalg.norm(buffer - reference_vector, axis=1)
        
        return distances

    def check_new_measurement(self, new_measurement):
        # The method checks if the new measurement expands the cluster
        
        # Calculate the distance of the new measurement to all existing points
        temp_distances = self.distance_calc(self.buffer, new_measurement)
        
        # Find the new maximum distance (d_max') and the index it connects to
        new_max_distance = np.max(temp_distances)
        new_max_idx = np.argmax(temp_distances)
        
        # Evaluate the replacement criterion (d_max' > d_max)
        if new_max_distance > self.max_distance:
            # If the new point creates the max distance WITH one of the min points, keep that min point!
            if new_max_idx == self.min_idx_1:
                replace_idx = self.min_idx_2
            else:
                replace_idx = self.min_idx_1
                
            return True, replace_idx # Returns true and the index of the measurement to be replaced
        else:
            return False, None
            
    def sync_state(self):
        # Syncs the state by completely recalculating all distances from scratch 
        
        # Get the number of samples (N)
        n = self.buffer.shape[0]
        
        # Initialize the NxN distance matrix with zeros
        pairwise_distances = np.zeros((n, n))
        
        # Iterate to calculate distance of each point (i) from every other point (j)
        for i in range(n):
            for j in range(n):
                # Compute the Euclidean distance (norm)
                diff = self.buffer[i] - self.buffer[j]
                pairwise_distances[i, j] = np.linalg.norm(diff)
                
        # Find Maximum Distance 
        np.fill_diagonal(pairwise_distances, 0.0) # Ignore self-distances (0.0) on the diagonal for max calculation
        self.max_distance = np.max(pairwise_distances)

        # Find Minimum Distance Pair 
        current_min_distance = float('inf') # Initialization
        min_index_1 = -1 # Initialization
        min_index_2 = -1 # Initialization
        
        # Iterate through the matrix to find the absolute minimum distance pair
        for i in range(n):
            for j in range(i + 1, n): # We start j from (i + 1) to only check the upper triangle of the matrix.
                if pairwise_distances[i, j] < current_min_distance:
                    current_min_distance = pairwise_distances[i, j]
                    min_index_1 = i
                    min_index_2 = j
                    
        # Save the indices of the closest pair found
        self.min_idx_1 = min_index_1
        self.min_idx_2 = min_index_2