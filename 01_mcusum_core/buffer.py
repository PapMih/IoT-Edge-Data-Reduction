import numpy as np

class DataBuffer:
    # The DataBuffer class implements the Buffer algorithm

    def __init__(self, size, variables_num):
        # Init method initiates the class variables and allocates memory

        self.size = size 
        self.data = np.zeros((size, variables_num), dtype=np.float32) # Initialize data array with zeros (size × variables_num)
        self.samples_ids = np.zeros(size, dtype=int) # Initialize samples_ids array with zeros (size × 1)
        self.global_indices = np.zeros(size, dtype=int) # Initialize the indices array
        self.total_samples_counter = 0
        self.is_full = False	

    def reset(self):
        # reset method resets the variables when an epoch change occurs

        self.data.fill(0) 
        self.samples_ids.fill(0) 
        self.global_indices.fill(0)
        self.total_samples_counter = 0
        self.is_full = False	

    def getDataAndIds(self):
         # getDataAndIds method returns read-only views of the data and sample_ids_read_only arrays.
        
        data_read_only = self.data.view()
        samples_ids_read_only = self.samples_ids.view()

        data_read_only.setflags(write=False)
        samples_ids_read_only.setflags(write=False)

        return data_read_only, samples_ids_read_only
    
    def getGlobalIndices(self):
        # Gets the indices array

        global_indices_read_only = self.global_indices.view()
        global_indices_read_only.setflags(write=False)
        return global_indices_read_only
    
    def replacement (self,new_data, old_data_index, global_index):
        # Replacement method implements Phase 2: In-place replacement

        if old_data_index < 0 or old_data_index >= self.size:
            return False # Index out of bounds
        else:
            self.data[old_data_index] = new_data
            # Assign new unique ID to maintain chronological info
            self.samples_ids[old_data_index] = self.total_samples_counter
            self.global_indices[old_data_index] = global_index
            self.total_samples_counter = self.total_samples_counter + 1
            return True

    def insert(self, new_data, global_index):
        # insert method implements Phase 1: Sequential filling

        self.data[self.total_samples_counter] = new_data
        self.samples_ids[self.total_samples_counter] = self.total_samples_counter
        self.global_indices[self.total_samples_counter] = global_index
        self.total_samples_counter = self.total_samples_counter + 1

        # Check if the buffer is full to signal Phase 2 transition
        if self.total_samples_counter >= self.size:
            self.is_full = True

    def isFull(self):
        # isFull method returns the status of the buffer
        
        return self.is_full