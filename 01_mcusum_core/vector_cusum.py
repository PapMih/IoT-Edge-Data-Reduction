import numpy as np
from numpy.linalg import inv
import math

class VectorCusum:
    # The VectorCusum class implements the Vector Multivariate Cumulative Sum algorithm

    def __init__(self, mean_vector, covariance_matrix, decision_interval, reference_value):
        # Init method initiates the class variables and calculates the inverse covariance matrix

        self.mean_vector = np.array(mean_vector) 
        self.covariance_matrix = np.array(covariance_matrix) 
        self.decision_interval = decision_interval # Decision interval H
        self.reference_value = reference_value # Reference value K

        # Inverse covariance matrix calculation Σ-1
        self.inverse_covariance_matrix = inv(covariance_matrix) 

        # Number of variables calculation
        self.variables_num = mean_vector.size

        self.s_vector = np.zeros(self.variables_num)
        self.length_c = 0.0 
        self.test_statistic = 0.0 # Test statistic Y_n
        self.out_of_control = False # Control variable. Turns true when a change of epoch occurs 
        self.run_length = 0 # Counter from the last reset 

    def reset(self):
        # Reset method resets the variables when an epoch change occurs

        self.s_vector = np.zeros(self.variables_num) 
        self.run_length = 0 
        self.out_of_control = False 

    def update(self, new_sample):
        # Update method processes a new sample and detects statistical shifts in the stream    
        
        s_temp = (new_sample - self.mean_vector) + self.s_vector # Temporary vector calculation s_temp = (x_n - μ) + s_{n-1}
        self.length_c = math.sqrt((s_temp.transpose())@self.inverse_covariance_matrix@s_temp) 
        self.run_length = self.run_length + 1 

        # Check if length C_n is within the reference value K
        if self.length_c <= self.reference_value :
            self.s_vector = np.zeros(self.variables_num)
            self.test_statistic = 0.0 # Test statistic Y_n reset
            self.out_of_control = False
        
        else:
            self.s_vector = (s_temp) * (1 - self.reference_value / self.length_c) 
            self.test_statistic = self.length_c - self.reference_value # Test statistic Y_n calculation

            #print(f"Index: {self.run_length}, Test Statistic (Y_n): {self.test_statistic}")

            # Check if Y_n exceeds decision interval H
            if self.test_statistic > self.decision_interval:
                self.out_of_control = True # Epoch change signal
            else:
                self.out_of_control = False
        
        # print("Y_n" + str(self.run_length) + ":" + str(round(self.test_statistic, 2)))
        # print("S_n" + str(self.run_length) + ":" + str(np.round(self.s_vector, 2)))

        return self.out_of_control