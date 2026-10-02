from vector_cusum import VectorCusum
import numpy as np


#Crosier example
data = np.loadtxt('Crosier_example_data.csv', delimiter = ',', skiprows = 1)
print(data) 
mean_vector = np.array([0,0])
print("Mean vector:" + str(mean_vector))
covariance_matrix = np.array([[1, 0.5], [0.5, 1]])
print("Covariance matrix:" + str(covariance_matrix))
decision_interval = 5.5
reference_value = 0.5
cusum = VectorCusum(mean_vector,covariance_matrix, decision_interval, reference_value )

for i in data:
    print(cusum.update(i))
