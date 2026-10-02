import numpy as np

class CovMeanCalculation:
    # The CovMeanCalculation class implements the calculation of covariance and mean matrices directly from an in-memory buffer

    def __init__(self):
        # Init method initializes the class as a stateless entity to minimize resource allocation in Edge environments
        pass

    def differential_calc(self, buffer_data):
        return np.diff(buffer_data, axis=0)  # Calculation of consecutive differences

    def load_cov_mean(self, data):

        # Vectorized calculation of the mean vector and the covariance matrix
        mean_vector = np.mean(data, axis=0)
        covariance_matrix = np.cov(data, rowvar=False)

        # Addition of a small epsilon to the diagonal to ensure mathematical stability during matrix inversion
        epsilon = 1e-4
        np.fill_diagonal(covariance_matrix, covariance_matrix.diagonal() + epsilon)

        #print(np.round(covariance_matrix, 4))

        return mean_vector, covariance_matrix