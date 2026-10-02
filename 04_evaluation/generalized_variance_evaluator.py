import numpy as np
import pandas as pd

class GeneralizedVarianceEvaluator:
    # Initialization of the Evaluator with the original and synopsis datasets
    def __init__(self, original_dataset_df, synopsis_df):
        self.original_dataset_df = original_dataset_df
        self.synopsis_df = synopsis_df
        
        # Selected variables for the multivariate Data Stream
        self.feature_columns = [
            'SPEEDKNOTS', 'MERPM', 'MEPOWER', 
            'RANGEX', 'RANGEY', 'INCLINOMETERXZC', 'INCLINOMETERYZC'
        ]

    # Calculation of the generalized variance (|Σ|) in the n-dimensional space
    def calculate_generalized_variance(self, data_matrix):
        cov_matrix = np.cov(data_matrix, rowvar=False) # Calculate the covariance matrix (Σ).
        generalized_variance = np.linalg.det(cov_matrix)   # Calculation of the generalized variance as the determinant of the covariance matrix
        return generalized_variance

    # Execution of the generalized variance comparison method across all Epochs
    def evaluate_epochs(self):
        epochs = self.synopsis_df['Epoch_ID'].unique() # Extraction of the unique Epoch IDs from the Buffer synopsis
        variance_ratios = [] # List of the epochs with zero generalized variance
        

        zero_gv_epochs = 0 # Keep track of how many epochs have zero volume (problematic ones)
        total_epochs = len(epochs) # Total number of epochs in the synopsis file
        zero_gv_epoch_ids = [] 

        # Loop through each Epoch to calculate the specific generalized variance ratio
        for epoch_id in epochs:
            # Isolation of the synopsis records for the current Epoch
            epoch_synopsis = self.synopsis_df[self.synopsis_df['Epoch_ID'] == epoch_id]
            
            # Retrieval of the global start and end indices
            start_idx = int(epoch_synopsis['Start_Index'].iloc[0])
            end_idx = int(epoch_synopsis['End_Index'].iloc[0])
            
            # Isolation of the corresponding measurements from the original dataset
            # (+1 is added to ensure the end_idx is included in the Pandas slice)
            epoch_dataset = self.original_dataset_df.iloc[start_idx : end_idx + 1]
            
            # Extraction of the n-dimensional feature columns as NumPy arrays
            synopsis_matrix = epoch_synopsis[self.feature_columns].values
            dataset_matrix = epoch_dataset[self.feature_columns].values
            
            # Calculation of the generalized variances (|Σ|) for the synopsis and the original dataset
            gv_synopsis = self.calculate_generalized_variance(synopsis_matrix)
            gv_dataset = self.calculate_generalized_variance(dataset_matrix)

            # Calculation of the generalized variance ratio for the current Epoch
            if gv_dataset == 0: # Check if original volume is zero
                zero_gv_epochs += 1 
                zero_gv_epoch_ids.append(epoch_id)
                if gv_synopsis == 0:
                    ratio = 1.0 # Both are zero thus, synopsis keeps the property
                else:
                    ratio = np.nan # The Original is zero but synopsis is not, set nan
            else:
                ratio = gv_synopsis / gv_dataset  # The normal case
                
            variance_ratios.append(ratio)

        # Save stats so glue code can check for problematic epochs 
        self.zero_gv_epochs = zero_gv_epochs
        self.total_epochs = total_epochs
        self.zero_gv_epoch_ids = zero_gv_epoch_ids 

        # Calculation of the overall Sampling Quality Score based on the generalized variance ratio
        final_ratio = np.mean(variance_ratios)

        # Return the final score rounded to two decimal places
        return round(final_ratio, 2)
    
    # Main orchestration method to execute the evaluation process
    def run(self):
        # Execution of the internal evaluation method
        final_R_score = self.evaluate_epochs()
        
        return (final_R_score, self.zero_gv_epochs, self.zero_gv_epoch_ids, self.total_epochs)