import numpy as np
import pandas as pd

class VolumeComparisonEvaluator:

    # Initialization of the Evaluator with the original and synopsis datasets
    def __init__(self, original_dataset_df, synopsis_df):
        self.original_dataset_df = original_dataset_df
        self.synopsis_df = synopsis_df
        
        # Selected variables for the multivariate Data Stream
        self.feature_columns = [
            'SPEEDKNOTS', 'MERPM', 'MEPOWER', 
            'RANGEX', 'RANGEY', 'INCLINOMETERXZC', 'INCLINOMETERYZC'
        ]

    # Calculation of the bounding box volume in the n-dimensional space
    def calculate_bounding_box_volume(self, data_matrix):
        # Determination of the minimum and maximum values for each dimension
        mins = np.min(data_matrix, axis=0)
        maxs = np.max(data_matrix, axis=0)
        
        
        ranges = maxs - mins # Calculation of the ranges for each variable
        volume = np.prod(ranges) # Calculation of the total volume as the product of all ranges
        
        return volume

    # Execution of the volume comparison method across all Epochs
    def evaluate_epochs(self):
        
        epochs = self.synopsis_df['Epoch_ID'].unique() # Extraction of the unique Epoch IDs from the Buffer synopsis
        volume_ratios = []

        zero_volume_epochs = 0 # Keep track of how many epochs have zero volume (problematic ones)
        total_epochs = len(epochs) # Total number of epochs in the synopsis file
        zero_volume_epoch_ids = [] # List of the epochs with zero volume 

        # Loop through each Epoch to calculate the specific volume ratio
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
            
            # Calculation of the volumes for the synopsis and the original dataset
            v_synopsis = self.calculate_bounding_box_volume(synopsis_matrix)
            v_dataset = self.calculate_bounding_box_volume(dataset_matrix)
           
            # Calculation of the volume ratio for the current Epoch
            if v_dataset == 0: # Check if original volume is zero
                zero_volume_epochs += 1
                zero_volume_epoch_ids.append(epoch_id)
                if v_synopsis == 0: 
                    ratio = 1.0 # Both are zero thus, synopsis keeps the property
                else:
                    ratio = np.nan # The Original is zero but synopsis is not, set nan
            else:
                ratio = v_synopsis / v_dataset # The normal case

            volume_ratios.append(ratio)

        # Save stats so glue code can check for problematic epochs    
        self.zero_volume_epochs = zero_volume_epochs
        self.total_epochs = total_epochs
        self.zero_volume_epoch_ids = zero_volume_epoch_ids

        # Calculation of the overall Sampling Quality Score based on the volume ratio
        final_ratio = np.mean(volume_ratios)

        # Return the final score rounded to two decimal places
        return round(final_ratio, 2)
    
    # Main orchestration method to execute the evaluation process
    def run(self):
        # Execution of the internal evaluation method
        final_R_score = self.evaluate_epochs()
        
        return (final_R_score, self.zero_volume_epochs, self.zero_volume_epoch_ids, self.total_epochs)