import numpy as np
import pandas as pd

class CompressionRatioEvaluator:
    # Initialization of the Evaluator with the original and synopsis datasets
    def __init__(self, original_dataset_df, synopsis_df):
        self.original_dataset_df = original_dataset_df
        self.synopsis_df = synopsis_df

    # Calculation of the global compression ratio across the entire data stream
    def _evaluate_compression(self):
        # Determination of the total number of measurements in the original data stream (N)
        N = len(self.original_dataset_df)
        
        # Determination of the total number of synopsis elements flushed from the Buffer (M)
        M = len(self.synopsis_df)
        
        # Calculation of the Compression Ratio (CR)
        compression_ratio = 1.0 - (M / N)
        
        return compression_ratio
    
    # Main orchestration method to execute the evaluation process
    def run(self):
        # Execution of the internal evaluation method
        final_CR_score = self._evaluate_compression()
        
        # Return the final score rounded to two decimal places
        return round(final_CR_score, 2)