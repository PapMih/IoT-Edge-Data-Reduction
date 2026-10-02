import os
import glob
import pandas as pd

from compression_ratio_evaluator import CompressionRatioEvaluator
from volume_comparison_evaluator import VolumeComparisonEvaluator
from generalized_variance_evaluator import GeneralizedVarianceEvaluator
from pateF1_evaluator import PateF1Evaluator

# Determine file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
mcusum_outputs_dir = os.path.join(parent_dir, 'mcusum_outputs')
synthetic_data_dir = os.path.join(parent_dir, 'synthetic_data')

# List of replacement strategies
replacement_methods = [
    'MahalanobisRangeReplacement',
    'MaxDistanceExpansionReplacement',
    'MaxRangeReplacement'
]

# List to store all evaluation results
all_evaluation_results = []

# List to store records of problematic files
problematic_records = []

# Feature columns used for multivariate analysis
feature_columns = [
    'SPEEDKNOTS', 'MERPM', 'MEPOWER', 
    'RANGEX', 'RANGEY', 'INCLINOMETERXZC', 'INCLINOMETERYZC'
]

# Drift points and buffer zone for PATE-F1 evaluation
drifts = [
    500, 1000, 2000, 3000, 3400, 3700, 4000, 6000, 6800, 7500, 
    8000, 11000, 12000, 13000, 14000, 14800, 15400, 15800, 
    16000, 18000, 19500, 21000, 22500, 24000
]

buffer_zone = 12

print("Beginning of calculations for the 10 datasets...")

# LOOP 1: Iterate over each replacement method 
for method in replacement_methods:
    method_dir = os.path.join(mcusum_outputs_dir, method)
    print(f"Method: {method}")   
    
    # LOOP 2: Iterate over each dataset 
    for dataset_id in range(1, 11):
        dataset_outputs_folder = os.path.join(method_dir, f"Synthetic ship dataset {dataset_id}")

        # Load the corresponding original dataset
        original_data_path = os.path.join(synthetic_data_dir, f"synthetic_ship_dataset_{dataset_id}.xlsx")
        original_df = pd.read_excel(original_data_path)

        # Find all synopsis CSV files for various h, k, size parameters
        synopsis_files = glob.glob(os.path.join(dataset_outputs_folder, "synopsis__*.csv"))

        for syn_file in synopsis_files:
            filename = os.path.basename(syn_file)

            # Extract parameters from filename
            parts = filename.replace('.csv', '').split('__')
            h_val = float(parts[1].split('_')[1])
            k_val = float(parts[2].split('_')[1])
            size_val = int(parts[3].split('_')[1])

            synopsis_df = pd.read_csv(syn_file)

            # 1. Calculate Compression Ratio (C)
            c_score = CompressionRatioEvaluator(original_df, synopsis_df).run()

            # 2. Calculate Volume Ratio (R)
            vol_evaluator = VolumeComparisonEvaluator(original_df, synopsis_df)
            r_score, r_zero_epochs, r_zero_epoch_ids, r_total_epochs = vol_evaluator.run()

            # 3. Calculate Generalized Variance Ratio (S) - Raw value
            gv_evaluator = GeneralizedVarianceEvaluator(original_df, synopsis_df)
            s_score, s_zero_epochs, s_zero_epoch_ids, s_total_epochs = gv_evaluator.run()

            # Check for problematic epochs (zero volume or zero generalized variance)
            if r_zero_epochs > 0 or s_zero_epochs > 0:
                problematic_records.append({
                    'Method': method,
                    'Dataset_ID': dataset_id,
                    'h': h_val,
                    'k': k_val,
                    'Size': size_val,
                    'Volume_Zero_Epochs': r_zero_epochs,
                    'GV_Zero_Epochs': s_zero_epochs,
                    'Total_Epochs': r_total_epochs,
                    'Volume_Zero_Epoch_IDs': ','.join(map(str, r_zero_epoch_ids)),
                    'GV_Zero_Epoch_IDs': ','.join(map(str, s_zero_epoch_ids))
                })  

            # 4. Calculate PATE-F1, True & False Alarms
            pate_results = PateF1Evaluator(synopsis_df).run()

            f1_score = pate_results.get("F1_Score", 0.0)
            count_TP = pate_results.get("True_Alarms", 0)
            count_FP = pate_results.get("False_Alarms", 0)

            # Store raw results for current run
            all_evaluation_results.append({
                'Method': method,
                'Dataset_ID': dataset_id,
                'h': h_val,
                'k': k_val,
                'Size': size_val,
                'Compression_Ratio_C': c_score,
                'Volume_Ratio_R': r_score,
                'Gen_Variance_Ratio_S': s_score,
                'True_Alarms': count_TP,
                'False_Alarms': count_FP,
                'F1_Score': f1_score
            })

# 5. Export averages over the 10 datasets and save 
results_df = pd.DataFrame(all_evaluation_results)

# Group by parameters and calculate the mean for all metrics (the final 120 combinations)
summary_df = results_df.groupby(['Method', 'h', 'k', 'Size']).mean(numeric_only=True).reset_index()

# Apply Min-Max Normalization to create S' from the averaged S values
s_min = summary_df['Gen_Variance_Ratio_S'].min()
s_max = summary_df['Gen_Variance_Ratio_S'].max()
    
summary_df["Gen_Variance_Ratio_S'"] = (summary_df['Gen_Variance_Ratio_S'] - s_min) / (s_max - s_min)

# Compute combined indices using the normalized S' values
summary_df['Sampling_Quality_SQ'] = (summary_df['Volume_Ratio_R'] + summary_df["Gen_Variance_Ratio_S'"]) / 2.0
summary_df['Statistical_Efficiency_SE'] = (summary_df['Volume_Ratio_R'] + summary_df["Gen_Variance_Ratio_S'"] + summary_df['Compression_Ratio_C']) / 3.0
summary_df['Total_Performance_TP'] = (summary_df['Compression_Ratio_C'] + summary_df['Volume_Ratio_R'] + summary_df["Gen_Variance_Ratio_S'"] + summary_df['F1_Score']) / 4.0

# Select and order columns for the final report
output_columns = [
    'Method', 'h', 'k', 'Size', 
    'Compression_Ratio_C', 'Volume_Ratio_R', 'Gen_Variance_Ratio_S', "Gen_Variance_Ratio_S'",
    'Sampling_Quality_SQ', 'Statistical_Efficiency_SE', 'Total_Performance_TP',
    'True_Alarms', 'False_Alarms', 'F1_Score'
]
summary_df = summary_df[output_columns]

# Save the summary to Excel file
final_output_path = os.path.join(script_dir, 'combined_evaluation_summary.xlsx')
summary_df.to_excel(final_output_path, index=False, engine='openpyxl')

# Save problematic files to a separate Excel file
problematic_df = pd.DataFrame(problematic_records)
problematic_path = os.path.join(script_dir, 'problematic_epochs.xlsx')
problematic_df.to_excel(problematic_path, index=False, engine='openpyxl')

print(f"\nThe process completed successfully!")
print(f"The results (average of 10 datasets) were saved to: {final_output_path}")
print(f"Problematic files saved to: {problematic_path}")