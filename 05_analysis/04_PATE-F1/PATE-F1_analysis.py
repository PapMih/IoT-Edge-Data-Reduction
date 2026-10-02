import os
import pandas as pd

# Determine paths
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(script_dir, '../..'))  
eval_dir = os.path.join(project_root, '04_evaluation')

# Input file 
input_file = os.path.join(eval_dir, 'combined_evaluation_summary.xlsx')

# Output file
output_file = os.path.join(script_dir, 'summary_stats_F1.xlsx')

# Read the data
df = pd.read_excel(input_file, engine='openpyxl')

# Filter out duplicate method rows to avoid statistical redundancy (std skew)
# F1_Score, True_Alarms, and False_Alarms are identical across replacement methods
first_method = df['Method'].unique()[0]
df_filtered = df[df['Method'] == first_method].copy()

# Create combined parameter (k, h) because they are confounded
df_filtered['k_h'] = df_filtered['k'].astype(str) + '_' + df_filtered['h'].astype(str)

# Define parameters to analyze (Method is removed)
params = ['k_h', 'Size']

# Define columns and their specific aggregations
agg_funcs = {
    'F1_Score': ['mean', 'std', 'min', 'max'],
    'True_Alarms': ['mean'],
    'False_Alarms': ['mean']
}

# Write results to Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for param in params:
        # Group and aggregate using the defined dictionary
        grouped = df_filtered.groupby(param).agg(agg_funcs).reset_index()
        
        # Flatten the MultiIndex columns correctly
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
        
        counts = df_filtered.groupby(param).size().reset_index(name='count')
        grouped = grouped.merge(counts, on=param)
        
        sheet_name = param[:31]
        grouped.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Summary statistics for F1 Score and Alarms saved to: {output_file}")