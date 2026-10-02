import os
import pandas as pd

# Determine paths
script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(script_dir, '../..'))  
eval_dir = os.path.join(project_root, '04_evaluation')

# Input file 
input_file = os.path.join(eval_dir, 'combined_evaluation_summary.xlsx')

# Output file
output_file = os.path.join(script_dir, 'summary_stats_C.xlsx')

# Read the data
df = pd.read_excel(input_file, engine='openpyxl')

# Calculate Total Epochs based on True and False alarms + 1
df['Total_Epochs'] = df['True_Alarms'] + df['False_Alarms'] + 1

# Filter out duplicate method rows to avoid statistical redundancy (std skew)
first_method = df['Method'].unique()[0]
df_filtered = df[df['Method'] == first_method].copy()

# Create combined parameter (k, h) because they are confounded
df_filtered['k_h'] = df_filtered['k'].astype(str) + '_' + df_filtered['h'].astype(str)

# Define parameters to analyze (Method is removed)
params = ['k_h', 'Size']

# Define columns and their specific aggregations
agg_funcs = {
    'Compression_Ratio_C': ['mean', 'std', 'min', 'max'],
    'Total_Epochs': ['mean']
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

print(f"Summary statistics for Compression Ratio (C) and Mean Epochs saved to: {output_file}")