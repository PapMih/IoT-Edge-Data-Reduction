import os
import pandas as pd

# Determine paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))  
eval_dir = os.path.join(project_root, '04_evaluation')

# Input file 
input_file = os.path.join(eval_dir, 'combined_evaluation_summary.xlsx')

# Output file
output_file = os.path.join(script_dir, 'summary_stats_SQ.xlsx')

# Read the data
df = pd.read_excel(input_file, engine='openpyxl')

# Create combined parameter (k, h) because they are confounded
df['k_h'] = df['k'].astype(str) + '_' + df['h'].astype(str)

# Define parameters to analyze (k and h are now combined)
params = ['Method', 'k_h', 'Size']

# Analyze Sampling_Quality_SQ
metric_cols = ['Sampling_Quality_SQ']

# Write results to Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for param in params:
        grouped = df.groupby(param)[metric_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
        grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]
        counts = df.groupby(param).size().reset_index(name='count')
        grouped = grouped.merge(counts, on=param)
        sheet_name = param[:31]
        grouped.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Summary statistics for Sampling_Quality_SQ saved to: {output_file}")