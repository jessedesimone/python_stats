import pandas as pd
import numpy as np
from scipy.stats import sem, t
import os

os.chdir('/Users/jessedesimone/Desktop')

# Load your DataFrame
df = pd.read_excel('roi_input_PD.xlsx')  

# Check if the DataFrame is loaded
if df is None or df.empty:
    raise ValueError("The DataFrame is empty or could not be loaded.")

# Drop unnecessary columns
cols_to_drop = ['Subject', 'Scanner', 'Site', 'Age', 'Sex', 'UPDRS']
df2 = df.drop(columns=cols_to_drop, errors='ignore') 

# Ensure GroupDir exists
group_column = 'GroupDir'
if group_column not in df2.columns:
    raise ValueError(f"The column '{group_column}' is not in the DataFrame.")

# Calculate statistics for each ROI

# Confidence level
confidence = 0.95

# Function to calculate statistics
def calculate_statistics(group):
    stats = {}
    for col in group.columns:
        if col == group_column:  # Skip the grouping column
            continue
        data = group[col].dropna()
        n = len(data)
        mean = np.mean(data)
        sd = np.std(data, ddof=1)
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        margin_error = sem(data) * t.ppf((1 + confidence) / 2, n - 1) if n > 1 else np.nan
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        stats[col] = [mean, sd, median, iqr, margin_error, ci_lower, ci_upper]
    return stats

# Apply statistics calculation for each group
grouped = df2.groupby(group_column)
results = {}

for name, group in grouped:
    group_stats = calculate_statistics(group)
    results[name] = pd.DataFrame(
        group_stats, 
        index=['Mean', 'SD', 'Median', 'IQR', 'Margin of Error', 'CI Lower', 'CI Upper']
    ).T

# Combine results into a single DataFrame
final_result = pd.concat(results, keys=results.keys(), names=[group_column, 'Variable'])

# Save to Excel
final_result.to_excel('PD_Group_Statistics.xlsx', sheet_name='Statistics')


# Perform paired samples t-tests
# Separate the data for GroupDir=1 and GroupDir=2
group1 = df2[df2[group_column] == 1].drop(columns=[group_column])
group2 = df2[df2[group_column] == 2].drop(columns=[group_column])

# Perform paired samples t-test for each column
from scipy.stats import ttest_rel
results = []
for col in group1.columns:
    stat, p_value = ttest_rel(group1[col].dropna(), group2[col].dropna())
    results.append({'Variable': col, 'T-Statistic': stat, 'P-Value': p_value})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save the results to Excel
results_df.to_excel('PD_Paired_TTest_Results.xlsx', index=False)