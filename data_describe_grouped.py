import pandas as pd

# Load the Excel file
file_path = '/Users/jessedesimone/Desktop/Book6.xlsx'  # Update with your actual file path

# Read the sheet into a DataFrame
df = pd.read_excel(file_path)

# Define the continuous variables and the column that defines the groups
continuous_vars = ['UPDRS_I', 'UPDRS_II', 'UPDRS_III', 'UPDRS_TOT', 'EPWORTH_TOT',
'HAMIL_DEPR_TOT', 'MOCA_TOT', 'MOD_SCH_ENG', 'PDQ_39', 'PSP_RS',
'UMSARS_I', 'UMSARS_II']  # Update with your actual variable names
group_column = 'DATASET'  # Update with your actual group column name ('train' and 'test')
dx_column = 'FINAL_DX'  # Column to group by

# Custom function to return mean (SD)
def mean_sd_format(series):
    mean = series.mean()
    std = series.std()
    return f"{mean:.1f} ({std:.1f})"

# Function to summarize data
def summarize_data_grouped(df, continuous_vars, dx_column):
    return df.groupby(dx_column)[continuous_vars].agg(mean_sd_format)

# Summary for all data, grouped by DX
summary_all = summarize_data_grouped(df, continuous_vars, dx_column)

# Summary for train group, grouped by DX
summary_train = summarize_data_grouped(df[df[group_column] == 'train'], continuous_vars, dx_column)

# Summary for test group, grouped by DX
summary_test = summarize_data_grouped(df[df[group_column] == 'test'], continuous_vars, dx_column)

# Output the summaries
print("Summary for All Data:")
print(summary_all)

print("\nSummary for Train Group:")
print(summary_train)

print("\nSummary for Test Group:")
print(summary_test)

# Optionally, you can save the summaries to an Excel file
with pd.ExcelWriter('/Users/jessedesimone/Desktop/summary_output_grouped.xlsx') as writer:
    summary_all.to_excel(writer, sheet_name='All_Data_Grouped')
    summary_train.to_excel(writer, sheet_name='Train_Group_Grouped')
    summary_test.to_excel(writer, sheet_name='Test_Group_Grouped')
