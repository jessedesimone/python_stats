#!/usr/local/bin/python3.9
#----------dataframe loop option 1----------
'''
pandas and scipy package
This is an option if data contains only a single group
Loop through columns of dataframe
calculate mean and SEM
print out values to new dataframe
save new dataframe as csv table
'''
import os
import pandas as pd
import numpy as np
import statistics
from scipy import stats

path_to_file = ''
file = '{input file}'
infile = os.path.join(path_to_file,file)
df = pd.read_csv(infile)

list_of_cols = list(df.columns)

list_of_mean = []
list_of_sem = []
for column in df.columns[0:]:
    mean = round(df[column].mean(), 5)
    list_of_mean.append(mean)
    sem = round(stats.sem(df[column], nan_policy='omit'),5)
    list_of_sem.append(sem)

d = {'Mean':list_of_mean, 'SEM':list_of_sem}
df = pd.DataFrame(d, index=list_of_cols)
print(df)
outfile=os.path.join(path_to_file,'calculations.csv')
df.to_csv(outfile)

#----------Method 2----------
'''
pandas groupby option
calculate for all specified colums for each group
creates separate dataframes for means and sem
'''
import pandas as pd
path_to_file = '/Users/jessedesimone/Desktop/test'
file = 'test.csv'
infile = os.path.join(path_to_file,file)
df = pd.read_csv(infile)
list_of_cols = list(df.columns[1:])     #define columns to test
df_mean=pd.DataFrame(df.groupby(['grp_id'])[list_of_cols].mean())
df_mean.to_csv('/Users/jessedesimone/Desktop/test/test_out.csv', index=True)
df_sem=pd.DataFrame(df.groupby(['grp_id'])[list_of_cols].sem())
df_sem.to_csv('/Users/jessedesimone/Desktop/test/test_out.csv', index=True)

'''
aggregate mean and sem into single dataframe
'''
df_agg = pd.DataFrame(df.groupby('grp_id')[list_of_cols].agg(['mean','sem']))
df_agg=df_agg.reset_index()
df_agg.to_csv('/Users/jessedesimone/Desktop/test/test_out.csv', index=False)

#---------------------Method 3---------------------
'''
two or more groups
group by analysis to calculate means and sem; then add to summary
'''
# Columns for which to calculate grouped means and standard errors
columns_to_process = list(df.columns[10:])  

grouped_summary = {}

# Loop through each column
for column in columns_to_process:
    #grouped means, sem, summary
    grouped_means = df.groupby('ptau_status_visual_cut')[column].mean()
    grouped_sem = df.groupby('ptau_status_visual_cut')[column].apply(lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    grouped_summary[column] = pd.DataFrame({'Mean': grouped_means, 'Standard_Error_of_Mean': grouped_sem})

# Print the results
for column, summary_df in grouped_summary.items():
    print(f"Summary for {column}:")
    print(summary_df)
    
#---------------------Method 2---------------------
'''
two or more groups
group by analysis to calculate means and sem; then add to dataframe
final dataframe with ROIs as rows and mean/sem as columns for each group - preffered version
'''
#output results as dataframe
columns_to_process = list(df.columns[10:])
df_mean=pd.DataFrame(df.groupby(['ptau_status_visual_cut'])[columns_to_process].mean())
df_sem=pd.DataFrame(df.groupby(['ptau_status_visual_cut'])[columns_to_process].sem())
df_agg = pd.DataFrame(df.groupby('ptau_status_visual_cut')[columns_to_process].agg(['mean','sem']))
pivoted_df_mean = df_mean.pivot_table(index=None, columns='ptau_status_visual_cut', values=columns_to_process)
pivoted_df_sem = df_sem.pivot_table(index=None, columns='ptau_status_visual_cut', values=columns_to_process)
concat_list=[pivoted_df_mean, pivoted_df_sem]
df_concat=pd.concat(concat_list, axis=1)
df_concat=df_concat.reset_index()
df_concat.to_csv('/Users/jessedesimone/Desktop/metrics.csv', index=False)