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

#----------dataframe loop option 2----------
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
df_agg = pd.DataFrame(df.groupby('grp_id').agg(['mean','sem']))
df_agg=df_agg.reset_index()
df_agg.to_csv('/Users/jessedesimone/Desktop/test/test_out.csv', index=False)