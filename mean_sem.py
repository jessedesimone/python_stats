#!/usr/local/bin/python3.9
'''
Module will loop through columsn of dataframe
calculate mean and SEM
print out values to new dataframe
save new dataframe as csv table

Change Log
=========
0.0.1 (2021-04-01)
0.0.2 (2021-04-21)
--------
Implemented option 2 for express calculation
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
    mean = round(df[column].mean(), 3)
    list_of_mean.append(mean)
    sem = round(stats.sem(df[column], nan_policy='omit'),3)
    list_of_sem.append(sem)

d = {'Mean':list_of_mean, 'SEM':list_of_sem}
df = pd.DataFrame(d, index=list_of_cols)
print(df)
outfile=os.path.join(path_to_file,'calculations.csv')
df.to_csv(outfile)



# Option 2 - no DataFrame
values = [10,116,96,327,147]
mean = round(np.mean(values), 3)
sem = round(stats.sem(values), 3)
print ('Mean: ', mean, '::     SEM', sem)