#!/usr/local/bin/python3.9
'''
Module for descriptive statistics
'''
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, skewnorm


'''
Plot PDF and distrubution histogram for single variable (data)
can replace "data" with column from dataframe
'''

# Generate random normal sample
data = np.random.normal(0, 10, 1000)

# Probability density function
f, ax1 = plt.subplots()
ax1.hist(data, bins='auto')
ax1.set_title('probability density (random)')
plt.tight_layout()

# Plot distribution
plt.figure(figsize=(9,5))
ax=sns.distplot(data, kde=False)
#create vertical line on mean
plt.axvline(np.mean(data), color="black", label="mean")
#create vertical lines to depict empirical distribution
#1 SD left and right of mean (68% of data)
plt.axvline(np.mean(data) + np.std(data), color = "blue", linestyle="dashed", label="+/-68%")
plt.axvline(np.mean(data) - np.std(data), color = "blue", linestyle="dashed")
#2 SD left and right of mean (95% of data)
plt.axvline(np.mean(data) + (np.std(data) * 2), color = "red", linestyle="dashed", label="+/-95%")
plt.axvline(np.mean(data) - (np.std(data) * 2), color = "red", linestyle="dashed")
#3 SD left and right of mean (99.7% of data)
plt.axvline(np.mean(data) + (np.std(data) * 3), color = "green", linestyle="dashed", label="+/-97.9%")
plt.axvline(np.mean(data) - (np.std(data) * 3), color = "green", linestyle="dashed")
plt.legend()
plt.title("Histogram of observations")
plt.xlabel("Observed value")
#plt.ylabel("KDE")                      #use if kde=True
plt.ylabel("Observation frequency")     #use if kde=False

# Descriptive statistics
data = [10,291,87,838,190]
np.mean(data)       #mean
np.median(data)     #median
stats.mode(data)    #mode
np.std(data)        #SD
stats.sem(data)     #SEM¸
np.var(data)        #variance
max(data)-min(data)     #range


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