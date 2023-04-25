#!/usr/local/bin/python3.9

#import packages
import pandas as pd

#read data
df = pd.read_csv('/path/to/csv')

#subset the dataframe
df_check=df.filter(regex='var1|var2|var n')

#describe data
df_check.groupby(['<grouping var>']).describe()

#permutation test for single variable


#permutation tests for multiple variables



#FDR correction for multivariate permutation tests