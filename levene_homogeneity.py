#!/usr/bin/env python3

'''perform Levene's test for homogeneity of variance for each column specified in dataframe'''

# import packages
import pandas as pd
from scipy import stats
from scipy.stats import levene

# read data
df=pd.read_csv('<path/to/infile')

# levene test for homogeneity of variance
df_check=df.drop(['col1', 'col2'], axis=1)      #drop specified columns from data
loop_cols = df_check.loc[:, df_check.columns!='grp']
col_names = [col for col in loop_cols]
for col in col_names:
    low=df_check.query('grp == "low_risk"')[col]
    high=df_check.query('grp == "high_risk"')[col]
    stat, p = levene(low, high)
    print(col)
    print('Levene test: Statistic=%.3f, P-Value=%.5f' % (stat, p))
    if p < 0.05:
        print('** violates variance homogeneity assumption **')
    print('\n')