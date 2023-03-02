#!/usr/bin/env python3

'''perform Shapiro test for normality for each column specified in dataframe'''

# import packages
import pandas as pd
from scipy import stats
from scipy.stats import shapiro

# read data
df=pd.read_csv('<path/to/infile')

# shapiro test for normality
df_check=df.drop(['col1','col2'], axis=1)     #remove unwanted columns
col_names = [col for col in df_check.columns]
for col in col_names:
    stat, p = shapiro(df_check[col])
    print(col)
    print('Shapiro test: Statistic=%.3f, P-Value=%.5f' % (stat, p))
    if p < 0.05:
        print('** violates normality assumption **')
    print('\n')