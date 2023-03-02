#!/usr/bin/env python3
'''
perform boxcox transformation on all specified columns in dataframe
create new dataframe columns containing the transformed data


'''
# import packages
import pandas as pd
from scipy import stats
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns

# read data
df=pd.read_csv('<path/to/infile')

loop_cols = df.loc[:, ~df.columns.isin(['grp', 'id_subj'])]         #do not transform these columns
#loop_cols = ['col1','col2']            #transform these columns
'''perform boxcox transform on specified columns'''
for col in loop_cols:   
    fitted_data, fitted_lambda = stats.boxcox(df[col])
    df[col + '_bxcx'] = fitted_data
'''retain only transformed cols'''
df=df.filter(regex='grp|id_subj|bxcx')
'''plot kde again to compare'''
df_check=df.drop(['grp', 'id_subj'], axis=1)
fig, axes = plt.subplots(ncols=len(df_check.columns), figsize=(20,8))
for ax, col in zip(axes, df_check.columns):
  sns.histplot(df_check[col], ax=ax, kde=True)
  plt.tight_layout() 
plt.show()