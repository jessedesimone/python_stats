#!/usr/bin/env python3
'''create kde subplots for each specified column in dataframe'''

#import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read data
df=pd.read_csv('<path/to/infile')

# distribution plots for dataframe
df_check=df.drop(['col1','col2'], axis=1)        #drop specified columns from data
fig, axes = plt.subplots(ncols=len(df_check.columns), figsize=(10,5))
for ax, col in zip(axes, df_check.columns):
  sns.histplot(df_check[col], ax=ax, kde=True)
  plt.tight_layout() 
plt.show()