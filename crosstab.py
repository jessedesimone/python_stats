#!/usr/local/bin/python3.9
'''

Module to for cross tabulation of two variables

Change Log
==========
0.0.1 (2021-07-07)
----------
Initial commit

'''

import pandas as pd

ROOT = '/Datasets/'
pd.set_option('display.max_columns', 100) #show max of 100 columns
df = pd.read_csv(ROOT + 'TIPS.csv')

#create crosstable of desired variables
tab=pd.crosstab(index=df["sex"],
                columns=df["smoker"],
                margins=True)
tab.columns=["No", "Yes", "Row Total"]
tab.index=["Female", "Male", "Column Total"]
print(tab)

#get the proportions for the rows
print('Proportions by row')
print(tab.div(tab["Row Total"],axis=0))

#get the proportions for the columns
print('Proportions by column')
print(tab/tab.loc["Column Total"])