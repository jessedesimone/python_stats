#!/usr/local/bin/python3.9
"""
Copyright (C) 2021 Jesse DeSimone, Ph.D.

Change Log
=============
0.0.1 (2021-06-28)
-------------
Initial commit

"""
# Import packages
import os
import pandas as pd
from scipy import stats

# Configure directories
ROOT_DIR = '/Users/jessedesimone/DeSimone_Py_Scripts/python_stats'
SUB_DIR = os.path.join(ROOT_DIR, 'Datasets')
os.chdir(SUB_DIR)

# Read data
df = pd.read_csv('FuelConsumptionCo2.csv')

# Correlation between 2 variables
a = df['CO2EMISSIONS']
b = df['ENGINESIZE']
test, p = stats.pearsonr(a,b)
print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)

test, p = stats.spearmanr(a,b)
print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)

# Correlation matrix
df.corr(method='pearson')['CO2EMISSIONS'].abs().sort_values(ascending=False)
df.corr(method='spearman')['CO2EMISSIONS'].abs().sort_values(ascending=False)