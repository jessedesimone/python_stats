#!/usr/local/bin/python3.9

'''module for Ordinary Least Squares (OLS) using statsmodels'''

#import packages
import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# set working dir
os.chdir('/Users/jessedesimone/Desktop')

# read data
df = pd.read_csv('regression_input.csv')

# define X and y
X = df['months_dx_to_v1']
y = df['psp_prob']

# adding constant term
X = sm.add_constant(X)

# plot scatter
#plt.scatter(X,y)
#plt.show()

# fit model
model1=sm.OLS(y, X).fit()

# print summary
print(model1.summary())