#!/usr/local/bin/python3.9
'''
Module for examining variance inflation factors for predictor features in regression model
Multicolinearity between predictors of a regression model are likely present if the VIF is between 5-10
VIF should be considered in parallel with the regression results before dropping variable

'''

# Import packages
import os
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Input data
fpath = '<path/to/datasets/>'
fname = '<file name>'
infile = os.path.join(fpath, fname)
df = pd.read_csv(infile)
df = df._get_numeric_data() #drop non-numeric cols
df.info()

response_var = input('\nSelect response variable: ')        #define response variable
features = df.drop([response_var], axis = 1)      #define predictor variables

# Create empty DataFrame for results
vif_data = pd.DataFrame()

# Calculate VIF for each feature and print to df
vif_data['Features'] = features.columns
vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif_data['VIF'] = round(vif_data['VIF'], 2)
vif_data = vif_data.sort_values(by = 'VIF', ascending=False)
vif_data

print('++Script Complete++')
