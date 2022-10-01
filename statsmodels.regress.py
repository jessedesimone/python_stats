#!/usr/local/bin/python3.9

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

# Read the given CSV file, and view some sample records
inpath = '/Users/jessedesimone/Desktop/'
df = pd.read_csv(inpath + 'data.csv')
df.info()
df.describe()

#define variables
X='month'
y='savings'

X_var = df[X]
y_var = df[y]

# Adding a constant to get an intercept
X_var_sm = sm.add_constant(X_var)

# Visualize the data wrt to Sales
sns.pairplot(df, x_vars=X, y_vars=y, height=4, aspect=1, kind='scatter')
plt.show()

# Visualize the data using heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()

# Fitting the resgression line using 'OLS'
mod = sm.OLS(y_var, X_var_sm).fit()

# Summary of the regression output
mod.summary()

# Model parameters
params=mod.params
print(params)

# Residual analysis
# Predicting y_value using traingn data of X
y_pred = mod.predict(X_var_sm)

# Creating residuals from the y_train data and predicted y_data
res = (y_var - y_pred)

# Predict future value
X_new=np.array(1,12)    # "1" refers to the intercept term
prediction=mod.get_prediction(X_new)
prediction.summary_frame()
