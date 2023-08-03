#!/usr/local/bin/python3.9
'''
Module for examining and plotting correlations

Change Log
==========
0.0.1 (2021-04-08)
----------
Initial commit

'''

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

ROOT = '/Datasets/'        #define base directory
df = pd.read_csv(ROOT + 'FuelConsumption.csv')

# Pearson Correlation Coefficient
np.corrcoef(df['ENGINESIZE'], df['CO2EMISSIONS'])

# Correlation matrix (Pearson or Spearmann)
df.corr(method='pearson')['CO2EMISSIONS'].abs().sort_values(ascending=False)  # can sub spearman for pearson

# Scatter plot
plt.scatter(data = df, x='ENGINESIZE', y='CO2EMISSIONS')

# Scatter plot with fitted line
x = df['ENGINESIZE']
y = df['CO2EMISSIONS']
res = stats.linregress(x, y)
plt.plot(x, y, 'o', label='Data')
plt.plot(x, res.intercept + res.slope * x, 'r', label='Fitted Line')
plt.legend()
plt.show()

# SIMPLE LINEAR REGRESSION PLOT
sns.set(rc={'figure.figsize': (8, 5)})
sns.set(font_scale=1)
sns.set_style("ticks")
X = df['ENGINESIZE']
Y = df['CO2EMISSIONS']
#plt.grid(False)
ax = sns.regplot(X, Y, ci=95, scatter_kws={"color": "black"}, line_kws={"color": "red"})
ax.set_title('Simple Linear Regression Plot', weight='bold')
sns.despine()

# Plot Correlation matrix
sns.set_context('paper')
f, ax = plt.subplots(figsize=(10, 7))
cor = df.corr()
plt.title('Feature Correlation Matrix', weight='bold', fontsize=20)
ax = sns.heatmap(cor, vmax=1, annot=True, cmap=sns.color_palette("coolwarm", 20))
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
# plt.savefig('CorrMat.png')
plt.show()

# define correlation function (spearman or pearson)
# def corrfunc(x, y, **kws):
#     r, p = stats.spearmanr(x, y)  # use stats.pearsonr for pearson
#     ax = plt.gca()
#     ax.annotate("r = {:.2f}".format(r),
#                 xy=(.1, .9), xycoords=ax.transAxes)
#     ax.annotate("p = {:.2f}".format(p),
#                 xy=(.6, .9), xycoords=ax.transAxes)

# Pairgrid plot
# NOTE: very computationally demanding; do not perform with high predictors, use correlation matrix above
mpl.rcParams['axes.labelsize']=5
g = sns.PairGrid(df, palette=["red"], height=1, aspect=1.5)
g.map_upper(plt.scatter, s=1)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="gist_heat")
g.map_lower(corrfunc)
g.map_upper(corrfunc)

'''
DataFrame of correlation (r) and coef. of determination (R^2) 
between predictor(s) and response variable

Loop through each column of dataframe
'''
# Create empty lists to append
slope_list = []
intercept_list = []
rvalue_list = []
r2_value_list = []
pvalue_list = []
stderr_list = []

df = df.drop(labels=['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS',
                     'TRANSMISSION', 'FUELTYPE'], axis = 1)      #drop unwanted columns
y = df['CO2EMISSIONS']          #define the response variable
# Run loop for each column
for i in df.columns[0:]:
    print(i)
    x = df[i]
    res = stats.linregress(x, y)
    slope_list.append(res.slope)
    intercept_list.append(res.intercept)
    rvalue_list.append(res.rvalue)
    r2_value = res.rvalue**2
    r2_value_list.append(r2_value)
    pvalue_list.append(res.pvalue)
    stderr_list.append(res.stderr)

# Create empty dataframe
corr_df = pd.DataFrame(columns=['Variable', 'Slope', 'Intercept', 'r', 'R^2', 'p', 'SE'])
variable_list = df.columns.tolist()

# Append lists to dataframe
corr_df['Variable'] = variable_list
corr_df['Slope'] = slope_list
corr_df['Intercept'] = intercept_list
corr_df['r'] = rvalue_list
corr_df['R^2'] = r2_value_list
corr_df['p'] = pvalue_list
corr_df['SE'] = stderr_list
pd.set_option('display.max_columns', None)
print(corr_df)

# Two-sided inverse Students t-distribution
# p - probability, df - degrees of freedom
x = df['ENGINESIZE']
y = df['CO2EMISSIONS']
res = stats.linregress(x, y)
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))
ts = tinv(0.05, len(x)-2)
print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
print(f"intercept (95%): {res.intercept:.6f}"
f" +/- {ts*res.intercept_stderr:.6f}")
