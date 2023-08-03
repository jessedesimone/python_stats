#!/usr/local/bin/python3.9

'''
One way ANOVA assumptions
The samples are independent.
Each sample is from a normally distributed population.
The population standard deviations of the groups are all equal. This property is known as homoscedasticity.

If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test (scipy.stats.kruskal) although with some loss of power.

Change Log
==========
0.0.1 (2021-04-08)
----------
Initial commit
'''

# Example 1 Basic
from scipy.stats import f_oneway
High = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735, 0.0659, 0.0923, 0.0836]
Low = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
Placebo = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]

np.mean(High)
np.mean(Low)
np.mean(Placebo)

# F-statistic
stat, p = f_oneway(High, Low, Placebo)
print('F-statistic: ', stat, '\nP-value: ', p)

# Independent samples t-tests
from scipy import stats
stat,p = stats.ttest_ind(High,Low)
print('T-statistic: ', stat, '\nP-value: ', p)

stat,p = stats.ttest_ind(High,Placebo)
print('T-statistic: ', stat, '\nP-value: ', p)

stat,p = stats.ttest_ind(Placebo,Low)
print('T-statistic: ', stat, '\nP-value: ', p)


#Example 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Generate some random numbers
np.random.seed(12345678)
High = np.random.normal(loc=20, scale=1.0, size=100)
Low = np.random.normal(loc=10, scale=1.0, size=100)
Placebo = np.random.normal(loc=8, scale=1.0, size=100)

# Convert to DataFrame
data = np.array([High,Low,Placebo])
df = pd.DataFrame(data=data.transpose(), columns = ['High_dose','Low_dose','Placebo'])
df.info()

# Plot boxplot
df.boxplot(column = ['High_dose','Low_dose','Placebo'], grid=False)
plt.show()

# Get group means
df.mean(axis = 0, skipna=True)

# F statistic
fvalue, pvalue = stats.f_oneway(df['High_dose'], df['Low_dose'], df['Placebo'])
print('F-statistic: ', fvalue, '\nP-value', pvalue)

# Type 2 ANOVA DataFrame
data_vals = (data).reshape(-1,1)[:,0]
data_heads = ['Drug A']*100+['Drug B']*100+['Placebo']*100
data_tot = [[data_heads[num] , i ] for num,i in enumerate(data_vals)]
df_data = pd.DataFrame(data_tot, columns= ['Drug','Motor_Score'])
lm = ols('Motor_Score ~ Drug',data=df_data).fit()
table = sm.stats.anova_lm(lm)
print(table)

# Pairwise comparisons
mc = pairwise_tukeyhsd(df_data['Motor_Score'],df_data['Drug'])
mc_results = mc.summary()
print(mc_results)

# Effect size
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq'] / aov[:]['df']
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / (
                sum(aov['sum_sq']) + aov['mean_sq'][-1])
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov
anova_table(table)