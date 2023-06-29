#!/usr/local/bin/python3.9

'''
Module to perform Pearson or Spearman correlations for dataframe
'''

# Import packages
import os
import pandas as pd
from scipy import stats

# Configure directories
ROOT_DIR = '<path/to/root/dir>'

# Read data
df = pd.read_csv(ROOT_DIR + '<data file>.csv')

# Correlation matrix
df.corr(method='pearson')['CO2EMISSIONS'].abs().sort_values(ascending=False)
df.corr(method='spearman')['CO2EMISSIONS'].abs().sort_values(ascending=False)

# Correlation between 2 variables
a = df['<variable 1>']
b = df['<variable 2>']

test, p = stats.pearsonr(a,b)
print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)
test, p = stats.spearmanr(a,b)
print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)

#run correlations in loop
'''create lists for final dataframe'''
response_var = 'GM_FW'
col_list=df.columns.to_list()
col_list.remove(response_var)
t_list = []
p_unc_list = []

loop_cols = df.loc[:, df.columns!=response_var]
col_names = [col for col in loop_cols]
for col in col_names:
    print('+++++' + col + '+++++')
    test, p = stats.spearmanr(df[response_var], df[col])
    t_list.append(test)
    p_unc_list.append(p)
    print('Test Statistic : %.3f' % test, 'P-value : %.3f' % p)
    if p < 0.05:
        print('** < 0.05 unc **')
    print('\n')

    #FDR correction for multivariate permutation tests
import statsmodels as sm
from statsmodels.stats.multitest import fdrcorrection
a=sm.stats.multitest.fdrcorrection(p_unc_list, 
                                          alpha=0.05, 
                                          method='indep', 
                                          is_sorted=False)
fdr_list=a[1].tolist()

#create a final dataframe
'''create dataframe for statistics and uncorrected pvalues'''
d = {'variable':col_list, 'corr_coef':t_list, 'p_unc':p_unc_list, 'fdr':fdr_list}
df_fdr = pd.DataFrame(d)
print(df_fdr)
df_fdr.to_csv('/Users/jessedesimone/Desktop/fdr.csv')


