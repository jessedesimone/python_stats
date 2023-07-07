#!/usr/local/bin/python3.9

#import packages
import pandas as pd
import numpy as np
from scipy.stats import permutation_test
from statsmodels.stats.multitest import fdrcorrection

#read data
df = pd.read_csv('/path/to/perm_test.csv')
df_c=df     #make a copy of df

#subset the dataframe
#df_c=df.filter(regex='var1|var2|var n')

#describe data
d=df_c.groupby(['<grouping var>']).describe()
print(d)
d.to_csv('/path/to/out.csv')

#define statistic for permutation test
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

#permutation test for single variable
'''define variables'''
test='ab42_40'
x = df_c[df_c['grp']=='<sub group e.g. control>']
y = df_c[df_c['grp']=='<sub group eg. treatment>']

'''run permutation test for single variable'''
res = permutation_test((x[test], y[test]), statistic, permutation_type='independent', 
                       n_resamples=10000, alternative='two-sided', random_state=None)
print('test statistic=',res.statistic,'     pvalue=',res.pvalue)

#permutation tests for multiple variables
'''create lists for final dataframe'''
col_list=df_c.columns.to_list()
col_list.remove('grp')
t_list = []
p_unc_list = []

'''permutation test loop'''
loop_cols = df_c.loc[:, df_c.columns!='<grouping var>']
col_names = [col for col in loop_cols]
for col in col_names:
    print('*** running permutation test for ',col)
    x = df_c[df_c['grp']=='<sub group e.g. control>']
    y = df_c[df_c['grp']=='<sub group eg. treatment>']
    res = permutation_test((x[col], y[col]), statistic, vectorized=True, permutation_type='independent', 
                       n_resamples=10000, alternative='two-sided', random_state=None)
    t_list.append(res.statistic)
    p_unc_list.append(res.pvalue)

#FDR correction for multivariate permutation tests
import statsmodels as sm
a=sm.stats.multitest.fdrcorrection(p_unc_list, 
                                          alpha=0.05, 
                                          method='indep', 
                                          is_sorted=False)
fdr_list=a[1].tolist()

#create a final dataframe
'''create dataframe for statistics and uncorrected pvalues'''
d = {'variable':col_list, 't_stat':t_list, 'p_unc':p_unc_list, 'fdr':fdr_list}
df_f = pd.DataFrame(d)
print(df_f)